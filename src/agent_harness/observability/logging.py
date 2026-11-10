"""JSON logs. The formatter redacts credential-shaped text."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime

from agent_harness.safety.redaction import redact


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": redact(record.getMessage()),
            "correlation_id": getattr(record, "correlation_id", None),
            "session_id": getattr(record, "session_id", None),
            "task_id": getattr(record, "task_id", None),
        }
        if record.exc_info:
            payload["error_type"] = record.exc_info[0].__name__ if record.exc_info[0] else None
        return json.dumps(payload, ensure_ascii=True)


def configure_logging(level: str) -> logging.Logger:
    logger = logging.getLogger("agent_harness")
    logger.setLevel(level)
    if not any(getattr(handler, "_agent_harness", False) for handler in logger.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())
        handler._agent_harness = True  # type: ignore[attr-defined]
        logger.addHandler(handler)
    logger.propagate = False
    return logger


class LogContext(logging.LoggerAdapter):
    def process(self, msg: str, kwargs: dict) -> tuple[str, dict]:
        extra = kwargs.setdefault("extra", {})
        for key in ("correlation_id", "session_id", "task_id"):
            extra.setdefault(key, self.extra.get(key))
        return msg, kwargs
