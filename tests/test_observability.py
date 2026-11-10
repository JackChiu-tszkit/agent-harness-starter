import io
import logging

from agent_harness.observability.logging import JsonFormatter
from agent_harness.runtime import build_runtime
from tests.conftest import make_settings


def test_traces_metrics_and_logs_omit_user_content(tmp_path) -> None:
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JsonFormatter())
    logger = logging.getLogger("agent_harness")
    logger.addHandler(handler)
    runtime = build_runtime(make_settings(tmp_path))
    try:
        session = runtime.create_session()
        result = runtime.run(
            session.id,
            "What is 125 multiplied by 48?",
            correlation_id="corr-observe",
        )
        assert result.status == "completed"
        spans = runtime.telemetry.span_exporter.spans
        names = {span.name for span in spans}
        assert {"agent.task", "agent.model", "agent.tool"} <= names
        rendered = " ".join(str(span.attributes) for span in spans)
        assert "125 multiplied" not in rendered
        assert "corr-observe" in rendered
        metrics = _metric_names(runtime.telemetry.metric_reader.get_metrics_data())
        assert "agent.task.duration" in metrics
        assert "agent.tool.calls" in metrics
    finally:
        logger.removeHandler(handler)
        runtime.close()
    logged = stream.getvalue()
    assert "corr-observe" in logged
    assert "125 multiplied" not in logged
    assert "task started" in logged


def _metric_names(data) -> set[str]:
    names = set()
    if data is None:
        return names
    for resource in data.resource_metrics:
        for scope in resource.scope_metrics:
            for metric in scope.metrics:
                names.add(metric.name)
    return names
