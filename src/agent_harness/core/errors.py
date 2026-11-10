"""Structured errors raised across the harness boundary."""

from __future__ import annotations


class HarnessError(Exception):
    """An error that can be shown to an API caller without internal details."""

    def __init__(self, code: str, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code


class ToolError(Exception):
    """A tool failed. Transient errors may be retried only for idempotent tools."""

    def __init__(self, message: str, *, transient: bool = False) -> None:
        super().__init__(message)
        self.transient = transient


class ToolTimeoutError(ToolError):
    def __init__(self, message: str) -> None:
        super().__init__(message, transient=True)


class InputError(HarnessError):
    def __init__(self, message: str) -> None:
        super().__init__("invalid_input", message, 400)


class OutputError(HarnessError):
    def __init__(self, message: str) -> None:
        super().__init__("invalid_output", message, 500)
