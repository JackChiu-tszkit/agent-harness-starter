"""OpenTelemetry traces and metrics owned by this process.

`console` prints spans locally. `memory` keeps them for tests. `none` records
metrics but does not export spans. No hosted collector is required.
"""

from __future__ import annotations

from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    ConsoleSpanExporter,
    SimpleSpanProcessor,
    SpanExporter,
    SpanExportResult,
)
from opentelemetry.trace import Tracer


class MemorySpanExporter(SpanExporter):
    """In-process span buffer. Used when a paid or remote collector is not configured."""

    def __init__(self) -> None:
        self.spans: list = []

    def export(self, spans) -> SpanExportResult:  # type: ignore[no-untyped-def]
        self.spans.extend(spans)
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        return None


class Telemetry:
    def __init__(self, exporter_name: str) -> None:
        self.exporter_name = exporter_name
        self.provider = TracerProvider()
        self.span_exporter: MemorySpanExporter | None = None
        if exporter_name == "console":
            self.provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
        elif exporter_name == "memory":
            self.span_exporter = MemorySpanExporter()
            self.provider.add_span_processor(SimpleSpanProcessor(self.span_exporter))
        self.tracer: Tracer = self.provider.get_tracer("agent_harness")
        self.metric_reader = InMemoryMetricReader()
        self.meter_provider = MeterProvider(metric_readers=[self.metric_reader])
        meter = self.meter_provider.get_meter("agent_harness")
        self.task_duration = meter.create_histogram("agent.task.duration", unit="ms")
        self.tool_calls = meter.create_counter("agent.tool.calls")
        self.errors = meter.create_counter("agent.errors")

    def shutdown(self) -> None:
        self.provider.shutdown()
        self.meter_provider.shutdown()
