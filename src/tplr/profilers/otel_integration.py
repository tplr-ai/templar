"""
OpenTelemetry integration for profilers.

Provides telemetry metrics export functionality for profiler data.
"""

import os
from typing import Dict, Any, Optional

from opentelemetry import metrics, trace
from opentelemetry.metrics import get_meter_provider
from opentelemetry.trace import get_tracer_provider
from opentelemetry.sdk.metrics import MeterProvider as SDKMeterProvider
from opentelemetry.sdk.trace import TracerProvider as SDKTracerProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.resources import Resource
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter


class OpenTelemetryProfilerIntegration:
    """
    OpenTelemetry integration for profiler metrics.

    Handles creation and management of OTEL metrics for profiler data.
    """

    def __init__(self, service_name: str = None):
        """
        Initialize OpenTelemetry integration.

        Args:
            service_name: Service name for telemetry identification
        """
        if service_name is None:
            # Auto-detect service name from environment or script name
            service_name = self._detect_service_name()
        self.service_name = service_name
        self.enabled = self._should_enable()
        self.meter: Optional[metrics.Meter] = None
        self.tracer: Optional[trace.Tracer] = None
        self.instruments: Dict[str, Any] = {}

        if self.enabled:
            self._setup_meter()
            self._setup_tracer()

    def _detect_service_name(self) -> str:
        """Auto-detect service name from environment or process."""
        # Try to get from environment first
        service_name = os.environ.get("OTEL_SERVICE_NAME")
        if service_name:
            return service_name
        
        # Try to detect from script name
        import sys
        script_name = os.path.basename(sys.argv[0]) if sys.argv else "unknown"
        
        if "validator" in script_name:
            return "tplr-validator"
        elif "miner" in script_name:
            return "tplr-miner"
        elif "aggregator" in script_name:
            return "tplr-aggregator"
        else:
            return "tplr-profiler"

    def _should_enable(self) -> bool:
        """Check if OpenTelemetry should be enabled based on environment."""
        return os.environ.get("TPLR_ENABLE_OTEL_PROFILING", "0") == "1"

    def _setup_meter(self) -> None:
        """Setup OpenTelemetry meter and instruments."""
        if not self.enabled:
            return

        # Get or create meter provider
        provider = get_meter_provider()
        if not isinstance(provider, SDKMeterProvider):
            # If no provider is configured, create a basic one with service resource
            endpoint = os.environ.get(
                "OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317"
            )
            exporter = OTLPMetricExporter(endpoint=endpoint)
            reader = PeriodicExportingMetricReader(
                exporter, export_interval_millis=30000
            )
            
            # Create resource with service name
            resource = Resource.create({"service.name": self.service_name})
            provider = SDKMeterProvider(metric_readers=[reader], resource=resource)
            metrics.set_meter_provider(provider)

        self.meter = provider.get_meter(
            f"{self.service_name}.profiler", "1.0.0"
        )

        # Create metric instruments
        self._create_instruments()

    def _setup_tracer(self) -> None:
        """Setup OpenTelemetry tracer."""
        if not self.enabled:
            return

        # Get or create tracer provider
        provider = get_tracer_provider()
        if not isinstance(provider, SDKTracerProvider):
            # If no provider is configured, create a basic one with service resource
            endpoint = os.environ.get(
                "OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317"
            )
            span_exporter = OTLPSpanExporter(endpoint=endpoint)
            span_processor = BatchSpanProcessor(
                span_exporter,
                max_queue_size=512,
                schedule_delay_millis=1000,  # Export every 1 second
                max_export_batch_size=64
            )
            
            # Create resource with service name
            resource = Resource.create({"service.name": self.service_name})
            provider = SDKTracerProvider(resource=resource)
            provider.add_span_processor(span_processor)
            trace.set_tracer_provider(provider)

        self.tracer = provider.get_tracer(
            f"{self.service_name}.profiler", "1.0.0"
        )

    def _create_instruments(self) -> None:
        """Create OpenTelemetry metric instruments."""
        if not self.meter:
            return

        # Timer profiler instruments
        self.instruments.update(
            {
                "function_duration": self.meter.create_histogram(
                    name="profiler_function_duration_seconds",
                    description="Function execution duration in seconds",
                    unit="s",
                ),
                "function_call_count": self.meter.create_counter(
                    name="profiler_function_calls_total",
                    description="Total number of function calls",
                ),
                "function_error_count": self.meter.create_counter(
                    name="profiler_function_errors_total",
                    description="Total number of function errors",
                ),
                # Shard profiler instruments
                "shard_read_duration": self.meter.create_histogram(
                    name="profiler_shard_read_duration_seconds",
                    description="Shard read operation duration in seconds",
                    unit="s",
                ),
                "shard_read_count": self.meter.create_counter(
                    name="profiler_shard_reads_total",
                    description="Total number of shard read operations",
                ),
                "shard_file_size": self.meter.create_histogram(
                    name="profiler_shard_file_size_bytes",
                    description="Shard file size in bytes",
                    unit="byte",
                ),
                "shard_row_count": self.meter.create_histogram(
                    name="profiler_shard_rows_processed",
                    description="Number of rows processed per shard operation",
                ),
            }
        )

    def record_function_timing(
        self,
        function_name: str,
        duration: float,
        error: bool = False,
        attributes: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Record function timing metrics.

        Args:
            function_name: Name of the function
            duration: Execution duration in seconds
            error: Whether an error occurred
            attributes: Additional attributes for the metric
        """
        if not self.enabled or not self.instruments:
            return

        base_attrs = {"function_name": function_name}
        if attributes:
            base_attrs.update(attributes)

        # Record duration histogram
        self.instruments["function_duration"].record(duration, attributes=base_attrs)

        # Record call count
        self.instruments["function_call_count"].add(1, attributes=base_attrs)

        # Record error count if applicable
        if error:
            self.instruments["function_error_count"].add(1, attributes=base_attrs)

    def create_span(self, operation_name: str, attributes: Optional[Dict[str, str]] = None):
        """
        Create a trace span for an operation.

        Args:
            operation_name: Name of the operation
            attributes: Additional attributes for the span

        Returns:
            Span context manager or None if tracing is disabled
        """
        if not self.enabled or not self.tracer:
            print(f"[OTEL DEBUG] Tracing disabled: enabled={self.enabled}, tracer={self.tracer is not None}")
            return None

        span_attrs = attributes or {}
        print(f"[OTEL DEBUG] Creating span '{operation_name}' for service '{self.service_name}'")
        return self.tracer.start_as_current_span(
            operation_name, attributes=span_attrs
        )

    def record_shard_metrics(
        self,
        shard_path: str,
        duration: float,
        file_size: Optional[int] = None,
        row_count: Optional[int] = None,
        attributes: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Record shard operation metrics.

        Args:
            shard_path: Path to the shard file
            duration: Read operation duration in seconds
            file_size: Size of the shard file in bytes
            row_count: Number of rows processed
            attributes: Additional attributes for the metric
        """
        if not self.enabled or not self.instruments:
            return

        base_attrs = {"shard_path": shard_path}
        if attributes:
            base_attrs.update(attributes)

        # Record read duration
        self.instruments["shard_read_duration"].record(duration, attributes=base_attrs)

        # Record read count
        self.instruments["shard_read_count"].add(1, attributes=base_attrs)

        # Record file size if available
        if file_size is not None and isinstance(file_size, (int, float)):
            self.instruments["shard_file_size"].record(file_size, attributes=base_attrs)

        # Record row count if available
        if row_count is not None:
            self.instruments["shard_row_count"].record(row_count, attributes=base_attrs)


# Global instance
_otel_integration: Optional[OpenTelemetryProfilerIntegration] = None


def get_otel_integration() -> OpenTelemetryProfilerIntegration:
    """Get or create the global OpenTelemetry integration instance."""
    global _otel_integration
    if _otel_integration is None:
        _otel_integration = OpenTelemetryProfilerIntegration()
    return _otel_integration


def is_otel_enabled() -> bool:
    """Check if OpenTelemetry integration is enabled and available."""
    return get_otel_integration().enabled
