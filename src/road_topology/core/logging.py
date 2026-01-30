"""Structured logging setup for Road Topology Segmentation."""
from __future__ import annotations

import logging
import sys
from contextlib import contextmanager
from pathlib import Path
from time import perf_counter
from typing import Any, Generator

import structlog
from structlog.types import Processor


def setup_logging(
    level: str = "INFO",
    format_type: str = "console",
    log_file: Path | None = None,
) -> structlog.BoundLogger:
    """Configure structured logging.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR).
        format_type: Output format - "console" for human-readable, "json" for structured.
        log_file: Optional file path for logging.

    Returns:
        Configured bound logger.
    """
    # Set log level
    log_level = getattr(logging, level.upper(), logging.INFO)

    # Configure processors
    shared_processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    if format_type == "json":
        # JSON format for production
        processors = shared_processors + [
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ]
    else:
        # Console format for development
        processors = shared_processors + [
            structlog.dev.ConsoleRenderer(colors=True),
        ]

    # Configure structlog
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=log_level,
    )

    # Add file handler if specified
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        logging.getLogger().addHandler(file_handler)

    return structlog.get_logger()


def get_logger(name: str) -> structlog.BoundLogger:
    """Get a logger with the given name.

    Args:
        name: Logger name (typically __name__).

    Returns:
        Bound logger instance.
    """
    return structlog.get_logger(name)


@contextmanager
def log_duration(
    logger: structlog.BoundLogger,
    operation: str,
    **extra: Any,
) -> Generator[None, None, None]:
    """Context manager to log operation duration.

    Args:
        logger: Logger instance.
        operation: Name of the operation being timed.
        **extra: Additional context to include in logs.

    Yields:
        None

    Example:
        with log_duration(logger, "process_video", video_path=str(path)):
            process(path)
    """
    logger.info(f"Starting {operation}", **extra)
    start = perf_counter()
    try:
        yield
    except Exception as e:
        duration = perf_counter() - start
        logger.error(
            f"Failed {operation}",
            duration_seconds=round(duration, 3),
            error=str(e),
            **extra,
        )
        raise
    else:
        duration = perf_counter() - start
        logger.info(
            f"Completed {operation}",
            duration_seconds=round(duration, 3),
            **extra,
        )


def bind_context(**context: Any) -> None:
    """Bind context variables for all subsequent log calls.

    Args:
        **context: Context key-value pairs to bind.

    Example:
        bind_context(video_id="abc123", frame_count=1000)
    """
    structlog.contextvars.bind_contextvars(**context)


def clear_context() -> None:
    """Clear all bound context variables."""
    structlog.contextvars.clear_contextvars()
