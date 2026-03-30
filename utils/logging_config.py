import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

LOG_FORMAT = "[%(levelname)s] %(asctime)s: %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def _make_file_handler(log_file: Path, level: int = logging.INFO) -> logging.FileHandler:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(log_file)
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(fmt=LOG_FORMAT, datefmt=LOG_DATE_FORMAT))
    return handler


def configure_root_logging(log_file: Path | None = None, level: int = logging.INFO) -> None:
    """Configure root logging for a run (console + optional file)."""
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file is not None:
        handlers.append(_make_file_handler(log_file, level=level))

    logging.basicConfig(
        level=level,
        format=LOG_FORMAT,
        datefmt=LOG_DATE_FORMAT,
        handlers=handlers,
        force=True,
    )


@contextmanager
def component_file_logging(log_file: Path | None, level: int = logging.INFO) -> Iterator[None]:
    """Temporarily add a file handler for component-scoped logging."""
    if log_file is None:
        yield
        return

    root_logger = logging.getLogger()
    handler = _make_file_handler(log_file, level=level)
    root_logger.addHandler(handler)
    try:
        yield
    finally:
        root_logger.removeHandler(handler)
        handler.close()


def log_raw_line(text: str) -> None:
    root = logging.getLogger()
    for handler in root.handlers:
        stream = getattr(handler, "stream", None)
        if stream is not None:
            stream.write(text + "\n")
            handler.flush()
