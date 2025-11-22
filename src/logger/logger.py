"""
Loguru-based logger for the project.
Provides synchronized file and console logging suitable for multiprocessing.
"""

import sys
from typing import Callable

from loguru import logger as loguru_logger


class StreamToLogger:
    """
    Fake file-like stream object that redirects writes to a logger instance.
    Used to capture sys.stderr and send it to the logger.
    """

    def __init__(self, logger_instance, level="ERROR"):
        self.logger = logger_instance
        self.level = level

    def write(self, buffer):
        """
        Writes the buffer to the logger.
        Splits lines to ensure each stderr line becomes a log entry.
        """
        for line in buffer.rstrip().splitlines():
            if line.strip():
                self.logger.opt(depth=1, exception=False).log(
                    self.level, line.rstrip()
                )

    def flush(self):
        """
        Flush is required for file-like objects, but we log immediately.
        """
        pass


class _Logger:
    """
    A configurable logger using Loguru that supports synchronized file
    and console outputs and can be safely used across multiple processes.
    """

    def __init__(
        self,
        console: bool = True,
        capture_stderr: bool = False,
    ):
        """
        Initializes and configures the logger.

        Args:
            console (bool): If True, logs will also be printed to the console.
            capture_stderr (bool): If True, redirects sys.stderr to the logger.
        """
        self.logger = loguru_logger
        self.logger.remove()

        self._console_sink_id = None
        self._file_sink_id = None

        if console:
            target = sys.__stderr__ if sys.__stderr__ else sys.stderr

            if sys.stdout.isatty():
                self._console_sink_id = self.logger.add(
                    target,
                    level="INFO",
                    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
                    filter=lambda record: not record["extra"].get(
                        "file_only", False
                    ),
                    colorize=True,
                )

        if capture_stderr:
            # Redirect standard error to our logger
            sys.stderr = StreamToLogger(self.logger, "ERROR")

    def add_file_sink(self, log_file_path: str) -> None:
        """Adds a synchronized file sink to the logger. Can only be called once."""
        if self._file_sink_id is not None:
            return

        self._file_sink_id = self.logger.add(
            log_file_path,
            level="INFO",
            enqueue=True,
            backtrace=True,
            diagnose=True,
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {message}",
        )

    def add_tui_sink(self, sink: Callable[[str], None]):
        """
        Redirects console output into the TUI logs panel.
        """
        if self._console_sink_id is not None:
            self.logger.remove(self._console_sink_id)
            self._console_sink_id = None

        self._console_sink_id = self.logger.add(
            sink,
            level="INFO",
            format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
            colorize=True,
            filter=lambda record: not record["extra"].get("file_only", False),
        )

    def info(self, msg: str, file_only: bool = False):
        """Logs an info message."""
        if file_only:
            self.logger.bind(file_only=True).info(msg)
        else:
            self.logger.info(msg)

    def warning(self, msg: str, file_only: bool = False):
        """Logs a warning message."""
        if file_only:
            self.logger.bind(file_only=True).warning(msg)
        else:
            self.logger.warning(msg)

    def error(self, msg: str, file_only: bool = False):
        """Logs an error message."""
        if file_only:
            self.logger.bind(file_only=True).error(msg)
        else:
            self.logger.error(msg)

    def exception(self, msg: str, file_only: bool = False):
        """Logs an error message."""
        if file_only:
            self.logger.bind(file_only=True).exception(msg)
        else:
            self.logger.exception(msg)


# Global instance for the main process
logger = _Logger()
