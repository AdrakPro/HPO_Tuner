"""
Loguru-based logger for the project.
Provides synchronized file and console logging suitable for multiprocessing.
"""

import sys
from typing import Callable

from loguru import logger as loguru_logger


class _Logger:
    """
    A configurable logger using Loguru that supports synchronized file
    and console outputs and can be safely used across multiple processes.
    """

    def __init__(
        self,
        console: bool = True,
    ):
        """
        Initializes and configures the logger.

        Args:
            console (bool): If True, logs will also be printed to the console.
        """
        self.logger = loguru_logger
        self.logger.remove()

        self._console_sink_id = None
        self._file_sink_id = None

        if console and sys.stdout.isatty():
            self._console_sink_id = self.logger.add(
                sys.stderr,
                level="INFO",
                format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
                filter=lambda record: not record["extra"].get(
                    "file_only", False
                ),
                colorize=True,
            )

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
