"""
Loguru-based logger for the project.
Provides synchronized file and console logging suitable for multiprocessing.
"""

import sys

from loguru import logger as loguru_logger


# TODO: After completion of optimization, change name log.log to unique name
class Logger:
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

        log_file_path = f"logs/log.log"
        self.logger.add(
            log_file_path,
            level="INFO",
            enqueue=True,
            backtrace=True,
            diagnose=True,
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {message}",
        )

        if console and sys.stdout.isatty():
            self.logger.add(
                sys.stderr,
                level="INFO",
                format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
                filter=lambda record: not record["extra"].get(
                    "file_only", False
                ),
                colorize=True,
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
logger = Logger()
