import threading
from enum import Enum, auto

from rich.console import Console
from rich.style import Style
from datetime import datetime
import os


class LogLevel(Enum):
    INFO = auto()
    WARNING = auto()
    ERROR = auto()
    SUCCESS = auto()
    NORMAL = auto()


class ExperimentLogger:
    """
    A singleton logger class for experiments with Rich formatting.
    Prints colored messages to the console and writes styled logs to a file.
    """

    _instance: "ExperimentLogger" = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls) -> "ExperimentLogger":
        """
        Returns the singleton instance of ExperimentLogger.
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(ExperimentLogger, cls).__new__(cls)
                    cls._instance._init_logger()
        return cls._instance

    def _init_logger(self) -> None:
        """
        Initializes console, styles, and log file with datetime in its name.
        """
        self.console: Console = Console(highlight=False)
        self.styles: dict[LogLevel, Style] = {
            LogLevel.SUCCESS: Style(color="green", bold=True),
            LogLevel.INFO: Style(color="cyan"),
            LogLevel.WARNING: Style(color="yellow", bold=True),
            LogLevel.ERROR: Style(color="red", bold=True),
            LogLevel.NORMAL: Style(color=None),
        }

        dt_str: str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        logs_dir: str = "logs"
        self.log_path = os.path.abspath(f"{logs_dir}/log_{dt_str}.log")

        os.makedirs(logs_dir, exist_ok=True)
        self.success(
            f"Logger został zainicjalizowany. Ścieżka plików log: {self.log_path}"
        )

    def log_console(self, msg: str, level: LogLevel = LogLevel.INFO) -> None:
        """
        Prints a message to the console with Rich formatting.

        Args:
            level (LogLevel): One of "success", "info", "warning", "error", "normal".
            msg (str): Message to print.
        """
        style: Style | None = self.styles.get(level, None)
        self.console.print(f"[{level.name}] {msg}", style=style)

    def log_file(self, msg: str, level: LogLevel = LogLevel.INFO) -> None:
        """
        Writes a message to the log file, including level and timestamp.

        Args:
            level (LogLevel): One of "success", "info", "warning", "error", "normal".
            msg (str): Message to write.
        """
        now: str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        style_tag: str = f"[{level.name}]"
        log_line: str = f"{now} {style_tag} {msg}\n"
        with open(self.log_path, "a") as f:
            f.write(log_line)

    def success(self, msg: str) -> None:
        """
        Print a success message (green, bold).

        Args:
            msg (str): Message to log.
        """
        self.log_console(msg, LogLevel.SUCCESS)
        self.log_file(msg, LogLevel.SUCCESS)

    def info(self, msg: str) -> None:
        """
        Print an informational message (cyan).

        Args:
            msg (str): Message to log.
        """
        self.log_console(msg, LogLevel.INFO)
        self.log_file(msg, LogLevel.INFO)

    def warning(self, msg: str) -> None:
        """
        Print a warning message (yellow, bold).

        Args:
            msg (str): Message to log.
        """
        self.log_console(msg, LogLevel.WARNING)
        self.log_file(msg, LogLevel.WARNING)

    def error(self, msg: str) -> None:
        """
        Print an error message (red, bold).

        Args:
            msg (str): Message to log.
        """
        self.log_console(msg, LogLevel.ERROR)
        self.log_file(msg, LogLevel.ERROR)


# Initialization of single instance of logger
logger = ExperimentLogger()
