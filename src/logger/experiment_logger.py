import sys
import threading
import queue
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


class LogMessage:
    def __init__(self, msg: str, level: LogLevel, console: bool):
        self.msg = msg
        self.level = level
        self.timestamp = datetime.now()
        self.console = console


class ExperimentAsyncLogger:
    """
    Asynchronous, singleton logger for experiments with Rich formatting.
    Multiple threads enqueue messages; a background thread writes to file and console.
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls, max_queue_size=100):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._init_logger(max_queue_size)
        return cls._instance

    def _init_logger(self, max_queue_size):
        self.console_enabled = sys.stdout.isatty()
        self.console = Console(highlight=False)
        self.styles = {
            LogLevel.SUCCESS: Style(color="green", bold=True),
            LogLevel.INFO: Style(color="cyan"),
            LogLevel.WARNING: Style(color="yellow", bold=True),
            LogLevel.ERROR: Style(color="red", bold=True),
            LogLevel.NORMAL: Style(color=None),
        }

        dt_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        logs_dir = "logs"
        self.log_path = os.path.abspath(f"{logs_dir}/log_{dt_str}.log")
        os.makedirs(logs_dir, exist_ok=True)
        self._file_handle = open(self.log_path, "a", buffering=1)

        self._msg_queue = queue.Queue(maxsize=max_queue_size)
        self._shutdown_event = threading.Event()
        self._worker_thread = threading.Thread(target=self._worker, daemon=True)
        self._worker_thread.start()
        self.success(
            f"Logger initialized. Log file path: {self.log_path}"
        )

    def _worker(self):
        while not self._shutdown_event.is_set() or not self._msg_queue.empty():
            try:
                msg_obj: LogMessage = self._msg_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            self._write_file(msg_obj)
            if msg_obj.console:
                self._write_console(msg_obj)
            self._msg_queue.task_done()

    def _write_file(self, msg_obj: LogMessage):
        now = msg_obj.timestamp.strftime("%Y-%m-%d %H:%M:%S")
        style_tag = f"[{msg_obj.level.name}]"
        log_line = f"{now} {style_tag} {msg_obj.msg}\n"
        self._file_handle.write(log_line)
        self._file_handle.flush()

    def _write_console(self, msg_obj: LogMessage):
        if self.console_enabled:
            style = self.styles.get(msg_obj.level, None)
            self.console.print(
                f"[{msg_obj.level.name}] {msg_obj.msg}", style=style
            )

    def _enqueue(self, msg: str, level: LogLevel, console: bool = True):
        try:
            msg_obj = LogMessage(msg, level, console)
            self._msg_queue.put(msg_obj, block=False)
        except queue.Full:
            self.warning("The log queue is full!")
            pass

    def success(self, msg: str):
        self._enqueue(msg, LogLevel.SUCCESS)

    def info(self, msg: str):
        self._enqueue(msg, LogLevel.INFO)

    def warning(self, msg: str):
        self._enqueue(msg, LogLevel.WARNING)

    def error(self, msg: str):
        self._enqueue(msg, LogLevel.ERROR)

    def file_only(self, msg: str, level: LogLevel = LogLevel.INFO):
        self._enqueue(msg, level, console=False)

    def close(self, timeout=5.0):
        """
        Signal shutdown, wait for queue to empty, close file.
        """
        self._shutdown_event.set()
        self._worker_thread.join(timeout)

        # Drain any remaining messages (in case of timeout)
        while not self._msg_queue.empty():
            try:
                msg_obj = self._msg_queue.get_nowait()
                self._write_file(msg_obj)
                self._write_console(msg_obj)
                self._msg_queue.task_done()
            except queue.Empty:
                break
        self._file_handle.close()


# Initialization of single instance of logger
logger = ExperimentAsyncLogger()
