import os
import sys
import warnings
from datetime import datetime, timezone
from typing import Optional

warnings.filterwarnings("ignore")

from torch.multiprocessing import set_start_method

from src.logger.logger import logger
from src.resampling.nested_resampling import run_nested_resampling
from src.tui.tui_configurator import run_tui_configurator
from src.tui.tui_screen import TUI
from src.utils.checkpoint_manager import GaState, checkpoint_manager
from src.utils.file_helper import ensure_dir_exists
from src.utils.signal_manager import signal_manager
from torch import set_num_threads, set_num_interop_threads
from torchvision import datasets
import logging
import multiprocessing
from logging.handlers import QueueListener


def start_logging_listener_in_main(log_file_path: str):
    """
    Sets up the central multiprocessing log queue/listener for worker logging.
    Returns:
        log_queue: The multiprocessing.Queue for log records.
        listener: The QueueListener instance (call .stop() when done).
    """
    log_queue = multiprocessing.Queue(-1)
    file_handler = logging.FileHandler(log_file_path, mode="a", encoding="utf-8")
    formatter = logging.Formatter("%(message)s")
    file_handler.setFormatter(formatter)
    listener = QueueListener(log_queue, file_handler)
    listener.start()
    return log_queue, listener


def prepare_cifar10_data():
    base = os.environ.get("SLURM_JOB_ID")
    array_id = os.environ.get("SLURM_ARRAY_TASK_ID", "0")
    tmpdir = f"./data"
    os.makedirs(tmpdir, exist_ok=True)

    logger.info(f"Downloading CIFAR-10 to {tmpdir} ...")
    datasets.CIFAR10(root=tmpdir, train=True, download=True)
    datasets.CIFAR10(root=tmpdir, train=False, download=True)
    logger.info("CIFAR-10 downloading completed.")


def main():
    try:
        # Ensure spawn (fork isn't supported for CUDA)
        if sys.platform != "win32":
            set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    def sigint_handler(signum, frame):
        logger.info("Main received SIGINT, shutting down gracefully...")
        signal_manager.handle_signal(signum, frame)

    signal_manager.initialize()

    tui = TUI()
    loaded_state: Optional[GaState] = None

    if checkpoint_manager.is_checkpoint_exists():
        loaded_state = checkpoint_manager.load_checkpoint()

    try:
        if loaded_state:
            config = loaded_state.config
            session_log_filename = loaded_state.session_log_filename
            logger.add_file_sink(session_log_filename)
        else:
            config = run_tui_configurator()
            if not config:
                return

            task_id = os.environ.get("SLURM_ARRAY_TASK_ID", "0")
            log_dir = f"/lustre/pd01/hpc-adamak7184-1759856296/logs/{task_id}"
            ensure_dir_exists(log_dir)
            timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
            session_log_filename = os.path.join(
                log_dir, f"log_{task_id}_{timestamp}.log"
            )

            logger.add_file_sink(session_log_filename)

        logger.add_tui_sink(tui.get_loguru_sink())

        with tui:
            logger.info(
                f"Logger initialized. Log file at {session_log_filename}"
            )
            prepare_cifar10_data()
            run_nested_resampling(
                config, tui, session_log_filename, loaded_state, log_queue=None
            )
            logger.info("Optimization complete.")

    except KeyboardInterrupt:
        logger.info("User terminated the program.")
        sys.exit(0)
    except SystemExit:
        logger.info("Program terminated gracefully.")
    except Exception as e:
        logger.exception(f"Unexpected error occurred: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
