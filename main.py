import os
import sys
import warnings
from datetime import UTC, datetime

warnings.filterwarnings("ignore")

from torch.multiprocessing import set_start_method

from src.logger.logger import logger
from src.resampling.nested_resampling import run_nested_resampling
from src.tui.tui_configurator import run_tui_configurator
from src.tui.tui_screen import TUI
from src.utils.checkpoint_manager import GaState, checkpoint_manager
from src.utils.file_helper import ensure_dir_exists
from src.utils.signal_manager import (
    signal_manager,
)


def main():
    try:
        # Ensure spawn, fork isn't supported for CUDA
        if sys.platform != "win32":
            set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    def sigint_handler(signum, frame):
        logger.info("Main received SIGINT, shutting down gracefully...")
        signal_manager.handle_signal(signum, frame)

    signal_manager.initialize()

    tui = TUI()
    loaded_state: GaState | None = None

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

            log_dir = "logs"
            ensure_dir_exists(log_dir)
            timestamp = datetime.now(UTC).strftime("%Y-%m-%d_%H-%M-%S")
            session_log_filename = os.path.join(log_dir, f"log_{timestamp}.log")

            logger.add_file_sink(session_log_filename)
        logger.add_tui_sink(tui.get_loguru_sink())

        # tui.build_config_panel(config)

        with tui:
            logger.info(
                f"Logger initialized. Log file at {session_log_filename}"
            )
            run_nested_resampling(
                config, tui, session_log_filename, loaded_state
            )

            logger.info("Optimization complete.")
    except KeyboardInterrupt:
        logger.info("User terminated the program.")
        sys.exit(0)
    except SystemExit:
        logger.info("Program terminated gracefully.")
    except Exception as e:
        logger.exception(f"Unexpected error occurred {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
