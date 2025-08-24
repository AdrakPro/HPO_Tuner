import sys
from pathlib import Path
from src.config.settings import ex
from src.tui import run_tui_configurator, print_final_config_panel
from src.logger.experiment_logger import logger

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))


@ex.main
def run_optimization(_config, _run):
    print_final_config_panel(_config)

    # ... (optimization logic) ...
    logger.info("1. Inicjalizacja populacji...")
    # ...
    logger.success("Optymalizacja zakończona.")


def main():
    try:
        config_overrides = run_tui_configurator()

        if config_overrides is not None:
            # loglevel: supress Sacred built-in logger
            ex.run(
                config_updates=config_overrides, options={"--loglevel": "ERROR"}
            )
    except KeyboardInterrupt:
        logger.error("\nUżytkownik zakończył działanie programu.")
        sys.exit(0)


if __name__ == "__main__":
    main()
