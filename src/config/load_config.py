import json
import os
from datetime import datetime
from typing import Dict

from src.logger.experiment_logger import logger


def prompt_and_load_json_config(
    default_config: Dict, console, config_dir: str
) -> Dict:
    """Asks user to load a config from JSON, looking inside CONFIG_DIR."""
    while True:
        filename = console.input(
            f"Podaj nazwę pliku konfiguracji w folderze '{config_dir}': "
        )

        if not filename:
            logger.error("Nazwa pliku nie może być pusta. Spróboj ponownie.")
            continue

        # Temporary for testing purposes (filename)
        path = os.path.join(config_dir, "config_2025-08-25_20-43-00.json")
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    logger.success(f"Wczytano konfigurację z {path}")
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                logger.error(f"Błąd podczas wczytywania pliku: {e}")
        else:
            logger.error(f"Plik '{path}' nie istnieje.")

    logger.warning("Wczytano domyślną konfigurację.")
    return default_config


def prompt_and_save_json_config(config_data: Dict, console, config_dir: str):
    """Asks user to save the final configuration to a JSON file in CONFIG_DIR."""
    choice = console.input(
        "\nCzy chcesz zapisać finalną konfigurację do pliku? (t/n): "
    ).lower()
    if choice == "t":
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        default_filename = f"config_{timestamp}.json"

        user_filename = console.input(
            f"Podaj nazwę pliku (Enter = [bold cyan]{default_filename}[/bold cyan]): ",
        )
        filename = user_filename or default_filename
        if not filename.endswith(".json"):
            filename += ".json"

        path = os.path.join(config_dir, filename)
        try:
            with open(path, "w") as f:
                json.dump(config_data, f, indent=4)
            logger.success(f"Konfiguracja została zapisana w '{path}'")
        except IOError as e:
            logger.error(f"Błąd podczas zapisu pliku: {e}")
