# TODO: sprawdzic odpornosc TUI, t/n gdy wybrana jest s to i tak poleci bo nie t
# TODO: future dont get block size, cuda can heursticly calculate it cudaOccupancyMaxPotentialBlockSize cuda_runtime.h
# TODO: Enhancet: load last config, show with numbers to load last 5 latest configs

"""
Text-based User Interface (TUI) for configuring the experiment.
This module collects user input and returns a dictionary of configuration overrides.
"""
import collections
import json
import os
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Union

import torch
from rich.console import Console
from rich.panel import Panel
from src.config.settings import ex
from src.logger.experiment_logger import logger

console = Console(highlight=False)

# --- Constants ---
CONFIG_DIR = os.path.abspath("configs")
HARDWARE_CONFIG = "hardware_config"
NN_CONFIG = "neural_network_config"
GA_CONFIG = "genetic_algorithm_config"
HYPERPARAM_SPACE = "hyperparameter_space"
GENETIC_OPERATORS = "genetic_operators"
CALIBRATION = "calibration"
MAIN_ALGORITHM = "main_algorithm"
STOP_CONDITIONS = "stop_conditions"


# --- Helper Functions ---
def _print_header(title: str):
    console.print(f"\n" + "-" * 60)
    console.print(f"  [bold cyan]{title.upper()}[/bold cyan]")
    console.print(f"-" * 60)


def _prompt_for_validated_input(
    prompt: str, validation_callable: Callable[[str], bool], error_message: str
) -> str:
    while True:
        user_input = console.input(prompt)
        if validation_callable(user_input):
            return user_input
        logger.error(error_message)


def _prompt_for_numeric(
    prompt: str,
    default_value: Optional[Union[int, float]],
    value_type: Union[type[int], type[float]] = int,
) -> Optional[Union[int, float]]:
    val_str = console.input(f"{prompt} (Enter = {default_value}): ")
    if not val_str:
        return None  # User hit Enter

    try:
        value = value_type(val_str)
        if value < 0:
            console.print(
                f"[yellow]Wartość ujemna nie jest dozwolona. Użyto domyślnej: {default_value}.[/yellow]"
            )
            return None
        return value
    except (ValueError, TypeError):
        logger.error(
            f"Nieprawidłowa wartość. Użyto domyślnej: {default_value}."
        )
        return None


def _get_nested_config(
    config: Dict, path: List[str], default: Any = None
) -> Any:
    """Safely retrieves a nested value from a dictionary."""
    for key in path:
        if not isinstance(config, dict) or key not in config:
            return default
        config = config[key]
    return config


def _deep_merge_dicts(base: Dict, updates: Dict) -> Dict:
    """Recursively merges dictionaries."""
    for key, value in updates.items():
        if isinstance(value, collections.abc.Mapping) and key in base:
            base[key] = _deep_merge_dicts(base[key], value)
        else:
            base[key] = value
    return base


# --- Hardware Configuration ---
def _detect_hardware_resources() -> Dict[str, Any]:
    """Detects available CPU and GPU resources."""
    gpus_available = torch.cuda.is_available()
    max_block_size = 0
    if gpus_available:
        try:
            # This is a placeholder as per the TODO.
            # In a real scenario, this might involve a device query.
            max_block_size = 512
        except Exception as e:
            logger.error(
                f"Nie udało się określić maks. rozmiaru bloku GPU: {e}"
            )

    return {
        "max_cpu_cores": os.cpu_count(),
        "gpus_available": gpus_available,
        "max_gpu_devices": torch.cuda.device_count() if gpus_available else 0,
        "max_block_size": max_block_size,
    }


def _prompt_for_evaluation_mode() -> str:
    """Prompts the user to select the evaluation mode (CPU, GPU, or both)."""
    mode_map = {"1": "CPU", "2": "GPU", "3": "CPU+GPU"}
    prompt = "Wykorzystaj:\n[1] CPU\n[2] GPU\n[3] CPU+GPU\n> "
    error_msg = "Nieprawidłowy wybór. Wprowadź 1, 2, lub 3."
    choice = _prompt_for_validated_input(
        prompt, lambda x: x in mode_map, error_msg
    )
    return mode_map[choice]


def _prompt_for_cpu_cores(max_cores: int) -> Optional[int]:
    """Prompts for the number of CPU cores to use."""
    prompt = (
        f"Podaj liczbę rdzeni CPU (Dostępne: {max_cores}, Enter = {max_cores}):"
    )
    cpu_input = console.input(prompt)
    if not cpu_input:
        return max_cores
    if cpu_input.isdigit():
        requested_cores = int(cpu_input)
        if 0 < requested_cores <= max_cores:
            return requested_cores
    console.print(
        f"[yellow]Nieprawidłowa wartość. Użyto wartości {max_cores}.[/yellow]"
    )
    return max_cores


def _prompt_for_gpu_settings(
    max_devices: int, max_block_size: int, defaults: Dict[str, Any]
) -> Dict[str, Any]:
    """Prompts for GPU device count and block size."""
    gpu_updates: Dict[str, Any] = {}
    if max_devices == 0:
        console.print("[yellow]Nie wykryto dostępnych urządzeń CUDA.[/yellow]")
        return {"gpu_devices": "-", "gpu_block_size": "-"}

    # Prompt for GPU devices
    gpu_prompt = (
        f"Podaj liczbę urządzeń CUDA (Dostępne: {max_devices}, Enter = 1): "
    )
    gpu_input = console.input(gpu_prompt)
    if gpu_input.isdigit():
        requested_gpus = int(gpu_input)
        if 0 <= requested_gpus <= max_devices:
            gpu_updates["gpu_devices"] = requested_gpus
        else:
            console.print(
                "[yellow]Żądana liczba GPU jest nieprawidłowa. Użyto wartości '1'.[/yellow]"
            )
            gpu_updates["gpu_devices"] = 1
    elif gpu_input != "":
        console.print(
            "[yellow]Nieprawidłowa wartość. Użyto wartości '1'.[/yellow]"
        )
        gpu_updates["gpu_devices"] = 1
    else:
        gpu_updates["gpu_devices"] = 1

    # Prompt for block size if GPUs are being used
    use_gpus = gpu_updates.get("gpu_devices", defaults.get("gpu_devices"))
    if use_gpus == "-" or (isinstance(use_gpus, int) and use_gpus > 0):
        block_prompt = f"Podaj rozmiar bloku GPU (Max: {max_block_size}, Enter = {max_block_size}): "
        block_input = console.input(block_prompt)
        if block_input.isdigit():
            requested_size = int(block_input)
            if 0 < requested_size <= max_block_size:
                gpu_updates["gpu_block_size"] = requested_size
            else:
                console.print(
                    "[yellow]Żądany rozmiar bloku GPU jest nieprawidłowy. Użyto wartości maksymalnej.[/yellow]"
                )
                gpu_updates["gpu_block_size"] = max_block_size
        elif block_input != "":
            console.print(
                "[yellow]Nieprawidłowa wartość. Użyto wartości maksymalnej.[/yellow]"
            )
            gpu_updates["gpu_block_size"] = max_block_size
        else:
            gpu_updates["gpu_block_size"] = max_block_size

    return gpu_updates


def _get_hardware_config(defaults: Dict[str, Any]) -> Dict[str, Any]:
    """Prompts user for hardware settings after detecting available resources."""
    _print_header("Wybór zasobów obliczeniowych")
    updates: Dict[str, Any] = {}
    hw_defaults = defaults.get(HARDWARE_CONFIG, {})
    resources = _detect_hardware_resources()

    # Get evaluation mode first to determine next steps
    eval_mode = _prompt_for_evaluation_mode()
    updates["evaluation_mode"] = eval_mode

    # CPU settings
    if "CPU" in eval_mode:
        cpu_cores = _prompt_for_cpu_cores(
            resources["max_cpu_cores"],
        )
        if cpu_cores is not None:
            updates["cpu_cores"] = cpu_cores

    # GPU settings
    if "GPU" in eval_mode:
        gpu_settings = _prompt_for_gpu_settings(
            resources["max_gpu_devices"],
            resources["max_block_size"],
            hw_defaults,
        )
        updates.update(gpu_settings)
    else:
        updates["gpu_devices"] = "-"
        updates["gpu_block_size"] = "-"

    return {HARDWARE_CONFIG: updates} if updates else {}


# --- Hyperparameter Configuration ---
def _prompt_for_hyperparameter_range(
    name: str, param_type: str, current_range: List
) -> Optional[Dict[str, List]]:
    """Prompts for a new min-max range for a hyperparameter."""
    prompt = f"  Podaj nowy zakres w formacie 'min-max' (domyślnie: {current_range}): "
    new_range_str = console.input(prompt)
    if "-" in new_range_str:
        try:
            value_caster = float if param_type == "float" else int
            low, high = map(value_caster, new_range_str.split("-"))
            if low > high:
                logger.error(
                    "Nie zastosowano zmiany. Wartość 'max' musi być większa od wartości 'min'."
                )
                return None

            console.print(
                f"  [green]Zaktualizowano zakres dla {name} na [{low}, {high}].[/green]"
            )
            return {"range": [low, high]}
        except ValueError:
            logger.error(
                "Błędny format. Użyj formatu 'min-max' z odpowiednimi liczbami."
            )
    return None


def _prompt_for_hyperparameter_enum(
    name: str, current_values: List
) -> Optional[Dict[str, List]]:
    """Prompts for new categorical values for a hyperparameter."""
    prompt = f"  Podaj nowe wartości oddzielone przecinkami (domyślnie: {current_values}): "
    new_values_str = console.input(prompt)
    if new_values_str:
        new_values = [v.strip() for v in new_values_str.split(",")]
        if current_values:
            try:
                original_type = type(current_values[0])
                new_values = [original_type(v) for v in new_values]
            except (ValueError, IndexError):
                pass
        console.print(
            f"  [green]Zaktualizowano wartości dla {name} na {new_values}.[/green]"
        )
        return {"values": new_values}
    return None


def _get_hyperparameter_config(defaults: Dict[str, Any]) -> Dict[str, Any]:
    """Prompts user for hyperparameter space definitions."""
    _print_header("Definicja przestrzeni hiperparametrów")
    hyperparam_updates: Dict[str, Any] = {}
    nn_defaults = defaults.get(NN_CONFIG, {})
    hyperparam_defaults = nn_defaults.get(HYPERPARAM_SPACE, {})

    for name, params in hyperparam_defaults.items():
        console.print(
            f"\n--- [bold]{name}[/bold] ({params.get('description', 'Brak opisu')}) ---"
        )
        change = console.input(
            "Czy chcesz zmienić przestrzeń dla tego parametru? (t/n): "
        ).lower()
        if change != "t":
            continue

        param_type = params.get("type")
        update = None
        if param_type in ["float", "int"]:
            update = _prompt_for_hyperparameter_range(
                name, param_type, params.get("range", "[?, ?]")
            )
        elif param_type == "enum":
            update = _prompt_for_hyperparameter_enum(
                name, params.get("values", "[]")
            )

        if update:
            hyperparam_updates[name] = update

    if hyperparam_updates:
        return {NN_CONFIG: {HYPERPARAM_SPACE: hyperparam_updates}}
    return {}


# --- Genetic Operators Configuration ---
def _select_active_operators(
    op_keys: List[str], available_ops: Dict[str, str]
) -> List[str]:
    """Handles the user selection of genetic operators."""
    console.print(
        "\n[bold]Wybierz operatory genetyczne odzielone przecinkami, które mają być użyte (min. 2):[/bold]"
    )
    for i, op_name in enumerate(available_ops.values(), 1):
        console.print(f"[{i}] {op_name}")

    # TODO: Czy pisać, że w każdej epoce są losowe, czy nie?
    console.print(
        "[R] Wybierz losowe operatory (wartości z pliku konfiguracyjnego)"
    )

    while True:
        choice_str = console.input("> ").lower().strip()
        if choice_str == "r":
            return ["random"]
        try:
            chosen_indices = [int(i.strip()) - 1 for i in choice_str.split(",")]
            if len(chosen_indices) < 2:
                logger.error("Musisz wybrać co najmniej 2 operatory.")
                continue
            if all(0 <= i < len(op_keys) for i in chosen_indices):
                return [op_keys[i] for i in chosen_indices]
            logger.error("Podano nieprawidłowy numer operatora.")
        except ValueError:
            logger.error(
                "Nieprawidłowy format. Podaj numery oddzielone przecinkami lub 'R'."
            )


def _tune_mutation_parameters(defaults: Dict[str, Any]) -> Dict[str, float]:
    """Prompts user to tune parameters for the mutation operator."""
    mutation_updates: Dict[str, float] = {}
    params_to_ask = {
        "mutation_prob_discrete": "prawdopodobieństwo mutacji dla parametrów dyskretnych",
        "mutation_prob_categorical": "prawdopodobieństwo mutacji dla parametrów kategorycznych",
        "mutation_prob_continuous": "prawdopodobieństwo mutacji dla parametrów ciągłych",
        "mutation_sigma_continuous": "odchylenie standardowe dla mutacji parametrów ciągłych",
    }

    for param, desc in params_to_ask.items():
        default_val = defaults.get(param, 0.1)
        is_prob = "prob" in param
        prompt = f"Podaj {desc}"
        val = _prompt_for_numeric(prompt, default_val, float)
        if val is not None:
            if is_prob and not (0.0 <= val <= 1.0):
                logger.error("Prawdopodobieństwo musi być w zakresie [0, 1].")
            else:
                mutation_updates[param] = val
    return mutation_updates


def _get_genetic_operators_config(defaults: Dict[str, Any]) -> Dict[str, Any]:
    """Prompts user for genetic operator settings."""
    _print_header("Wybór operatorów genetycznych")
    updates: Dict[str, Any] = {}
    ga_defaults = defaults.get(GA_CONFIG, {})
    op_defaults = ga_defaults.get(GENETIC_OPERATORS, {})

    available_operators = {
        "selection": "Tournament Selection",
        "crossover": "Uniform Crossover",
        "mutation": "Gaussian Mutation",
        "elitism": "Elitism",
    }
    op_keys = list(available_operators.keys())

    active_ops = _select_active_operators(op_keys, available_operators)
    updates["active"] = active_ops

    if "random" in active_ops:
        return {GA_CONFIG: {GENETIC_OPERATORS: updates}}

    console.print("\n[bold]Dostosuj parametry dla wybranych operatorów:[/bold]")

    if "selection" in active_ops:
        tourn_default = _get_nested_config(
            op_defaults, ["selection", "tournament_size"], 5
        )
        tourn_size = _prompt_for_numeric(
            "Podaj rozmiar turnieju dla selekcji", tourn_default
        )
        if tourn_size is not None and tourn_size > 1:
            updates.setdefault("selection", {})["tournament_size"] = tourn_size

    if "elitism" in active_ops:
        elitism_default = op_defaults.get("elitism_percent", 0.05)
        elitism = _prompt_for_numeric(
            "Podaj procent elitaryzmu", elitism_default, float
        )
        if elitism is not None and 0.0 <= elitism < 1.0:
            updates["elitism_percent"] = elitism

    if "mutation" in active_ops:
        mutation_defaults = op_defaults.get("mutation", {})
        mutation_updates = _tune_mutation_parameters(mutation_defaults)
        if mutation_updates:
            updates["mutation"] = mutation_updates

    return {GA_CONFIG: {GENETIC_OPERATORS: updates}} if updates else {}


# --- Algorithm Settings (Calibration & Main) ---
def _prompt_for_stop_conditions(defaults: Dict[str, Any]) -> Dict[str, Any]:
    """Prompts for algorithm stop conditions."""
    stop_updates: Dict[str, Any] = {}
    stop_defaults = defaults.get(STOP_CONDITIONS, {})

    params = {
        "max_generations": ("Maksymalna liczba generacji", int),
        "early_stop_generations": (
            "Liczba generacji do wczesnego zatrzymania",
            int,
        ),
        "fitness_goal": ("Docelowa wartość funkcji celu (fitness)", float),
        "time_limit_minutes": ("Limit czasowy (w minutach)", int),
    }

    for key, (desc, val_type) in params.items():
        val = _prompt_for_numeric(desc, stop_defaults.get(key), val_type)

        if val is not None:
            if key == "fitness_goal" and val > 1.0:
                console.print(
                    f"[yellow]Wartość funkcji celu nie może być większa niż 1.0! Użytwo wartości {stop_defaults.get(key)}.[/yellow]"
                )
                stop_updates[key] = stop_defaults.get(key)
                continue

            stop_updates[key] = val

    return stop_updates


def _get_algorithm_settings(
    title: str, config_key: str, defaults: Dict[str, Any]
) -> Dict[str, Any]:
    """Generic function to prompt for main or calibration algorithm settings."""
    _print_header(title)
    updates: Dict[str, Any] = {}
    algo_defaults = _get_nested_config(defaults, [GA_CONFIG, config_key], {})

    if config_key == CALIBRATION:
        enabled_choice = console.input(
            "Czy włączyć kalibrację? (t/n): "
        ).lower()
        if enabled_choice == "n":
            updates["enabled"] = False
            return {GA_CONFIG: {CALIBRATION: updates}}
        elif enabled_choice == "t":
            updates["enabled"] = True

    # Parameter prompts
    params = {
        "population_size": ("Podaj liczbę chromosomów w populacji", int),
        "generations": ("Podaj liczbę generacji", int),
        "training_epochs": ("Podaj liczbę epok treningowych", int),
    }
    if config_key == CALIBRATION:
        params["data_subset_percentage"] = (
            "Podaj procent podzbioru danych (np. 0.2)",
            float,
        )

    for key, (desc, val_type) in params.items():
        val = _prompt_for_numeric(desc, algo_defaults.get(key), val_type)
        if val is not None:
            if key == "data_subset_percentage" and val > 1.0:
                console.print(
                    f"[yellow]Procent podzbioru danych nie może być większy niż 1.0! Użytwo wartości {algo_defaults.get(key)}.[/yellow]"
                )
                updates[key] = algo_defaults.get(key)
                continue

            updates[key] = val

    # Stop conditions
    console.print(f"\n[bold]Warunki zatrzymania:[/bold]")
    stop_updates = _prompt_for_stop_conditions(algo_defaults)
    if stop_updates:
        updates[STOP_CONDITIONS] = stop_updates

    return {GA_CONFIG: {config_key: updates}} if updates else {}


# --- Main TUI Runner ---
def _ensure_config_dir_exists():
    """Creates the CONFIG_DIR if it does not already exist."""
    if not os.path.exists(CONFIG_DIR):
        os.makedirs(CONFIG_DIR)
        logger.info(
            f"Utworzono katalog '{CONFIG_DIR}' na pliki konfiguracyjne."
        )


def _prompt_and_load_json_config(default_config: Dict) -> Dict:
    """Asks user to load a config from JSON, looking inside CONFIG_DIR."""
    while True:
        filename = console.input(
            f"Podaj nazwę pliku konfiguracji w folderze '{CONFIG_DIR}': "
        )
        print(filename)
        if not filename:
            logger.error("Nazwa pliku nie może być pusta. Spróboj ponownie.")
            continue

        path = os.path.join(CONFIG_DIR, filename)
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


def _prompt_and_save_json_config(config_data: Dict):
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

        path = os.path.join(CONFIG_DIR, filename)
        try:
            with open(path, "w") as f:
                json.dump(config_data, f, indent=4)
            logger.success(f"Konfiguracja została zapisana w '{path}'")
        except IOError as e:
            logger.error(f"Błąd podczas zapisu pliku: {e}")


def run_tui_configurator() -> Optional[Dict[str, Any]]:
    """Main function to run the TUI and collect all configuration overrides."""
    _ensure_config_dir_exists()
    default_config = ex.configurations[0]()

    _print_header("GENETYCZNA OPTYMALIZACJA SIECI CNN")
    prompt = (
        "[1] Utwórz nową konfigurację\n[2] Wczytaj konfigurację\n[3] Wyjdź\n> "
    )
    error_msg = "Nieprawidłowy wybór. Wprowadź 1, 2, lub 3."
    choice = _prompt_for_validated_input(
        prompt, lambda x: x in ["1", "2", "3"], error_msg
    )

    if choice == "3":
        logger.warning("Zamykanie programu.")
        return None
    if choice == "2":
        return _prompt_and_load_json_config(default_config)

    logger.log_file("Rozpoczęto nowa konfiguracje")

    # Interactive configuration
    config_updates: Dict[str, Any] = {}

    config_updates.update(_get_hardware_config(default_config))
    config_updates.update(_get_hyperparameter_config(default_config))
    config_updates.update(_get_genetic_operators_config(default_config))

    calib_updates = _get_algorithm_settings(
        "Ustawienia wstępnej kalibracji", CALIBRATION, default_config
    )
    if calib_updates.get(GA_CONFIG):
        config_updates.setdefault(GA_CONFIG, {}).update(
            calib_updates[GA_CONFIG]
        )

    main_updates = _get_algorithm_settings(
        "Ustawienia głównego algorytmu", MAIN_ALGORITHM, default_config
    )
    if main_updates.get(GA_CONFIG):
        config_updates.setdefault(GA_CONFIG, {}).update(main_updates[GA_CONFIG])

    console.print(
        "\n[bold green]Interaktywna konfiguracja zakończona.[/bold green]"
    )
    final_config = _deep_merge_dicts(default_config, config_updates)
    _prompt_and_save_json_config(final_config)
    return final_config


# --- Final Configuration Display ---
def print_final_config_panel(_config: Dict[str, Any]):
    """Displays the final configuration in a formatted panel."""
    hyperparams = _get_nested_config(_config, [NN_CONFIG, HYPERPARAM_SPACE], {})
    hyper_str = "\n".join(
        f"  [cyan]{name}:[/cyan] {details.get('range') or details.get('values')}"
        for name, details in hyperparams.items()
    )

    main_algo_path = [GA_CONFIG, MAIN_ALGORITHM]
    stop_cond_path = main_algo_path + [STOP_CONDITIONS]

    config_details = (
        f"[cyan]Experiment Name:[/cyan] {_get_nested_config(_config, ['project', 'name'])}\n"
        f"[cyan]Seed:[/cyan] {_get_nested_config(_config, ['project', 'seed'])}\n"
        f"[cyan]Evaluation Mode:[/cyan] {_get_nested_config(_config, [HARDWARE_CONFIG, 'evaluation_mode'])}\n"
        f"[cyan]CPU Cores:[/cyan] {_get_nested_config(_config, [HARDWARE_CONFIG, 'cpu_cores'])}\n"
        f"[cyan]GPU Devices:[/cyan] {_get_nested_config(_config, [HARDWARE_CONFIG, 'gpu_devices'])}\n"
        f"[cyan]GPU Block Size:[/cyan] {_get_nested_config(_config, [HARDWARE_CONFIG, 'gpu_block_size'])}\n"
        f"-------------------\n"
        f"[bold]CNN Params:[/bold]\n"
        f"  [cyan]Conv Blocks:[/cyan] {_get_nested_config(_config, [NN_CONFIG, 'conv_blocks'])}\n"
        f"  [cyan]Activation:[/cyan] {_get_nested_config(_config, [NN_CONFIG, 'fixed_parameters', 'activation_function'])}\n"
        f"  [cyan]Padding:[/cyan] {_get_nested_config(_config, [NN_CONFIG, 'fixed_parameters', 'padding'])}\n"
        f"  [cyan]Stride:[/cyan] {_get_nested_config(_config, [NN_CONFIG, 'fixed_parameters', 'stride'])}\n"
        f"-------------------\n"
        f"[bold]Hyperparameter Space:[/bold]\n{hyper_str}\n"
        f"-------------------\n"
        f"[cyan]Nested Validation Enabled:[/cyan] {_get_nested_config(_config, ['nested_validation_config', 'enabled'])}\n"
        f"[cyan]Outer K Folds:[/cyan] {_get_nested_config(_config, ['nested_validation_config', 'outer_k_folds'])}\n"
        f"-------------------\n"
        f"[bold]Main Algorithm:[/bold]\n"
        f"  [cyan]Calibration Enabled:[/cyan] {_get_nested_config(_config, [GA_CONFIG, CALIBRATION, 'enabled'])}\n"
        f"  [cyan]Genetic Operators Active:[/cyan] {', '.join(_get_nested_config(_config, [GA_CONFIG, GENETIC_OPERATORS, 'active'], []))}\n"
        f"  [cyan]Population Size:[/cyan] {_get_nested_config(_config, main_algo_path + ['population_size'])}\n"
        f"  [cyan]Generations:[/cyan] {_get_nested_config(_config, main_algo_path + ['generations'])}\n"
        f"  [cyan]Training Epochs:[/cyan] {_get_nested_config(_config, main_algo_path + ['training_epochs'])}\n"
        f"  [cyan]Stop Conditions:[/cyan]\n"
        f"    [cyan]Max Gen:[/cyan] {_get_nested_config(_config, stop_cond_path + ['max_generations'])}\n"
        f"    [cyan]Early Stop Gen:[/cyan] {_get_nested_config(_config, stop_cond_path + ['early_stop_generations'])}\n"
        f"    [cyan]Fitness Goal:[/cyan] {_get_nested_config(_config, stop_cond_path + ['fitness_goal'])}\n"
        f"    [cyan]Time Limit (min):[/cyan] {_get_nested_config(_config, stop_cond_path + ['time_limit_minutes'])}"
    )

    panel = Panel(
        config_details,
        title="[bold cyan]Final Configuration[/bold cyan]",
        border_style="cyan",
        expand=False,
    )
    console.print(panel)
    logger.log_file(f"Konfiguracja\n: {json.dumps(_config, indent=4)}")
