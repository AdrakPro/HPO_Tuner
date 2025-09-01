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
from typing import Any, Callable, Dict, List, Optional, Union

import torch
from rich.console import Console
from rich.panel import Panel

from src.config.load_config import (
    prompt_and_load_json_config,
    prompt_and_save_json_config,
)
from src.config.default_config import ex
from src.logger.experiment_logger import logger
from src.model.chromosome import OptimizerSchedule, AugmentationIntensity
from src.utils.ensure_dir import ensure_dir_exists

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
    positive_only: bool = False,
) -> Optional[Union[int, float]]:
    val_str = console.input(f"{prompt} (Enter = {default_value}): ")
    if not val_str:
        return None  # User hit Enter

    try:
        value = value_type(val_str)
        if positive_only and value <= 0:
            console.print(
                f"[yellow]Value must be a positive non-zero number. Using default: {default_value}.[/yellow]"
            )
            return None
        if value < 0:
            console.print(
                f"[yellow]Negative value is not allowed. Using default: {default_value}.[/yellow]"
            )
            return None
        return value
    except (ValueError, TypeError):
        logger.error(f"Invalid value. Using default: {default_value}.")
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
            logger.error(f"Failed to determine max GPU block size: {e}")

    return {
        "max_cpu_cores": os.cpu_count(),
        "gpus_available": gpus_available,
        "max_gpu_devices": torch.cuda.device_count() if gpus_available else 0,
        "max_block_size": max_block_size,
    }


def _prompt_for_evaluation_mode() -> str:
    """Prompts the user to select the evaluation mode (CPU, GPU, or both)."""
    mode_map = {"1": "CPU", "2": "GPU", "3": "CPU+GPU"}
    prompt = "Select execution mode:\n[1] CPU\n[2] GPU\n[3] CPU+GPU\n> "
    error_msg = "Invalid choice. Please enter 1, 2, or 3."
    choice = _prompt_for_validated_input(
        prompt, lambda x: x in mode_map, error_msg
    )
    return mode_map[choice]


def _prompt_for_cpu_cores(max_cores: int) -> Optional[int]:
    """Prompts for the number of CPU cores to use."""
    prompt = f"Enter the number of CPU cores (Available: {max_cores}, Enter = {max_cores}):"
    cpu_input = console.input(prompt)
    if not cpu_input:
        return max_cores
    if cpu_input.isdigit():
        requested_cores = int(cpu_input)
        if 0 < requested_cores <= max_cores:
            return requested_cores
    console.print(f"[yellow]Invalid value. Using value {max_cores}.[/yellow]")
    return max_cores


def _prompt_for_gpu_settings(
    max_devices: int, max_block_size: int, defaults: Dict[str, Any]
) -> Dict[str, Any]:
    """Prompts for GPU device count and block size."""
    gpu_updates: Dict[str, Any] = {}
    if max_devices == 0:
        console.print("[yellow]No available CUDA devices detected.[/yellow]")
        return {"gpu_devices": "-", "gpu_block_size": "-"}

    # Prompt for GPU devices
    gpu_prompt = f"Enter the number of CUDA devices (Available: {max_devices}, Enter = 1): "
    gpu_input = console.input(gpu_prompt)
    if gpu_input.isdigit():
        requested_gpus = int(gpu_input)
        if 0 <= requested_gpus <= max_devices:
            gpu_updates["gpu_devices"] = requested_gpus
        else:
            console.print(
                "[yellow]The requested number of GPUs is invalid. Using '1'.[/yellow]"
            )
            gpu_updates["gpu_devices"] = 1
    elif gpu_input != "":
        console.print("[yellow]Invalid value. Using '1'.[/yellow]")
        gpu_updates["gpu_devices"] = 1
    else:
        gpu_updates["gpu_devices"] = 1

    # Prompt for block size if GPUs are being used
    use_gpus = gpu_updates.get("gpu_devices", defaults.get("gpu_devices"))
    if use_gpus == "-" or (isinstance(use_gpus, int) and use_gpus > 0):
        block_prompt = f"Enter GPU block size (Max: {max_block_size}, Enter = {max_block_size}): "
        block_input = console.input(block_prompt)
        if block_input.isdigit():
            requested_size = int(block_input)
            if 0 < requested_size <= max_block_size:
                gpu_updates["gpu_block_size"] = requested_size
            else:
                console.print(
                    "[yellow]Requested GPU block size is invalid. Using maximum value.[/yellow]"
                )
                gpu_updates["gpu_block_size"] = max_block_size
        elif block_input != "":
            console.print(
                "[yellow]Invalid value. Using maximum value.[/yellow]"
            )
            gpu_updates["gpu_block_size"] = max_block_size
        else:
            gpu_updates["gpu_block_size"] = max_block_size

    return gpu_updates


def _get_hardware_config(defaults: Dict[str, Any]) -> Dict[str, Any]:
    """Prompts user for hardware settings after detecting available resources."""
    _print_header("Computational Resources Selection")
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
    prompt = (
        f"  Enter new range in 'min-max' format (default: {current_range}): "
    )

    if name == "base_lr":
        console.print(
            f"  [cyan]SGD prefers higher LR ranges like '0.01-0.1', ADAMW prefers lower ranges '0.0001-0.005'. If LR would be too high there will be warmup start.[/cyan]"
        )

    new_range_str = console.input(prompt)
    if "-" in new_range_str:
        try:
            value_caster = float if param_type == "float" else int
            low, high = map(value_caster, new_range_str.split("-"))
            if low > high:
                logger.error(
                    "Change not applied. 'max' value must be greater than 'min' value."
                )
                return None

            console.print(
                f"  [green]Updated range for {name} to [{low}, {high}].[/green]"
            )
            return {"range": [low, high]}
        except ValueError:
            logger.error(
                "Incorrect format. Use 'min-max' format with appropriate numbers."
            )
    return None


def _prompt_for_hyperparameter_enum(
    name: str, current_values: List
) -> Optional[Dict[str, List]]:
    """
    Prompts for new categorical values, validating against a predefined set for specific parameters.
    """
    string_enum_values = {
        "optimizer_schedule": [e.name for e in OptimizerSchedule],
        "aug_intensity": [e.name for e in AugmentationIntensity],
    }
    integer_enum_values = ["batch_size", "fc1_units"]

    if name in string_enum_values:
        allowed = string_enum_values[name]
        prompt = f"Enter new comma-separated values (available: {allowed}): "

        new_values_str = console.input(prompt)

        if not new_values_str:
            # User pressed Enter, no changes
            return None

        raw_new_values = [v.strip() for v in new_values_str.split(",")]
        invalid_values = [v for v in raw_new_values if v not in allowed]

        if invalid_values:
            logger.error(
                f"Invalid values for `{name}: {invalid_values}. Allowed are: {allowed}"
            )
            logger.warning(f"Resrtoed default values: {current_values}")
            return None
        new_values = raw_new_values

    elif name in integer_enum_values:
        prompt = f"Enter new comma-separated integer values for '{name}': "
        new_values_str = console.input(prompt)

        try:
            processed_values = [
                int(v.strip()) for v in new_values_str.split(",")
            ]

            if any(v <= 0 for v in processed_values):
                logger.error(f"Values for '{name}' must be positive integers.")
                logger.warning(f"Restored default values: {current_values}.")
                return None

            new_values = processed_values

        except ValueError:
            logger.error(
                f"Invalid integer format for '{name}'. Please enter comma-separated integers."
            )
            logger.warning(f"Restored default values: {current_values}.")
            return None
    else:
        logger.info(
            f"Configuration for '{name}' is not defined as a standard enum in this function."
        )
        return None

    if new_values is not None:
        console.print(
            f"  [green]Updated values for {name} to {new_values}.[/green]"
        )
        return {"values": new_values}

    return None


# --- Neural Network Configuration ---
def _get_neural_network_config(defaults: Dict[str, Any]) -> Dict[str, Any]:
    """Prompts user for neural network architecture settings."""
    _print_header("Neural Network Architecture")
    nn_updates: Dict[str, Any] = {}
    fixed_param_updates: Dict[str, Any] = {}
    nn_defaults = defaults.get(NN_CONFIG, {})
    fixed_defaults = nn_defaults.get("fixed_parameters", {})

    params_to_prompt = {
        "conv_blocks": {
            "prompt": "Number of convolutional blocks",
            "default": nn_defaults.get("conv_blocks", 2),
            "target": nn_updates,
        },
        "base_filters": {
            "prompt": "Number of base filters",
            "default": fixed_defaults.get("base_filters", 32),
            "target": fixed_param_updates,
        },
        "padding": {
            "prompt": "Padding for convolutional layers",
            "default": fixed_defaults.get("padding", 1),
            "target": fixed_param_updates,
        },
        "stride": {
            "prompt": "Stride for convolutional layers",
            "default": fixed_defaults.get("stride", 2),
            "target": fixed_param_updates,
        },
    }

    for key, config in params_to_prompt.items():
        val = _prompt_for_numeric(
            config["prompt"], config["default"], int, positive_only=True
        )
        if val is not None:
            config["target"][key] = val

    if fixed_param_updates:
        nn_updates["fixed_parameters"] = fixed_param_updates

    return {NN_CONFIG: nn_updates} if nn_updates else {}


def _get_hyperparameter_config(defaults: Dict[str, Any]) -> Dict[str, Any]:
    """Prompts user for hyperparameter space definitions."""
    _print_header("Hyperparameter Space Definition")
    hyperparam_updates: Dict[str, Any] = {}
    nn_defaults = defaults.get(NN_CONFIG, {})
    hyperparam_defaults = nn_defaults.get(HYPERPARAM_SPACE, {})

    for name, params in hyperparam_defaults.items():
        console.print(
            f"\n--- [bold]{name}[/bold] ({params.get('description', 'No description')}) ---"
        )
        change = console.input(
            "Do you want to change the space for this parameter? (y/n): "
        ).lower()
        if change != "y":
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
        "\n[bold]Select the genetic operators to be used, separated by commas (min. 2):[/bold]"
    )
    for i, op_name in enumerate(available_ops.values(), 1):
        console.print(f"[{i}] {op_name}")

    # TODO: Czy pisać, że w każdej epoce są losowe, czy nie?
    console.print("[R] Select random operators (values from config file)")

    while True:
        choice_str = console.input("> ").lower().strip()
        if choice_str == "r":
            return ["random"]
        try:
            chosen_indices = [int(i.strip()) - 1 for i in choice_str.split(",")]
            if len(chosen_indices) < 2:
                logger.error("You must select at least 2 operators.")
                continue
            if all(0 <= i < len(op_keys) for i in chosen_indices):
                return [op_keys[i] for i in chosen_indices]
            logger.error("Invalid operator number provided.")
        except ValueError:
            logger.error(
                "Invalid format. Enter numbers separated by commas or 'R'."
            )


def _tune_mutation_parameters(defaults: Dict[str, Any]) -> Dict[str, float]:
    """Prompts user to tune parameters for the mutation operator."""
    mutation_updates: Dict[str, float] = {}
    params_to_ask = {
        "mutation_prob_discrete": "mutation probability for discrete parameters",
        "mutation_prob_categorical": "mutation probability for categorical parameters",
        "mutation_prob_continuous": "mutation probability for continuous parameters",
        "mutation_sigma_continuous": "standard deviation for continuous parameter mutation",
    }

    for param, desc in params_to_ask.items():
        default_val = defaults.get(param, 0.1)
        is_prob = "prob" in param
        prompt = f"Enter {desc}"
        val = _prompt_for_numeric(prompt, default_val, float)
        if val is not None:
            if is_prob and not (0.0 <= val <= 1.0):
                logger.error("Probability must be in the range [0, 1].")
            else:
                mutation_updates[param] = val
    return mutation_updates


def _get_genetic_operators_config(defaults: Dict[str, Any]) -> Dict[str, Any]:
    """Prompts user for genetic operator settings."""
    _print_header("Genetic Operator Selection")
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

    console.print(
        "\n[bold]Adjust parameters for the selected operators:[/bold]"
    )

    if "selection" in active_ops:
        tourn_default = _get_nested_config(
            op_defaults, ["selection", "tournament_size"], 5
        )
        tourn_size = _prompt_for_numeric(
            "Enter tournament size for selection", tourn_default
        )
        if tourn_size is not None and tourn_size > 1:
            updates.setdefault("selection", {})["tournament_size"] = tourn_size

    if "elitism" in active_ops:
        elitism_default = op_defaults.get("elitism_percent", 0.05)
        elitism = _prompt_for_numeric(
            "Enter elitism percentage", elitism_default, float
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
        "max_generations": ("Maximum number of generations", int),
        "early_stop_generations": (
            "Number of generations for early stopping",
            int,
        ),
        "fitness_goal": ("Target fitness value", float),
        "time_limit_minutes": ("Time limit (in minutes)", int),
    }

    for key, (desc, val_type) in params.items():
        val = _prompt_for_numeric(desc, stop_defaults.get(key), val_type)

        if val is not None:
            if key == "fitness_goal" and val > 1.0:
                console.print(
                    f"[yellow]Fitness goal cannot be greater than 1.0! Using value {stop_defaults.get(key)}.[/yellow]"
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
            "Enable calibration? (y/n) (Enter = yes): "
        ).lower()
        if enabled_choice == "n":
            updates["enabled"] = False
            return {GA_CONFIG: {CALIBRATION: updates}}
        elif enabled_choice == "y":
            updates["enabled"] = True
        else:
            updates["enabled"] = True

    # Parameter prompts
    params = {
        "population_size": (
            "Enter the number of chromosomes in the population",
            int,
        ),
        "generations": ("Enter the number of generations", int),
        "training_epochs": ("Enter the number of training epochs", int),
    }
    if config_key == CALIBRATION:
        params["data_subset_percentage"] = (
            "Enter the data subset percentage",
            float,
        )

    for key, (desc, val_type) in params.items():
        val = _prompt_for_numeric(desc, algo_defaults.get(key), val_type)
        if val is not None:
            if key == "data_subset_percentage" and val > 1.0:
                console.print(
                    f"[yellow]Data subset percentage cannot be greater than 1.0! Using value {algo_defaults.get(key)}.[/yellow]"
                )
                updates[key] = algo_defaults.get(key)
                continue

            updates[key] = val

    # Stop conditions
    console.print(f"\n[bold]Stop Conditions:[/bold]")
    stop_updates = _prompt_for_stop_conditions(algo_defaults)
    if stop_updates:
        updates[STOP_CONDITIONS] = stop_updates

    return {GA_CONFIG: {config_key: updates}} if updates else {}


# --- Main TUI Runner ---
def run_tui_configurator() -> Optional[Dict[str, Any]]:
    """Main function to run the TUI and collect all configuration overrides."""
    ensure_dir_exists(CONFIG_DIR)
    default_config = ex.configurations[0]()

    _print_header("GENETIC OPTIMIZATION OF CNN")
    prompt = (
        "[1] Create new configuration\n[2] Load configuration\n[3] Exit\n> "
    )
    error_msg = "Invalid choice. Please enter 1, 2, or 3."
    choice = _prompt_for_validated_input(
        prompt, lambda x: x in ["1", "2", "3"], error_msg
    )

    if choice == "3":
        logger.warning("Exiting program.")
        return None
    if choice == "2":
        return prompt_and_load_json_config(default_config, console, CONFIG_DIR)

    logger.file_only("New configuration started")

    # Interactive configuration
    config_updates: Dict[str, Any] = {}

    config_updates.update(_get_hardware_config(default_config))
    nn_updates = _get_neural_network_config(default_config)
    hyperparam_updates = _get_hyperparameter_config(default_config)

    if nn_updates.get(NN_CONFIG):
        config_updates.setdefault(NN_CONFIG, {}).update(nn_updates[NN_CONFIG])
    if hyperparam_updates.get(NN_CONFIG):
        config_updates.setdefault(NN_CONFIG, {}).update(
            hyperparam_updates[NN_CONFIG]
        )

    config_updates.update(_get_genetic_operators_config(default_config))

    calib_updates = _get_algorithm_settings(
        "Initial Calibration Settings", CALIBRATION, default_config
    )
    if calib_updates.get(GA_CONFIG):
        config_updates.setdefault(GA_CONFIG, {}).update(
            calib_updates[GA_CONFIG]
        )

    main_updates = _get_algorithm_settings(
        "Main Algorithm Settings", MAIN_ALGORITHM, default_config
    )
    if main_updates.get(GA_CONFIG):
        config_updates.setdefault(GA_CONFIG, {}).update(main_updates[GA_CONFIG])

    console.print(
        "\n[bold green]Interactive configuration complete.[/bold green]"
    )
    final_config = _deep_merge_dicts(default_config, config_updates)
    prompt_and_save_json_config(final_config, console, CONFIG_DIR)
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
    logger.file_only(f"Configuration\n: {json.dumps(_config, indent=4)}")
