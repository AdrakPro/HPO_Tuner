import json
import os
import sys
from datetime import datetime
from typing import Dict, Any

from src.logger.experiment_logger import logger
from src.model.chromosome import OptimizerSchedule, AugmentationIntensity
from src.nn.neural_network import ActivationFunction
from src.utils.enum_helper import get_enum_names, get_enum_values


# --- Validation Helper Functions ---


def _check_non_negative_int(value: Any, name: str):
    """Checks if a value is a non-negative integer."""
    if not isinstance(value, int) or value < 0:
        raise ValueError(f"'{name}' must be a non-negative integer.")


def _check_non_negative_float(value: Any, name: str):
    """Checks if a value is a non-negative float."""
    if not isinstance(value, (float, int)) or value < 0:
        raise ValueError(f"'{name}' must be a non-negative float.")


def _check_bool(value: Any, name: str):
    """Checks if a value is a boolean."""
    if not isinstance(value, bool):
        raise ValueError(f"'{name}' must be a boolean value (True or False).")


def _check_float_in_range(value: Any, name: str):
    """Checks if a float is in the range (0.0, 1.0)."""
    if not isinstance(value, (float, int)) or not (0.0 < value < 1.0):
        raise ValueError(f"'{name}' must be a float in the range (0.0, 1.0).")


def _validate_project_config(config: Dict):
    """Validates the 'project' section."""
    if not isinstance(config["name"], str):
        raise ValueError("'project.name' must be a string.")
    if not isinstance(config["seed"], int):
        raise ValueError("'project.seed' must be an integer.")


def _validate_hardware_config(config: Dict):
    """Validates the 'hardware_config' section."""
    allowed_modes = ["CPU", "GPU", "CPU+GPU"]
    if config["evaluation_mode"] not in allowed_modes:
        raise ValueError(
            f"'evaluation_mode' must be one of the options: {allowed_modes}."
        )

    for key in ["cpu_cores", "gpu_devices", "gpu_block_size"]:
        value = config[key]
        if not ((isinstance(value, int) and value >= 0) or value == "-"):
            raise ValueError(f"'{key}' must be a non-negative integer or '-'.")


def _validate_neural_network_config(config: Dict):
    """Validates the 'neural_network_config' section."""
    # --- Base parameter validation ---
    if len(config["input_shape"]) != 3:
        raise ValueError("input_shape must have exactly 3 values.")
    _check_non_negative_int(config["conv_blocks"], "conv_blocks")
    _check_non_negative_int(config["output_classes"], "output_classes")

    # --- Fixed parameter validation ---
    fixed_params = config["fixed_parameters"]
    activation_function = fixed_params["activation_function"].lower()
    allowed = get_enum_values(ActivationFunction)
    if activation_function not in allowed:
        raise ValueError(
            f"Allowed 'activation_function' are {allowed}', but received: '{fixed_params['activation_function']}'."
        )
    _check_non_negative_int(fixed_params["padding"], "padding")
    _check_non_negative_int(fixed_params["stride"], "stride")
    _check_non_negative_int(fixed_params["base_filters"], "base_filters")

    # --- Hyperparameter space validation ---
    hyperparameters = config["hyperparameter_space"]
    for name, params in hyperparameters.items():
        param_type = params["type"]

        # --- Range validation ---
        if "range" in params:
            if (
                not isinstance(params["range"], list)
                or len(params["range"]) != 2
            ):
                raise ValueError(
                    f"The range for '{name}' must be a list containing exactly 2 values."
                )

            min_val, max_val = params["range"]
            if min_val >= max_val:
                raise ValueError(
                    f"Range for '{name}': the minimum value must be less than the maximum."
                )

            if param_type == "int" and not all(
                isinstance(v, int) for v in params["range"]
            ):
                raise ValueError(
                    f"'{name}' is of type 'int', so the values in the range must be integers."
                )
            if param_type == "float" and not all(
                isinstance(v, (int, float)) for v in params["range"]
            ):
                raise ValueError(
                    f"'{name}' is of type 'float', so the values in the range must be numbers."
                )

            if any(v < 0 for v in params["range"]):
                raise ValueError(
                    f"Values in the range for '{name}' cannot be negative."
                )

        # --- Values validation (for enums) ---
        if "values" in params:
            if not isinstance(params["values"], list):
                raise ValueError(
                    f"The 'values' field for '{name}' must be a list."
                )

            if name in ("batch_size", "fc1_units"):
                # TODO Enhancement: here I can force values to be power of two
                if not all(
                    isinstance(v, int) and v > 0 for v in params["values"]
                ):
                    raise ValueError(
                        f"All values for '{name}' must be positive non-zero integers."
                    )
            elif param_type == "enum":
                if not all(isinstance(v, str) for v in params["values"]):
                    raise ValueError(
                        f"'{name}' is of type 'enum' with strings, but the 'values' list contains other types."
                    )
                if name == "optimizer_schedule":
                    allowed = get_enum_names(OptimizerSchedule)

                    if not set(params["values"]).issubset(set(allowed)):
                        raise ValueError(
                            f"Disallowed values for 'optimizer_schedule'. Subsets of the following are allowed: {allowed}."
                        )
                if name == "aug_intensity":
                    allowed = get_enum_names(AugmentationIntensity)

                    if not set(params["values"]).issubset(set(allowed)):
                        raise ValueError(
                            f"Disallowed values for 'aug_intensity'. Subsets of the following are allowed: {allowed}."
                        )
            elif param_type == "int":
                if not all(isinstance(v, int) for v in params["values"]):
                    raise ValueError(
                        f"'{name}' is of type 'enum' with integers, but the 'values' list contains other types."
                    )
        if "step" in params:
            if not isinstance(params["step"], int) or params["step"] <= 0:
                raise ValueError(
                    f"The 'step' value must be a natural number and cannot be zero."
                )


def _validate_nested_validation_config(config: Dict):
    """Validates the 'nested_validation_config' section."""
    _check_bool(config["enabled"], "nested_validation_config.enabled")
    _check_non_negative_int(config["outer_k_folds"], "outer_k_folds")


def _validate_stop_conditions(config: Dict, prefix: str):
    """Validates a 'stop_conditions' block."""
    for key in [
        "max_generations",
        "early_stop_generations",
        "early_stop_epochs",
    ]:
        _check_non_negative_int(config[key], f"{prefix}.{key}")
    _check_non_negative_float(
        config["time_limit_minutes"], f"{prefix}.time_limit_minutes"
    )
    _check_float_in_range(config["fitness_goal"], f"{prefix}.fitness_goal")


def _validate_algorithm_run_config(
    config: Dict, prefix: str, is_calibration: bool
):
    """Validates 'calibration' or 'main_algorithm' sections."""
    if is_calibration:
        _check_bool(config["enabled"], f"{prefix}.enabled")
        _check_float_in_range(
            config["data_subset_percentage"],
            f"{prefix}.data_subset_percentage",
        )

    for key in ["population_size", "generations", "training_epochs"]:
        _check_non_negative_int(config[key], f"{prefix}.{key}")

    _validate_stop_conditions(config["stop_conditions"], prefix)


def _validate_genetic_algorithm_config(config: Dict):
    """Validates the 'genetic_algorithm_config' section."""
    operators = config["genetic_operators"]
    allowed_ops = ["selection", "mutation", "crossover", "elitism"]
    if not all(op in allowed_ops for op in operators["active"]):
        raise ValueError(
            f"Operators in 'active' must be a subset of {allowed_ops}."
        )

    if operators["selection"]["type"] != "tournament":
        raise ValueError("'selection.type' must be 'tournament'.")
    _check_non_negative_int(
        operators["selection"]["tournament_size"], "tournament_size"
    )

    if operators["crossover"]["type"] != "uniform":
        raise ValueError("'crossover.type' must be 'uniform'.")

    _check_non_negative_float(
        operators["crossover"]["crossover_prob"], "crossover_prob"
    )

    mutation_probs = operators["mutation"]
    for key in mutation_probs:
        _check_non_negative_float(mutation_probs[key], key)

    _check_non_negative_float(operators["elitism_percent"], "elitism_percent")

    _validate_algorithm_run_config(
        config["calibration"], "calibration", is_calibration=True
    )
    _validate_algorithm_run_config(
        config["main_algorithm"], "main_algorithm", is_calibration=False
    )


# --- Main Sanitization Function ---


def sanitize_config(config: Dict) -> Dict:
    """
    Validates and sanitizes the entire configuration dictionary.
    Exits the program if any validation fails.
    """
    try:
        _validate_project_config(config["project"])
        _validate_hardware_config(config["hardware_config"])
        _validate_neural_network_config(config["neural_network_config"])
        _validate_nested_validation_config(config["nested_validation_config"])
        _validate_genetic_algorithm_config(config["genetic_algorithm_config"])

    except (KeyError, ValueError) as e:
        logger.error(f"Configuration validation error: {e}")
        sys.exit(1)

    logger.success("Configuration has been successfully checked.")
    return config


def prompt_and_load_json_config(
    default_config: Dict, console, config_dir: str
) -> Dict:
    """Asks user to load a config from JSON, looking inside CONFIG_DIR."""
    while True:
        filename = console.input(
            f"Enter the name of the configuration file in the '{config_dir}' folder: "
        )

        if not filename:
            logger.error("Filename cannot be empty. Please try again.")
            continue

        # Temporary for testing purposes (filename)
        path = os.path.join(config_dir, "config_2025-08-28_20-12-33.json")
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    loaded_config = json.load(f)
                    logger.success(f"Loaded configuration from {path}")
                    return sanitize_config(loaded_config)
            except (json.JSONDecodeError, IOError) as e:
                logger.error(f"Error while loading the file: {e}")
        else:
            logger.error(f"File '{path}' does not exist.")

    logger.warning("Default configuration loaded.")
    return default_config


def prompt_and_save_json_config(config_data: Dict, console, config_dir: str):
    """Asks user to save the final configuration to a JSON file in CONFIG_DIR."""
    choice = console.input(
        "\nDo you want to save the final configuration to a file? (y/n): "
    ).lower()
    if choice == "y":
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        default_filename = f"config_{timestamp}.json"

        user_filename = console.input(
            f"Enter the filename (Enter = [bold cyan]{default_filename}[/bold cyan]): ",
        )
        filename = user_filename or default_filename
        if not filename.endswith(".json"):
            filename += ".json"

        path = os.path.join(config_dir, filename)
        try:
            with open(path, "w") as f:
                json.dump(config_data, f, indent=4)
            logger.success(f"Configuration has been saved to '{path}'")
        except IOError as e:
            logger.error(f"Error while saving the file: {e}")
