"""
Configuration settings for the project using Sacred.
This file defines the experiment and its configuration based on the provided specification.
"""

from sacred import Experiment

ex = Experiment("Optymalizacja")


@ex.config
def default_config():
    project = {"name": "HPO Paper-Based Experiment", "seed": 2137}

    # TODO: integrate
    environment = {
        "data_source": {"dataset": "CIFAR-10", "path": "./model_data"},
        "logging": {
            "log_file": "logs/experiment.log",
            "error_log_file": "logs/error.log",
            "report_directory": "reports",
        },
    }

    hardware_config = {
        "evaluation_mode": "CPU+GPU",  # Options: "CPU", "GPU", "CPU+GPU"
        "cpu_cores": "-",
        "gpu_devices": "-",
        "gpu_block_size": "-",
    }

    neural_network_config = {
        "input_shape": [3, 32, 32],  # (channels, height, width)
        "output_classes": 10,
        "conv_blocks": 2,
        # Parameters not subject to optimization
        "fixed_parameters": {
            "activation_function": "relu",  # Options: relu, gelu, leaky_relu
            "padding": 1,
            "stride": 2,
            # TODO: Maybe this should also be available for search with width_scale
            "base_filters": 32,
        },
        "hyperparameter_space": {
            "width_scale": {
                "type": "float",
                "range": [
                    0.75,
                    2.0,
                ],
                "description": "Scales the number of filters in convolutional layers",
            },
            "fc1_units": {
                "type": "enum",
                "values": [256, 512, 1024],
                "description": "Number of neurons in the first fully connected layer",
            },
            "dropout_rate": {
                "type": "float",
                "range": [0.2, 0.5],
                "description": "Dropout intensity in the first fully connected layer",
            },
            "optimizer_schedule": {
                "type": "enum",
                "values": [
                    "SGD_ONECYCLE",
                    "SGD_COSINE",
                    "SGD_EXPONENTIAL",
                ],  # Options: [SGD_COSINE, SGD_ONECYCLE, SGD_EXPONENTIAL, ADAMW_COSINE, ADAMW_ONECYLE, ADAMW_EXPONENTIAL]
                "description": "Optimizer type + Learning rate scheduler",
            },
            "base_lr": {
                "type": "float",
                "range": [0.01, 0.1],
                "scale": "log",
                "description": "Base learning rate value, log scale.",
            },
            "aug_intensity": {
                "type": "enum",
                "values": [
                    "NONE",
                    "LIGHT",
                    "MEDIUM",
                    "STRONG",
                ],  # Options: [NONE, LIGHT, MEDIUM, STRONG]
                "description": "Level of data augmentation",
            },
            "weight_decay": {
                "type": "float",
                "range": [1e-4, 5e-4],
                "scale": "log",
                "description": "L2 regularization coefficient, log scale",
            },
            "batch_size": {
                "type": "enum",
                "values": [64, 128, 256],
                "description": "Training batch size",
            },
        },
    }

    nested_validation_config = {"enabled": True, "outer_k_folds": 3}

    genetic_algorithm_config = {
        "genetic_operators": {
            "active": [
                "selection",
                "mutation",
                "crossover",
                "elitism",
            ],  # Options: selection, mutation, crossover, elitism or random
            "selection": {"type": "tournament", "tournament_size": 3},
            "crossover": {"type": "uniform", "crossover_prob": 0.8},
            "mutation": {
                "mutation_prob_discrete": 0.15,
                "mutation_prob_categorical": 0.15,
                "mutation_sigma_continuous": 0.1,
                "mutation_prob_continuous": 0.1,
            },
            "elitism_percent": 0.1,
        },
        "calibration": {
            "enabled": True,
            "population_size": 30,
            "generations": 15,
            "training_epochs": 5,
            "data_subset_percentage": 0.25,
            "stop_conditions": {
                "max_generations": 15,
                "early_stop_generations": 5,
                "fitness_goal": 0.70,
                # TODO: change to HOURS
                "time_limit_minutes": 5,
            },
        },
        "main_algorithm": {
            "population_size": 40,
            "generations": 50,
            "training_epochs": 100,
            "stop_conditions": {
                "max_generations": 50,
                "early_stop_generations": 10,
                "fitness_goal": 0.995,
                "time_limit_minutes": 5,
            },
        },
    }
