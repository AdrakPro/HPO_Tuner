"""
Configuration settings for the project using Sacred.
This file defines the experiment and its configuration based on the provided specification.
"""

from sacred import Experiment

ex = Experiment("Optymalizacja_CNN")


@ex.config
def config():
    project = {"name": "Eksperyment 1.0", "seed": 2137}

    environment = {
        "data_source": {"dataset": "CIFAR-10", "path": "./data"},
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
        "base_architecture": "CNN",
        "input_shape": [3, 32, 32],  # (channels, height, width)
        "output_classes": 10,
        "conv_blocks": 2,
        # Parameters not subject to optimization
        "fixed_parameters": {
            "activation_function": "ReLu",
            "padding": 1,
            "stride": 2,
        },
        "hyperparameter_space": {
            "width_scale": {
                "type": "float",
                "range": [0.5, 2.0],
                "description": "Skaluje liczbę filtrów w warstawach conv",
            },
            "fc1_units": {
                "type": "int",
                "range": [128, 1024],
                "step": 64,
                "description": "Liczba neuronów w pierwszej warstwie FC",
            },
            "dropout_rate": {
                "type": "float",
                "range": [0.0, 0.6],
                "description": "Intensywność dropout w FC1",
            },
            "optimizer_schedule": {
                "type": "enum",
                "values": [
                    "SGD_STEP",
                    "SGD_COSINE",
                    "ADAMW_COSINE",
                    "ADAMW_ONECYCLE",
                ],
                "description": "Typ optymalizatora + scheduler LR",
            },
            "base_lr": {
                "type": "float",
                "range": [0.0001, 0.3],
                "scale": "log",
                "description": "Bazowa wartość learning rate",
            },
            "aug_intensity": {
                "type": "enum",
                "values": ["NONE", "LIGHT", "MEDIUM", "STRONG"],
                "description": "Poziom augmentacji danych",
            },
            "weight_decay": {
                "type": "float",
                "range": [1e-05, 0.01],
                "scale": "log",
                "description": "Współczynnik regularyzacji L2",
            },
            "batch_size": {
                "type": "enum",
                "values": [32, 64, 128, 256],
                "description": "Rozmiar batcha treningowego",
            },
        },
    }

    nested_validation_config = {"enabled": True, "outer_k_folds": 3}

    genetic_algorithm_config = {
        "genetic_operators": {
            "active": ["selection", "mutation", "crossover", "elitism"],
            "selection": {"type": "tournament", "tournament_size": 5},
            "crossover": {"type": "uniform"},
            "mutation": {
                "mutation_prob_discrete": 0.05,
                "mutation_prob_categorical": 0.05,
                "mutation_sigma_continuous": 0.05,
                "mutation_prob_continuous": 0.05,
            },
            "elitism_percent": 0.05,
        },
        "calibration": {
            "enabled": True,
            "population_size": 20,
            "generations": 10,
            "training_epochs": 3,
            "data_subset_percentage": 0.2,
            "stop_conditions": {
                "max_generations": 10,
                "early_stop_generations": 3,
                "fitness_goal": 0.99,
                "time_limit_minutes": 5,
            },
        },
        "main_algorithm": {
            "population_size": 50,
            "generations": 50,
            "training_epochs": 25,
            "stop_conditions": {
                "max_generations": 50,
                "early_stop_generations": 10,
                "fitness_goal": 0.995,
                "time_limit_minutes": 5,
            },
        },
    }
