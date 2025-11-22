from typing import Any, Dict


def get_default_config() -> Dict[str, Any]:
    """
    Configuration settings for the project using Sacred.
    This function returns the experiment configuration as a nested dictionary.
    """

    return {
        "project": {"name": "HPO Paper-Based Experiment", "seed": None},
        # Checkpoints
        "checkpoint_config": {"interval_per_gen": 1},
        # Parallel execution and scheduling
        "parallel_config": {
            "execution": {
                "evaluation_mode": "HYBRID",  # Options: "CPU", "GPU", "HYBRID"
                "enable_parallel": True,
                "gpu_workers": 4,
                "cpu_workers": 4,
                "dataloader_workers": {
                    "per_gpu": 1,
                    "per_cpu": 1,
                },
            },
        },
        # Neural network configuration
        "neural_network_config": {
            "input_shape": [3, 32, 32],  # (channels, height, width)
            "output_classes": 10,
            "conv_blocks": 2,
            # Parameters not subject to optimization
            "fixed_parameters": {
                "activation_function": "relu",  # Options: relu, gelu, leaky_relu
                "base_filters": 20,
            },
            "hyperparameter_space": {
                "width_scale": {
                    "type": "float",
                    "range": [0.7, 1.8],
                    "description": "Scales the number of filters in convolutional layers",
                },
                "activation_fn": {
                    "type": "enum",
                    "values": ["RELU", "LEAKY_RELU", "GELU"],
                    "description": "Activation function",
                },
                "mixup_alpha": {
                    "type": "float",
                    "range": [0.2, 1.0],
                    "description": "Base MixUp alpha value",
                },
                "dropout_rate": {
                    "type": "float",
                    "range": [0.2, 0.6],
                    "description": "Dropout intensity in the first fully connected layer",
                },
                "optimizer_schedule": {
                    "type": "enum",
                    "values": [
                        "SGD_ONECYCLE",
                        "SGD_COSINE",
                        "SGD_EXPONENTIAL",
                    ],
                    # Options: [SGD_COSINE, SGD_ONECYCLE, SGD_EXPONENTIAL, ADAMW_COSINE, ADAMW_ONECYLE, ADAMW_EXPONENTIAL]
                    "description": "Optimizer type + Learning rate scheduler",
                },
                "base_lr": {
                    "type": "float",
                    "range": [0.001, 0.1],
                    "scale": "log",
                    "description": "Base learning rate value, log scale",
                },
                "aug_intensity": {
                    "type": "enum",
                    "values": ["MEDIUM", "STRONG"],
                    # Options: [NONE, LIGHT, MEDIUM, STRONG]
                    "description": "Level of data augmentation",
                },
                "weight_decay": {
                    "type": "float",
                    "range": [1e-4, 1e-3],
                    "scale": "log",
                    "description": "L2 regularization coefficient, log scale",
                },
                "batch_size": {
                    "type": "enum",
                    "values": [128, 256],
                    "description": "Training batch size",
                },
            },
        },
        # Nested validation
        "nested_validation_config": {"enabled": True, "outer_k_folds": 2},
        # Genetic algorithm configuration
        "genetic_algorithm_config": {
            "genetic_operators": {
                "active": [
                    "selection",
                    "mutation",
                    "crossover",
                    "elitism",
                ],  # Options: selection, mutation, crossover, elitism or random
                "selection": {"type": "tournament", "tournament_size": 10},
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
                "population_size": 80,
                "generations": 10,
                "training_epochs": 20,
                "data_subset_percentage": 1.0,
                "mutation_decay_rate": 0.98,
                "stratification_bins": 10,
                "stop_conditions": {
                    "max_generations": 10,
                    "early_stop_generations": 999,
                    "early_stop_epochs": 6,
                    "fitness_goal": 0.99,
                    "time_limit_minutes": 0,
                },
            },
            "main_algorithm": {
                "population_size": 53,
                "generations": 70,
                "training_epochs": 100,
                "mutation_decay_rate": 0.98,
                "stratification_bins": 3,
                "stop_conditions": {
                    "max_generations": 70,
                    "early_stop_generations": 20,
                    "early_stop_epochs": 15,
                    "fitness_goal": 0.99,
                    "time_limit_minutes": 0,
                },
            },
        },
    }
