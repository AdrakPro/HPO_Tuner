from typing import Any, Dict


def get_default_config() -> Dict[str, Any]:
    """
    Configuration settings for the project using Sacred.
    This function returns the experiment configuration as a nested dictionary.
    """

    return {
        "project": {"name": "HPO Paper-Based Experiment", "seed": None},
        # Environment settings
        "environment": {
            "data_source": {"dataset": "CIFAR-10", "path": "./model_data"},
            "logging": {
                "log_file": "logs/experiment.log",
                "error_log_file": "logs/error.log",
                "report_directory": "reports",
            },
        },
        # Parallel execution and scheduling
        "parallel_config": {
            "execution": {
                "evaluation_mode": "HYBRID",  # Options: "CPU", "GPU", "HYBRID"
                "enable_parallel": True,
                "gpu_workers": 1,
                "cpu_workers": 12,
                # TODO: Dynamic allocation, cpu oversubscription, set_threads
                "dataloader_workers": {
                    "per_gpu": 4,
                    "per_cpu": 2,
                },
            },
            "scheduling": {
                "min_job_duration_seconds": 300,
                "checkpoint_interval": 2,
            },
            "monitoring": {
                "enable_metrics": True,
                "track_resources": True,
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
                # TODO: Maybe this should also be available for search with width_scale
                "base_filters": 32,
            },
            "hyperparameter_space": {
                "width_scale": {
                    "type": "float",
                    "range": [0.75, 2.0],
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
                        "ADAMW_ONECYCLE",
                        "ADAMW_COSINE",
                        "ADAMW_EXPONENTIAL",
                    ],
                    # Options: [SGD_COSINE, SGD_ONECYCLE, SGD_EXPONENTIAL, ADAMW_COSINE, ADAMW_ONECYLE, ADAMW_EXPONENTIAL]
                    "description": "Optimizer type + Learning rate scheduler",
                },
                "base_lr": {
                    "type": "float",
                    "range": [0.01, 0.1],
                    "scale": "log",
                    "description": "Base learning rate value, log scale",
                },
                "aug_intensity": {
                    "type": "enum",
                    "values": ["NONE", "LIGHT", "MEDIUM", "STRONG"],
                    # Options: [NONE, LIGHT, MEDIUM, STRONG]
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
        },
        # Nested validation
        "nested_validation_config": {"enabled": True, "outer_k_folds": 3},
        # Genetic algorithm configuration
        "genetic_algorithm_config": {
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
                "population_size": 40,
                "generations": 8,
                "training_epochs": 25,
                "data_subset_percentage": 0.2,
                "mutation_decay_rate": 0.98,
                "stratification_bins": 3,
                "stop_conditions": {
                    "max_generations": 8,
                    "early_stop_generations": 3,
                    "early_stop_epochs": 5,
                    "fitness_goal": 0.75,
                    "time_limit_minutes": 0,
                },
            },
            "main_algorithm": {
                "population_size": 20,
                "generations": 15,
                "training_epochs": 80,
                "mutation_decay_rate": 0.98,
                "stratification_bins": 5,
                "stop_conditions": {
                    "max_generations": 15,
                    "early_stop_generations": 5,
                    "early_stop_epochs": 10,
                    "fitness_goal": 0.945,
                    "time_limit_minutes": 0,
                },
            },
        },
    }
