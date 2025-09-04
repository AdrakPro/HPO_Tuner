from src.genetic.individual_evaluator import IndividualEvaluator
from src.logger.logger import logger
from src.parallel.parallel_evaluator import ParallelEvaluator


def create_evaluator(
    config: dict,
    training_epochs: int,
    early_stop_epochs: int,
    subset_percentage: float,
):
    """
    Factory function to create the appropriate evaluator based on config.
    """
    enable_parallel: bool = config["parallel_config"]["execution"][
        "enable_parallel"
    ]

    if enable_parallel:
        logger.info("Using parallel evaluator.")
        return ParallelEvaluator(
            config,
            training_epochs,
            early_stop_epochs,
            subset_percentage,
        )
    else:
        logger.info("Using sequential evaluator.")
        return IndividualEvaluator(
            config, training_epochs, early_stop_epochs, subset_percentage
        )
