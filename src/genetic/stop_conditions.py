"""
Manages the stopping conditions for the genetic algorithm.
"""

import time
from typing import List


# TODO: clean max_generations
class StopConditions:
    """
    Encapsulates the logic for stopping the genetic algorithm based on configured criteria.
    """

    def __init__(self, config: dict):
        """
        Initializes the StopConditions with parameters from the configuration.

        Args:
            config: A dictionary containing stop condition parameters, e.g.,
                    'max_generations', 'early_stop_generations', 'fitness_goal', 'time_limit_minutes'.
        """
        self.early_stop_generations: int = config["early_stop_generations"]
        self.fitness_goal: float = config["fitness_goal"]
        self.time_limit_minutes: float | None = config["time_limit_minutes"]

        self.start_time = time.monotonic()
        self.best_fitness_history: List[float] = []

    def should_stop_evaluation(
        self, individual_fitness: float
    ) -> tuple[bool, str]:
        """
        Checks if the evaluation of the current generation should stop because
        an individual has reached the fitness goal.

        Args:
            individual_fitness: The fitness of the most recently evaluated individual.

        Returns:
            True if the fitness goal is met or exceeded, False otherwise.
        """
        if individual_fitness >= self.fitness_goal:
            return (
                True,
                f"Fitness goal of {self.fitness_goal} reached. Stopping evaluation for current generation and proceeding to the next.",
            )
        return False, ""

    def should_stop_algorithm(
        self, current_generation: int, best_fitness: float
    ) -> tuple[bool, str]:
        """
        Checks if the main genetic algorithm loop should terminate.

        Args:
            current_generation: The current generation number (1-based).
            best_fitness: The best fitness achieved in the latest generation.

        Returns:
            A tuple (should_stop, reason).
        """
        self.best_fitness_history.append(best_fitness)

        # 1. Max generations
        # if current_generation >= self.max_generations:
        #     return True, f"Reached max generations ({self.max_generations})."

        # 2. Time limit
        if self.time_limit_minutes != 0:
            elapsed_time_minutes = (time.monotonic() - self.start_time) / 60
            if elapsed_time_minutes >= self.time_limit_minutes:
                return (
                    True,
                    f"Exceeded time limit of {self.time_limit_minutes} minutes.",
                )

        # 3. Early stopping
        if current_generation > self.early_stop_generations:
            # Look at the last `early_stop_generations` fitness values
            recent_history = self.best_fitness_history[
                -self.early_stop_generations :
            ]
            if len(set(recent_history)) == 1:
                return True, (
                    f"No improvement in best fitness for the last "
                    f"{self.early_stop_generations} generations."
                )

        return False, ""
