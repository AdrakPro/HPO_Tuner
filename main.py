import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from src.factory.create_evaluator import create_evaluator
from src.genetic.genetic_algorithm import (
    GeneticAlgorithm,
    get_chromosome_search_space,
)
from src.genetic.stop_conditions import StopConditions
from src.logger.logger import logger
from src.tui import print_final_config_panel, run_tui_configurator
from src.utils.seed import seed_everything

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

# --- Main Parameters ---
MUTATION_DECAY_RATE = 0.98
ENABLE_PROGRESSIVE_EPOCHS = True
# Start bins depends on the population size, number of continuous parameters,
# and your desired coverage/granularity of the search space
CAL_STRAT_BINS = 3
MAIN_START_BINS = 5

CAL_PHASE = "calibration"
MAIN_PHASE = "main_algorithm"


def run_ga_phase(
    phase_name: str,
    config: dict,
    ga: GeneticAlgorithm,
    starting_population: List[Dict],
) -> Tuple[List[Dict], float, float]:
    """
    Runs a complete phase (calibration or main) of the genetic algorithm,
    respecting the defined stop conditions.
    """
    phase_config = config["genetic_algorithm_config"][phase_name]
    stop_conditions = StopConditions(phase_config["stop_conditions"])
    early_stop_epochs = phase_config["stop_conditions"]["early_stop_epochs"]

    population = starting_population
    max_generations = stop_conditions.max_generations

    initial_epochs = phase_config["training_epochs"]
    minimum_viable_epochs = 20

    subset_percentage = (
        phase_config["data_subset_percentage"]
        if phase_name == CAL_PHASE
        else 1.0
    )

    logger.info(f"--- Starting {phase_name.upper()} Phase ---")
    logger.info(
        f"Generations: up to {stop_conditions.max_generations}, Population: {len(starting_population)}"
    )

    evaluator = create_evaluator(
        config, initial_epochs, early_stop_epochs, subset_percentage
    )
    try:
        for gen in range(1, stop_conditions.max_generations + 1):
            logger.info(
                f"{phase_name.upper()} - Generation {gen}/{stop_conditions.max_generations}"
            )

            if (
                ENABLE_PROGRESSIVE_EPOCHS
                and initial_epochs >= minimum_viable_epochs
            ):
                progress = gen / max_generations
                if progress <= 0.3:
                    epoch_multiplier = 0.2
                elif progress <= 0.7:
                    epoch_multiplier = 0.6
                else:
                    epoch_multiplier = 1.0
                training_epochs = int(round(initial_epochs * epoch_multiplier))
            else:
                training_epochs = initial_epochs
                logger.warning("Progressive epochs are disabled!")

            evaluator.set_training_epochs(training_epochs)

            fitness_scores, loss_scores = evaluator.evaluate_population(
                population, stop_conditions
            )

            best_idx = np.argmax(fitness_scores)
            best_fitness = fitness_scores[best_idx]
            logger.info(f"  Best Fitness (Accuracy): {best_fitness:.4f}")
            logger.info(
                f"  Best Individual's Loss: {loss_scores[best_idx]:.4f}"
            )

            should_stop, reason = stop_conditions.should_stop_algorithm(
                gen, best_fitness
            )
            if should_stop:
                logger.warning(
                    f"Stopping GA for phase '{phase_name}': {reason}"
                )
                sorted_indices = np.argsort(fitness_scores)[::-1]
                population = [population[i] for i in sorted_indices]
                break

            ga.set_adaptive_mutation(MUTATION_DECAY_RATE, gen)
            population = ga.run_generation(population, fitness_scores)

        # Re-evaluate the final population with full epochs to get a final, fair ranking
        evaluator.set_training_epochs(initial_epochs)
        # TODO: Helper function if progressive data subsetting
        # evaluator.set_subset_percentage(subset_percentage)

        # Save the best trained model
        is_final = phase_name == MAIN_PHASE

        final_fitness, final_loss = evaluator.evaluate_population(
            population, stop_conditions=None, is_final=is_final
        )

        sorted_indices = np.argsort(final_fitness)[::-1]
        sorted_population = [population[i] for i in sorted_indices]
        best_fitness = final_fitness[sorted_indices[0]]
        best_loss = final_loss[sorted_indices[0]]

    except KeyboardInterrupt:
        logger.warning(f"User interrupted during {phase_name}. Cleaning up...")
        if hasattr(evaluator, "cleanup_workers"):
            evaluator.cleanup_workers()
        raise

    logger.info(f"--- {phase_name.upper()} Phase Finished ---")
    return sorted_population, best_fitness, best_loss


def run_optimization(config):
    """
    Main experiment entry point, now with a two-stage (calibration -> main) GA process.
    """
    seed_everything(config["project"]["seed"])
    print_final_config_panel(config)

    ga_config = config["genetic_algorithm_config"]
    chromosome_space = get_chromosome_search_space(config)
    ga = GeneticAlgorithm(ga_config["genetic_operators"], chromosome_space)

    # --- STAGE 1: CALIBRATION ---
    calibrated_population = []
    if ga_config[CAL_PHASE]["enabled"]:
        initial_pop_size = ga_config[CAL_PHASE]["population_size"]
        initial_population = ga.initial_population(
            initial_pop_size, CAL_STRAT_BINS
        )
        calibrated_population, best_fitness, best_loss = run_ga_phase(
            CAL_PHASE, config, ga, initial_population
        )
        logger.info(
            f"The best individual of calibrated individual -> -> Accuracy: {best_fitness:.4f}, Loss: {best_loss:.4f}"
        )

    # --- STAGE 2: MAIN ALGORITHM ---
    main_pop_size = ga_config[MAIN_PHASE]["population_size"]
    main_starting_population = []

    if calibrated_population:
        logger.info(
            "Seeding main algorithm with population from calibration phase."
        )
        num_elites = int(main_pop_size * 0.9)
        num_random = main_pop_size - num_elites

        elites = calibrated_population[:num_elites]
        main_starting_population.extend(elites)
        logger.info(
            f"Transferring top {len(elites)} individuals from calibration."
        )

        if num_random > 0:
            logger.info(
                f"Adding {num_random} new random individuals for diversity."
            )
            main_starting_population.extend(
                ga.initial_population(num_random, MAIN_START_BINS)
            )
    else:
        logger.info(
            "Calibration disabled. Starting main algorithm with random population."
        )
        main_starting_population = ga.initial_population(
            main_pop_size, MAIN_START_BINS
        )

    final_population, best_fitness, best_loss = run_ga_phase(
        MAIN_PHASE, config, ga, main_starting_population
    )

    logger.info("Full optimization process finished.")

    best_individual = final_population[0]

    logger.info("\n--- Best Overall Result ---")
    logger.info(f"Best Fitness (Accuracy): {best_fitness:.4f}")
    logger.info(f"Corresponding Loss: {best_loss:.4f}")
    logger.info(f"Optimal Hyperparameters: {best_individual}")


def main():
    try:
        config = run_tui_configurator()
        run_optimization(config)
    except KeyboardInterrupt:
        logger.info("User terminated the program.")
        sys.exit(0)
    except Exception as e:
        logger.error(f"An unexpected critical error occurred: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
