import sys
from pathlib import Path
from typing import List, Dict

import numpy as np

from src.config.default_config import ex
from src.genetic.genetic_algorithm import (
    GeneticAlgorithm,
    get_chromosome_search_space,
)
from src.genetic.individual_evaluator import IndividualEvaluator
from src.genetic.stop_conditions import StopConditions
from src.logger.experiment_logger import logger
from src.tui import run_tui_configurator, print_final_config_panel
from src.utils.seed import seed_everything

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

# --- Main Parameters ---
MUTATION_DECAY_RATE = 0.98
ENABLE_PROGRESSIVE_EPOCHS = True
# Start bins depends on the population size, number of continuous parameters,
# and your desired coverage/granularity of the search space
CAL_STRAT_BINS = 3
MAIN_START_BINS = 5


def run_ga_phase(
    phase_name: str,
    config: dict,
    ga: GeneticAlgorithm,
    starting_population: List[Dict],
) -> List[Dict]:
    """
    Runs a complete phase (calibration or main) of the genetic algorithm,
    respecting the defined stop conditions.
    """
    phase_config = config["genetic_algorithm_config"][phase_name]
    stop_conditions = StopConditions(phase_config["stop_conditions"])
    population = starting_population
    max_generations = stop_conditions.max_generations

    logger.info(f"--- Starting {phase_name.upper()} Phase ---")
    logger.info(
        f"Generations: up to {stop_conditions.max_generations}, Population: {len(starting_population)}"
    )

    for gen in range(1, stop_conditions.max_generations + 1):
        logger.info(
            f"\n{phase_name.upper()} - Generation {gen}/{stop_conditions.max_generations}"
        )

        initial_epochs = phase_config["training_epochs"]
        minimum_viable_epochs = 20

        # Progressive epochs and subset percentage
        if (
            ENABLE_PROGRESSIVE_EPOCHS
            and initial_epochs >= minimum_viable_epochs
        ):
            progress = gen / max_generations

            if progress <= 0.3:
                training_epochs = int(round(initial_epochs * 0.1))
                subset_percentage = 0.3
            elif progress <= 0.7:
                training_epochs = int(round(initial_epochs * 0.4))
                subset_percentage = 0.7
            else:
                training_epochs = int(round(initial_epochs))
                subset_percentage = 1.0
        else:
            logger.warning(
                "Progressive epochs and subset percentage are disabled!"
            )
            training_epochs = initial_epochs
            subset_percentage = (
                phase_config["data_subset_percentage"]
                if phase_name.upper() == "CALIBRATION"
                else 1.0
            )

        evaluator = IndividualEvaluator(
            config=config,
            training_epochs=training_epochs,
            subset_percentage=subset_percentage,
        )

        fitness_scores, loss_scores = evaluator.evaluate_population(
            population, stop_conditions
        )

        best_idx = np.argmax(fitness_scores)
        best_fitness = fitness_scores[best_idx]
        logger.info(f"  Best Fitness (Accuracy): {best_fitness:.4f}")
        logger.info(f"  Best Individual's Loss: {loss_scores[best_idx]:.4f}")

        # Check algorithm-wide stop conditions
        should_stop, reason = stop_conditions.should_stop_algorithm(
            gen, best_fitness
        )
        if should_stop:
            logger.warning(f"Stopping GA for phase '{phase_name}': {reason}")
            sorted_indices = np.argsort(fitness_scores)[::-1]
            population = [population[i] for i in sorted_indices]
            break

        ga.set_adaptive_mutation(MUTATION_DECAY_RATE, gen)
        # TODO: impl maintain diversity function
        population = ga.run_generation(population, fitness_scores)

    logger.success(f"--- {phase_name.upper()} Phase Finished ---")
    return population


@ex.main
def run_optimization(_config, _run):
    """
    Main experiment entry point, now with a two-stage (calibration -> main) GA process.
    """
    print_final_config_panel(_config)

    ga_config = _config["genetic_algorithm_config"]
    chromosome_space = get_chromosome_search_space(_config)
    ga = GeneticAlgorithm(ga_config["genetic_operators"], chromosome_space)

    # --- STAGE 1: CALIBRATION ---
    calibrated_population = []
    if ga_config["calibration"]["enabled"]:
        initial_pop_size = ga_config["calibration"]["population_size"]
        initial_population = ga.initial_population(
            initial_pop_size, CAL_STRAT_BINS
        )
        calibrated_population = run_ga_phase(
            "calibration", _config, ga, initial_population
        )

    # --- STAGE 2: MAIN ALGORITHM ---
    main_pop_size = ga_config["main_algorithm"]["population_size"]
    main_starting_population = []

    if calibrated_population:
        logger.info(
            "Seeding main algorithm with population from calibration phase."
        )
        main_starting_population.extend(calibrated_population)
        if main_pop_size > len(main_starting_population):
            num_to_add = main_pop_size - len(main_starting_population)
            logger.info(
                f"Adding {num_to_add} new random individuals to the population."
            )
            main_starting_population.extend(
                ga.initial_population(num_to_add, MAIN_START_BINS)
            )
        else:
            main_starting_population = main_starting_population[:main_pop_size]
    else:
        logger.info(
            "Calibration disabled. Starting main algorithm with random population."
        )
        main_starting_population = ga.initial_population(
            main_pop_size, MAIN_START_BINS
        )

    final_population = run_ga_phase(
        "main_algorithm", _config, ga, main_starting_population
    )

    logger.success("Full optimization process finished.")

    # Find and display the best result from the final population
    best_individual = final_population[0]
    final_evaluator = IndividualEvaluator(
        _config, ga_config["main_algorithm"]["training_epochs"]
    )
    final_fitness, final_loss = final_evaluator.evaluate_population(
        [best_individual], None, is_final=True
    )

    logger.success("\n--- Best Overall Result ---")
    logger.success(f"Best Fitness (Accuracy): {final_fitness[0]:.4f}")
    logger.success(f"Corresponding Loss: {final_loss[0]:.4f}")
    logger.success(f"Optimal Hyperparameters: {best_individual}")


def main():
    try:
        seed_everything()
        config_overrides = run_tui_configurator()
        if config_overrides is not None:
            ex.run(
                config_updates=config_overrides, options={"--loglevel": "ERROR"}
            )
    except KeyboardInterrupt:
        logger.error("\nUser terminated the program.")
        logger.close()
        sys.exit(0)
    except Exception as e:
        logger.error(
            f"An unexpected critical error occurred: {e}", exc_info=True
        )
        logger.close()
        sys.exit(1)
    else:
        logger.close()


if __name__ == "__main__":
    main()
