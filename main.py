import sys
from pathlib import Path
import numpy as np
from typing import List, Dict, Any, Tuple

from src.config.default_config import ex
from src.genetic.genetic_algorithm import (
    GeneticAlgorithm,
    get_chromosome_search_space,
)
from src.model.chromosome import Chromosome
from src.nn.train_and_eval import train_and_eval
from src.tui import run_tui_configurator, print_final_config_panel
from src.logger.experiment_logger import logger
from src.utils.seed import seed_everything

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))


def evaluate_population(
    population: List[Dict], config: Dict[str, Any], training_epochs: int
) -> Tuple[List[float], List[float]]:
    """
    Fitness function for the GA. Evaluates each individual.
    """
    fitness_scores, loss_scores = [], []
    for i, individual_dict in enumerate(population):
        logger.info(
            f"Evaluating Individual {i+1}/{len(population)} ({training_epochs} epochs)"
        )
        logger.info(f"Hyperparameters: {individual_dict}")
        try:
            chromosome = Chromosome.from_dict(individual_dict)
            accuracy, loss = train_and_eval(chromosome, config, training_epochs)
            logger.info(
                f"Individual {i+1} -> Accuracy: {accuracy:.4f}, Loss: {loss:.4f}"
            )
            fitness_scores.append(accuracy)
            loss_scores.append(loss)
        except Exception as e:
            logger.error(f"Error evaluating individual {i+1}: {e}")
            fitness_scores.append(0.0)
            loss_scores.append(float("inf"))
    return fitness_scores, loss_scores


def run_ga_phase(
    phase_name: str,
    config: dict,
    ga: GeneticAlgorithm,
    starting_population: List[Dict],
) -> List[Dict]:
    """
    Runs a complete phase (calibration or main) of the genetic algorithm.
    """
    phase_config = config["genetic_algorithm_config"][phase_name]
    num_generations = phase_config["generations"]
    training_epochs = phase_config["training_epochs"]

    logger.info(f"--- Starting {phase_name.upper()} Phase ---")
    logger.info(
        f"Generations: {num_generations}, Population: {len(starting_population)}, Training Epochs: {training_epochs}"
    )

    population = starting_population
    for gen in range(num_generations):
        logger.info(
            f"\n{phase_name.upper()} - Generation {gen + 1}/{num_generations}"
        )
        fitness_scores, loss_scores = evaluate_population(
            population, config, training_epochs
        )

        best_idx = np.argmax(fitness_scores)
        logger.info(
            f"  Best Fitness (Accuracy): {fitness_scores[best_idx]:.4f}"
        )
        logger.info(f"  Best Individual's Loss: {loss_scores[best_idx]:.4f}")

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
        initial_population = ga.initial_population(initial_pop_size)
        calibrated_population = run_ga_phase(
            "calibration", _config, ga, initial_population
        )

    # --- STAGE 2: MAIN ALGORITHM ---
    main_pop_size = ga_config["main_algorithm"]["population_size"]

    # Create the starting population for the main phase
    if calibrated_population:
        logger.info(
            "Seeding main algorithm with population from calibration phase."
        )
        # Sort the calibrated population by fitness to select the best individuals
        calib_fitness, _ = evaluate_population(
            calibrated_population,
            _config,
            ga_config["calibration"]["training_epochs"],
        )
        sorted_indices = np.argsort(calib_fitness)[::-1]  # Descending order

        main_starting_population = [
            calibrated_population[i] for i in sorted_indices
        ]

        if main_pop_size > len(main_starting_population):
            # If main pop is larger, fill the rest with new random individuals
            num_to_add = main_pop_size - len(main_starting_population)
            main_starting_population.extend(ga.initial_population(num_to_add))
        else:
            # If main pop is smaller or equal, take the top N best individuals
            main_starting_population = main_starting_population[:main_pop_size]
    else:
        # If no calibration, start the main algorithm with a fresh random population
        logger.info(
            "Calibration disabled. Starting main algorithm with random population."
        )
        main_starting_population = ga.initial_population(main_pop_size)

    final_population = run_ga_phase(
        "main_algorithm", _config, ga, main_starting_population
    )

    logger.success("Full optimization process finished.")

    # Final evaluation to find and display the best result
    final_fitness, final_loss = evaluate_population(
        final_population,
        _config,
        ga_config["main_algorithm"]["training_epochs"],
    )
    best_idx = np.argmax(final_fitness)

    logger.success("\n--- Best Overall Result ---")
    logger.success(f"Best Fitness (Accuracy): {final_fitness[best_idx]:.4f}")
    logger.success(f"Corresponding Loss: {final_loss[best_idx]:.4f}")
    logger.success(f"Optimal Hyperparameters: {final_population[best_idx]}")


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
        sys.exit(0)
    except Exception as e:
        logger.error(
            f"An unexpected critical error occurred: {e}", exc_info=True
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
