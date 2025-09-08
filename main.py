import warnings

import signal
import sys
import time
from typing import Dict, List, Tuple

warnings.filterwarnings("ignore")

from src.tui.tui_screen import TUI
from src.utils.file_helper import clear_base_log_file, rename_base_log_file

# Pre-import heavy modules to avoid issues in worker processes
try:
    import torch
    import torchvision
except ImportError:
    pass

from src.factory.create_evaluator import create_evaluator
from src.genetic.genetic_algorithm import (
    GeneticAlgorithm,
    get_chromosome_search_space,
)
from src.genetic.stop_conditions import StopConditions
from src.logger.logger import logger
from src.tui.tui_configurator import run_tui_configurator
from src.utils.seed import seed_everything

MUTATION_DECAY_RATE = 0.98
CAL_STRAT_BINS = 3
MAIN_START_BINS = 5
CAL_PHASE = "calibration"
MAIN_PHASE = "main_algorithm"


def run_ga_phase(
    phase_name: str,
    config: dict,
    ga: GeneticAlgorithm,
    starting_population: List[Dict],
    tui: TUI,
) -> Tuple[List[Dict], float, float]:
    """
    Runs a complete phase (calibration or main) of the genetic algorithm,
    respecting the defined stop conditions.
    """
    phase_config = config["genetic_algorithm_config"][phase_name]
    stop_conditions = StopConditions(phase_config["stop_conditions"])
    early_stop_epochs = phase_config["stop_conditions"]["early_stop_epochs"]
    generations = phase_config["generations"]
    initial_epochs = phase_config["training_epochs"]
    minimum_viable_epochs = 20
    subset_percentage = (
        phase_config["data_subset_percentage"]
        if phase_name == CAL_PHASE
        else 1.0
    )

    logger.info(f"--- Starting {phase_name.upper()} Phase ---")
    logger.info(
        f"Generations: {generations}, Population: {len(starting_population)}"
    )

    population = starting_population
    total_evaluations = generations * len(population)
    phase_task_id = tui.progress.add_task(
        f"{phase_name.capitalize()} Phase", total=total_evaluations
    )

    with create_evaluator(
        config,
        initial_epochs,
        early_stop_epochs,
        subset_percentage,
        tui,
        phase_task_id,
    ) as evaluator:
        try:
            training_epochs = initial_epochs
            enable_progressive_epochs = initial_epochs >= minimum_viable_epochs

            if not enable_progressive_epochs:
                logger.warning(
                    f"Progressive epochs are disabled! To enable progression minimal training epochs must be at least ({minimum_viable_epochs})."
                )

            for gen in range(1, generations + 1):
                tui.update()

                logger.info(
                    f"--- {phase_name.upper()} - Generation {gen}/{generations} ---"
                )

                if enable_progressive_epochs:
                    progress = gen / generations
                    if progress <= 0.3:
                        epoch_multiplier = 0.2
                    elif progress <= 0.7:
                        epoch_multiplier = 0.6
                    else:
                        epoch_multiplier = 1.0

                    training_epochs = int(
                        round(initial_epochs * epoch_multiplier)
                    )

                evaluator.set_training_epochs(training_epochs)

                evaluation_start_time = time.perf_counter()

                results = evaluator.evaluate_population(
                    population, stop_conditions
                )

                evaluation_time = time.perf_counter() - evaluation_start_time

                fitness_scores = [r.fitness for r in results]
                loss_scores = [r.loss for r in results]
                durations = [r.duration_seconds for r in results]

                total_worker_time = sum(durations)
                num_workers = evaluator.num_workers

                total_available_time = evaluation_time * num_workers
                worker_utilization = (
                    (total_worker_time / total_available_time) * 100
                    if total_available_time > 0.0
                    else 0.0
                )
                true_overhead_seconds = max(
                    0.0, total_available_time - total_worker_time
                )

                best_idx = max(
                    range(len(fitness_scores)), key=fitness_scores.__getitem__
                )
                best_fitness = fitness_scores[best_idx]
                best_loss = loss_scores[best_idx]

                logger.info(f" --- Generation ({gen}) report:  ---")
                logger.info(
                    f"Best Fitness: {best_fitness:.4f} | Loss: {best_loss:.4f}"
                )
                logger.info(f"Wall-Clock Time: {evaluation_time:.2f}s")
                logger.info(f"Worker Utilization: {worker_utilization:.2f}%")
                logger.info(f"Total Worker Busy Time: {total_worker_time:.2f}s")
                logger.info(
                    f"Total Worker Idle Time (Overhead): {true_overhead_seconds:.2f}s"
                )

                should_stop, reason = stop_conditions.should_stop_algorithm(
                    gen, best_fitness
                )
                if should_stop:
                    logger.warning(
                        f"Stopping GA for phase '{phase_name}': {reason}"
                    )
                    sorted_indices = sorted(
                        range(len(fitness_scores)),
                        key=fitness_scores.__getitem__,
                        reverse=True,
                    )
                    population = [population[i] for i in sorted_indices]
                    break

                ga.set_adaptive_mutation(MUTATION_DECAY_RATE, gen)
                population = ga.run_generation(population, fitness_scores)

            logger.info("Starting final evaluation for the best population...")
            evaluator.set_training_epochs(initial_epochs)
            is_final = phase_name == MAIN_PHASE

            # Re-evaluate the final population with full epochs to get a final, fair ranking

            final_results = evaluator.evaluate_population(
                population, stop_conditions=None, is_final=is_final
            )
            final_fitness = [r.fitness for r in final_results]
            final_loss = [r.loss for r in final_results]

            sorted_indices = sorted(
                range(len(final_fitness)),
                key=final_fitness.__getitem__,
                reverse=True,
            )
            sorted_population = [population[i] for i in sorted_indices]
            best_fitness = final_fitness[sorted_indices[0]]
            best_loss = final_loss[sorted_indices[0]]

        except KeyboardInterrupt:
            logger.warning(
                f"User interrupted during {phase_name}. Cleaning up..."
            )
            raise
        finally:
            tui.progress.remove_task(phase_task_id)

    logger.info(f"--- {phase_name.upper()} Phase Finished ---")
    return sorted_population, best_fitness, best_loss


def run_optimization(config: Dict, tui: TUI) -> None:
    """
    Main experiment entry point, now with a two-stage (calibration -> main) GA process.
    """
    seed_everything(config["project"]["seed"])

    ga_config = config["genetic_algorithm_config"]
    chromosome_space = get_chromosome_search_space(config)
    ga = GeneticAlgorithm(ga_config["genetic_operators"], chromosome_space)

    calibrated_population = []
    if ga_config[CAL_PHASE]["enabled"]:
        initial_pop_size = ga_config[CAL_PHASE]["population_size"]
        initial_population = ga.initial_population(
            initial_pop_size, CAL_STRAT_BINS
        )
        calibrated_population, best_fitness, best_loss = run_ga_phase(
            CAL_PHASE, config, ga, initial_population, tui
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
                ga.initial_population(
                    num_random, MAIN_START_BINS, print_warning=False
                )
            )
    else:
        logger.info(
            "Calibration disabled. Starting main algorithm with random population."
        )
        main_starting_population = ga.initial_population(
            main_pop_size, MAIN_START_BINS
        )

    final_population, best_fitness, best_loss = run_ga_phase(
        MAIN_PHASE, config, ga, main_starting_population, tui
    )

    logger.info("Full optimization process finished.")
    best_individual = final_population[0]
    logger.info("\n--- Best Overall Result ---")
    logger.info(f"Best Fitness (Accuracy): {best_fitness:.4f}")
    logger.info(f"Corresponding Loss: {best_loss:.4f}")
    logger.info(f"Optimal Hyperparameters: {best_individual}")


def main():
    def sigint_handler(signum, frame):
        logger.info("Main received SIGINT, shutting down gracefully...")
        sys.exit(0)

    # Set up signal handler
    signal.signal(signal.SIGINT, sigint_handler)

    try:
        clear_base_log_file()
        config = run_tui_configurator()

        tui = TUI()
        tui_sink = tui.get_loguru_sink()
        logger.add_tui_sink(tui_sink)

        tui.build_config_panel(config)

        with tui:
            run_optimization(config, tui)

        rename_base_log_file()
    except KeyboardInterrupt:
        logger.info("User terminated the program.")
        sys.exit(0)
    except SystemExit:
        logger.info("Program terminated gracefully.")
    except Exception as e:
        logger.exception(f"Unexpected error occurred {e}")
        sys.exit(1)


if __name__ == "__main__":
    try:
        # Ensure spawn, fork doesn't work well with CUDA
        if sys.platform != "win32":
            torch.multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    main()
