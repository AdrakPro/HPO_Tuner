import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.evaluator.create_evaluator import create_evaluator
from src.genetic.genetic_algorithm import (
    GeneticAlgorithm,
    get_chromosome_search_space,
)
from src.genetic.stop_conditions import StopConditions
from src.logger.logger import logger
from src.tui.tui_screen import TUI
from src.utils.checkpoint_manager import GaState, checkpoint_manager
from src.utils.seed import seed_everything

CAL_PHASE = "calibration"
MAIN_PHASE = "main_algorithm"


def run_ga_phase(
    phase_name: str,
    config: Dict[str, Any],
    ga: GeneticAlgorithm,
    starting_population: List[Dict],
    tui: TUI,
    session_log_filename: str,
    start_gen: int,
    outer_fold_k: int,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    fitness_scores: List[float] = None,
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
    mutation_decay_rate = phase_config["mutation_decay_rate"]
    minimum_viable_epochs = 20
    subset_percentage = phase_config.get("data_subset_percentage", 1.0)
    checkpoint_interval_per_generation = config["checkpoint_config"][
        "interval_per_gen"
    ]

    logger.info(f"--- Starting {phase_name.upper()} Phase ---")
    logger.info(
        f"Generations: {generations}, Population: {len(starting_population)}"
    )

    population = starting_population
    starting_pop_size = len(population)
    # Keep the final evaluation in evaluation track
    total_evaluations = (generations * starting_pop_size) + starting_pop_size
    completed_evaluations = starting_pop_size * (start_gen - 1)

    phase_task_id = tui.progress.add_task(
        f"{phase_name.capitalize()} Phase",
        total=total_evaluations,
        completed=completed_evaluations,
    )

    gen = start_gen - 1

    with create_evaluator(
        config,
        initial_epochs,
        early_stop_epochs,
        subset_percentage,
        tui,
        phase_task_id,
        session_log_filename,
        train_indices,
        test_indices,
    ) as evaluator:
        try:
            training_epochs = initial_epochs
            enable_progressive_epochs = initial_epochs >= minimum_viable_epochs

            if not enable_progressive_epochs:
                logger.warning(
                    f"Progressive epochs are disabled! To enable progression minimal training epochs must be at least ({minimum_viable_epochs})."
                )

            for gen in range(start_gen, generations + 1):
                tui.update()
                logger.info(
                    f"--- {phase_name.upper()} - Generation {gen}/{generations} ---"
                )

                if gen == start_gen and fitness_scores:
                    logger.info(
                        f"Using fitness scores from loaded checkpoint for this generation."
                    )
                    loss_scores = [0.0] * len(fitness_scores)
                else:
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

                    evaluation_time = (
                        time.perf_counter() - evaluation_start_time
                    )

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

                    logger.info(f" --- Generation ({gen}) report:  ---")
                    logger.info(f"Wall-Clock Time: {evaluation_time:.2f}s")
                    logger.info(
                        f"Worker Utilization: {worker_utilization:.2f}%"
                    )
                    logger.info(
                        f"Total Worker Busy Time: {total_worker_time:.2f}s"
                    )
                    logger.info(
                        f"Total Worker Idle Time (Overhead): {true_overhead_seconds:.2f}s"
                    )

                best_idx = max(
                    range(len(fitness_scores)), key=fitness_scores.__getitem__
                )
                best_fitness = fitness_scores[best_idx]
                best_loss = loss_scores[best_idx]

                logger.info(
                    f"Best Fitness: {best_fitness:.4f} | Loss: {best_loss:.4f}"
                )

                # Save state after a generation if fully processed
                if (
                    checkpoint_interval_per_generation != 0
                    and gen % checkpoint_interval_per_generation == 0
                ):
                    phase_completed = False

                    checkpoint_state = GaState(
                        gen,
                        population,
                        fitness_scores,
                        phase_name,
                        config,
                        session_log_filename,
                        phase_completed,
                        outer_fold_k,
                    )
                    checkpoint_manager.save_checkpoint(checkpoint_state)

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

                ga.set_adaptive_mutation(mutation_decay_rate, gen)
                population = ga.run_generation(population, fitness_scores)

            logger.info("Starting final evaluation for the best population...")
            evaluator.set_training_epochs(initial_epochs)
            is_final = phase_name == MAIN_PHASE

            # Re-evaluate the final population with full epochs to get a final, fair ranking
            final_results = evaluator.evaluate_population(
                population, stop_conditions=None, is_final=is_final
            )
            final_fitness_scores = [r.fitness for r in final_results]
            final_loss_scores = [r.loss for r in final_results]

            sorted_indices = sorted(
                range(len(final_fitness_scores)),
                key=final_fitness_scores.__getitem__,
                reverse=True,
            )
            sorted_population = [population[i] for i in sorted_indices]
            sorted_final_fitness = [
                final_fitness_scores[i] for i in sorted_indices
            ]

            best_fitness = sorted_final_fitness[0]
            best_loss = final_loss_scores[sorted_indices[0]]

        except KeyboardInterrupt:
            logger.warning(
                f"User interrupted during {phase_name}. Cleaning up..."
            )
            raise
        finally:
            tui.progress.remove_task(phase_task_id)

    logger.info(f"--- {phase_name.upper()} Phase Finished ---")

    phase_completed = True
    checkpoint_state = GaState(
        gen,
        population,
        sorted_final_fitness,
        phase_name,
        config,
        session_log_filename,
        phase_completed,
        outer_fold_k,
    )
    checkpoint_manager.save_checkpoint(checkpoint_state)

    return sorted_population, best_fitness, best_loss


def run_optimization(
    config: Dict,
    tui: TUI,
    session_log_filename: str,
    loaded_state: GaState = None,
    outer_fold_k: int = -1,
    train_indices: Optional[np.ndarray] = None,
    test_indices: Optional[np.ndarray] = None,
) -> Tuple[Dict, float, float]:
    """
    Main experiment entry point, now with a two-stage (calibration -> main) GA process.
    """
    if not loaded_state:
        seed_everything(config["project"]["seed"])

    ga_config = config["genetic_algorithm_config"]
    chromosome_space = get_chromosome_search_space(config)
    ga = GeneticAlgorithm(ga_config["genetic_operators"], chromosome_space)

    # Skip calibration if we are resuming from the main phase
    is_calibration_complete = False
    calibrated_population = []

    if loaded_state and loaded_state.outer_fold_k == outer_fold_k:
        if loaded_state.phase == MAIN_PHASE:
            is_calibration_complete = True
            logger.info("Resuming main phase, skipping calibration.")
        elif loaded_state.phase == CAL_PHASE and loaded_state.phase_completed:
            is_calibration_complete = True
            logger.info(
                "Calibration phase already completed. Skipping to main phase."
            )
            calibrated_population = loaded_state.population

    if not is_calibration_complete and ga_config[CAL_PHASE]["enabled"]:
        start_gen_cal = 1
        fitness_scores_cal = None

        if loaded_state and loaded_state.phase == CAL_PHASE:
            logger.info(
                f"Resuming calibration phase from generation {loaded_state.generation}."
            )
            initial_population_cal = loaded_state.population
            start_gen_cal = loaded_state.generation + 1
            fitness_scores_cal = loaded_state.fitness_scores
        else:
            logger.info("Starting new calibration phase.")
            initial_pop_size = ga_config[CAL_PHASE]["population_size"]
            cal_strat_bins = ga_config[CAL_PHASE]["stratification_bins"]
            initial_population_cal = ga.initial_population(
                initial_pop_size, cal_strat_bins
            )

        calibrated_population, best_fitness, best_loss = run_ga_phase(
            CAL_PHASE,
            config,
            ga,
            initial_population_cal,
            tui,
            session_log_filename,
            start_gen_cal,
            outer_fold_k,
            train_indices,
            test_indices,
            fitness_scores_cal,
        )
        logger.info(
            f"The best individual of calibrated individual -> Accuracy: {best_fitness:.4f}, Loss: {best_loss:.4f}"
        )

    # --- STAGE 2: MAIN ALGORITHM ---
    main_starting_population = []
    start_gen_main = 1
    fitness_scores_main = None

    if loaded_state and loaded_state.phase == MAIN_PHASE:
        logger.info(
            f"Resuming main phase from generation {loaded_state.generation}."
        )
        main_starting_population = loaded_state.population
        start_gen_main = loaded_state.generation
        fitness_scores_main = loaded_state.fitness_scores
    else:
        main_pop_size = ga_config[MAIN_PHASE]["population_size"]
        main_strat_bins = ga_config[MAIN_PHASE]["stratification_bins"]

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
                        num_random, main_strat_bins, print_warning=False
                    )
                )
        else:
            logger.info(
                "Calibration disabled. Starting main algorithm with random population."
            )
            main_starting_population = ga.initial_population(
                main_pop_size, main_strat_bins
            )

    final_population, best_fitness, best_loss = run_ga_phase(
        MAIN_PHASE,
        config,
        ga,
        main_starting_population,
        tui,
        session_log_filename,
        start_gen_main,
        outer_fold_k,
        train_indices,
        test_indices,
        fitness_scores_main,
    )

    fold_info = f" for Fold {outer_fold_k + 1}" if outer_fold_k != -1 else ""
    logger.info(f"Optimization{fold_info} finished.")

    return final_population[0], best_fitness, best_loss
