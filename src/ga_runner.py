import random
import time
from copy import deepcopy
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
from src.utils.performance_tracker import performance_tracker, PhaseType

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
    evaluator=None,
) -> Tuple[List[Dict], float, float, bool]:
    """
    Runs a complete phase (calibration or main) of the genetic algorithm,
    respecting the defined stop conditions.
    """

    best_fitness = -float("inf")
    best_loss = float("inf")
    fitness_scores = []
    num_workers = 1

    phase_config = config["genetic_algorithm_config"][phase_name]
    stop_conditions = StopConditions(phase_config["stop_conditions"])
    generations = phase_config["generations"]
    initial_epochs = phase_config["training_epochs"]
    mutation_decay_rate = phase_config["mutation_decay_rate"]
    MINIMUM_PROGRESSIVE_EPOCHS = 20
    checkpoint_interval_per_generation = config["checkpoint_config"][
        "interval_per_gen"
    ]

    logger.info(f"--- Starting {phase_name.upper()} Phase ---")
    logger.info(
        f"Generations: {generations}, Population: {len(starting_population)}"
    )

    phase_type = (
        PhaseType.CALIBRATION if phase_name == "calibration" else PhaseType.MAIN
    )

    population = starting_population
    starting_pop_size = len(population)

    total_evaluations = generations * starting_pop_size

    # Count final evaluation
    if phase_name == CAL_PHASE:
        total_evaluations += starting_pop_size

    completed_evaluations = starting_pop_size * (start_gen - 1)

    phase_task_id = tui.progress.add_task(
        f"{phase_name.capitalize()} Phase",
        total=total_evaluations,
        completed=completed_evaluations,
    )

    evaluator.set_task_id(phase_task_id)

    gen = start_gen - 1

    try:
        training_epochs = initial_epochs
        should_stop = False
        enable_progressive_epochs = (
            phase_name == MAIN_PHASE
            and initial_epochs >= MINIMUM_PROGRESSIVE_EPOCHS
        )
        epoch_multiplier = 1.0

        if not enable_progressive_epochs:
            logger.warning(
                f"Progressive epochs are disabled! To enable progression minimal training epochs must be at least ({MINIMUM_PROGRESSIVE_EPOCHS})."
            )

        for gen in range(start_gen, generations + 1):
            performance_tracker.start_generation(gen, phase_type)

            tui.update()

            logger.info(
                f"--- {phase_name.upper()} - Generation {gen}/{generations} ---"
            )

            seq_prep_start = time.perf_counter()
            if enable_progressive_epochs:
                progress = gen / generations
                if progress <= 0.3:
                    epoch_multiplier = 0.5
                elif progress <= 0.7:
                    epoch_multiplier = 0.75
                else:
                    epoch_multiplier = 1.0

                training_epochs = int(round(initial_epochs * epoch_multiplier))

            evaluator.set_training_epochs(training_epochs)
            seq_prep_time = time.perf_counter() - seq_prep_start
            performance_tracker.record_sequential_time(
                seq_prep_time,
                "generation_preparation",
                {
                    "training_epochs": training_epochs,
                    "epoch_multiplier": (
                        epoch_multiplier if enable_progressive_epochs else 1.0
                    ),
                },
            )

            performance_tracker.start_parallel_section()

            evaluation_start_time = time.perf_counter()

            results = evaluator.evaluate_population(population, stop_conditions)

            evaluation_time = time.perf_counter() - evaluation_start_time

            seq_process_start = time.perf_counter()

            # Safe check if resources are cleanuped
            fitness_scores = [
                r.fitness for r in results if r.fitness is not None
            ]
            loss_scores = [r.loss for r in results if r.loss is not None]
            durations = [
                r.duration_seconds
                for r in results
                if r.duration_seconds is not None
            ]

            total_worker_time = sum(durations) if durations else 0.0
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

            seq_process_time = time.perf_counter() - seq_process_start
            performance_tracker.record_sequential_time(
                seq_process_time,
                "result_processing",
                {"population_size": len(population)},
            )

            logger.info("=" * 40)
            logger.info(f"GENERATION {gen} REPORT")
            logger.info("=" * 40)
            logger.info(f"Wall-Clock Time: {evaluation_time:.2f}s")
            logger.info(f"Worker Utilization: {worker_utilization:.2f}%")
            logger.info(f"Total Worker Busy Time: {total_worker_time:.2f}s")
            logger.info(
                f"Total Worker Idle Time (Overhead): {true_overhead_seconds:.2f}s"
            )

            if fitness_scores and loss_scores:
                best_idx = max(
                    range(len(fitness_scores)), key=fitness_scores.__getitem__
                )
                best_fitness = fitness_scores[best_idx]
                best_loss = loss_scores[best_idx]

                logger.info(
                    f"Best Fitness: {best_fitness:.4f} | Loss: {best_loss:.4f}"
                )
            else:
                logger.warning("No valid fitness scores available")

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

            stop_check_start = time.perf_counter()
            should_stop, reason = stop_conditions.should_stop_algorithm(
                gen, best_fitness
            )
            stop_check_time = time.perf_counter() - stop_check_start
            performance_tracker.record_sequential_time(
                stop_check_time, "stop_condition_check"
            )

            if should_stop:
                logger.warning(
                    f"Stopping GA for phase '{phase_name}': {reason}"
                )
                sort_start = time.perf_counter()
                sorted_indices = sorted(
                    range(len(fitness_scores)),
                    key=fitness_scores.__getitem__,
                    reverse=True,
                )
                population = [population[i] for i in sorted_indices]
                sort_time = time.perf_counter() - sort_start
                performance_tracker.record_sequential_time(
                    sort_time, "population_sorting"
                )

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
                break

            genetic_start = time.perf_counter()
            ga.set_adaptive_mutation(mutation_decay_rate, gen)
            population = ga.run_generation(population, fitness_scores)
            genetic_time = time.perf_counter() - genetic_start
            performance_tracker.record_sequential_time(
                genetic_time,
                "genetic_operations",
                {"mutation_decay_rate": mutation_decay_rate, "generation": gen},
            )

            performance_tracker.end_generation(
                num_workers=num_workers,
                best_fitness=best_fitness,
                worker_utilization=worker_utilization,
                population_size=len(population),
            )

            performance_tracker.print_generation_report(gen)

        if phase_name == CAL_PHASE:
            logger.info(
                "Starting final evaluation with full data and long epochs for the best population..."
            )

            evaluator.set_training_epochs(
                config["genetic_algorithm_config"][MAIN_PHASE][
                    "training_epochs"
                ]
            )
            evaluator.set_subset_percentage(1.0)

            is_final = True

            performance_tracker.start_generation(gen + 1, phase_type)
            performance_tracker.start_parallel_section()

            final_start_time = time.perf_counter()
            final_results = evaluator.evaluate_population(
                population, stop_conditions=None, is_final=is_final
            )
            final_evaluation_time = time.perf_counter() - final_start_time

            final_process_start = time.perf_counter()
            final_fitness_scores = [
                r.fitness for r in final_results if r.fitness is not None
            ]
            final_loss_scores = [
                r.loss for r in final_results if r.loss is not None
            ]

            if final_fitness_scores and final_loss_scores:
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
            else:
                logger.warning("No valid final fitness scores available")
                sorted_population = population

            final_process_time = time.perf_counter() - final_process_start
            performance_tracker.record_sequential_time(
                final_process_time, "final_processing"
            )

            final_durations = [
                r.duration_seconds
                for r in final_results
                if r.duration_seconds is not None
            ]
            final_total_worker_time = (
                sum(final_durations) if final_durations else 0
            )
            final_total_available_time = final_evaluation_time * num_workers
            final_worker_utilization = (
                (final_total_worker_time / final_total_available_time) * 100
                if final_total_available_time > 0.0
                else 0.0
            )

            performance_tracker.end_generation(
                num_workers=num_workers,
                best_fitness=best_fitness,
                worker_utilization=final_worker_utilization,
                population_size=len(population),
            )

            performance_tracker.print_generation_report(gen)

            if not should_stop:
                phase_completed = True
                checkpoint_state = GaState(
                    gen,
                    sorted_population,
                    (
                        sorted_final_fitness
                        if "sorted_final_fitness" in locals()
                        else []
                    ),
                    phase_name,
                    config,
                    session_log_filename,
                    phase_completed,
                    outer_fold_k,
                )
                checkpoint_manager.save_checkpoint(checkpoint_state)

            return sorted_population, best_fitness, best_loss, should_stop

    except KeyboardInterrupt:
        logger.warning(f"User interrupted during {phase_name}. Cleaning up...")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in {phase_name}: {e}")
        raise
    finally:
        tui.progress.remove_task(phase_task_id)

    logger.info(f"--- {phase_name.upper()} Phase Finished ---")

    phase_completed = True
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

    return population, best_fitness, best_loss


def run_optimization(
    config: Dict,
    tui: TUI,
    session_log_filename: str,
    loaded_state: GaState = None,
    outer_fold_k: int = -1,
    train_indices: Optional[np.ndarray] = None,
    test_indices: Optional[np.ndarray] = None,
    evaluator=None,
) -> Tuple[Dict, float, float]:
    """
    Main experiment entry point, now with a two-stage (calibration -> main) GA process.
    """
    if not loaded_state:
        seed_everything(config["project"]["seed"])

    ga_config = config["genetic_algorithm_config"]
    chromosome_space = get_chromosome_search_space(config)
    ga = GeneticAlgorithm(ga_config["genetic_operators"], chromosome_space)

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

    close_evaluator = False

    if not is_calibration_complete and ga_config[CAL_PHASE]["enabled"]:
        fixed_batch_size = 64
    else:
        fixed_batch_size = None

    if evaluator is None:
        evaluator = create_evaluator(
            config,
            ga_config[CAL_PHASE]["training_epochs"],
            ga_config[CAL_PHASE]["stop_conditions"]["early_stop_epochs"],
            ga_config[CAL_PHASE].get("data_subset_percentage", 1.0),
            tui,
            session_log_filename,
            train_indices,
            test_indices,
            fixed_batch_size,
        ).__enter__()
        close_evaluator = True

    try:
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

            calibrated_population, best_fitness, best_loss, _ = run_ga_phase(
                CAL_PHASE,
                config,
                ga,
                initial_population_cal,
                tui,
                session_log_filename,
                start_gen_cal,
                outer_fold_k,
                evaluator,
            )
            logger.info(
                f"The best individual of calibrated individual -> Accuracy: {best_fitness:.4f}, Loss: {best_loss:.4f}"
            )

        main_starting_population = []
        start_gen_main = 1
        fitness_scores_main = None

        if loaded_state and loaded_state.phase == MAIN_PHASE:
            logger.info(
                f"Resuming main phase from generation {loaded_state.generation + 1}."
            )
            main_starting_population = loaded_state.population
            # TODO What if i am resuming for the last one, would generation exceeded?
            if (
                loaded_state.generation + 1
                <= loaded_state.config["genetic_algorithm_config"][
                    "main_algorithm"
                ]["generations"]
            ):
                start_gen_main = loaded_state.generation + 1
            else:
                start_gen_main = loaded_state.generation
            fitness_scores_main = loaded_state.fitness_scores
        else:
            main_pop_size = ga_config[MAIN_PHASE]["population_size"]
            main_strat_bins = ga_config[MAIN_PHASE]["stratification_bins"]

            if calibrated_population:
                logger.info(
                    "Seeding main algorithm with population from calibration phase."
                )
                num_elites = int(main_pop_size * 0.4)
                num_mutated = int(main_pop_size * 0.3)
                num_random = main_pop_size - num_mutated - num_elites

                elites = calibrated_population[:num_elites]
                main_starting_population.extend(deepcopy(elites))

                logger.info(
                    f"Transferring top {len(elites)} individuals from calibration."
                )

                mutated_offspring = []
                for _ in range(num_mutated):
                    parent = random.choice(elites)
                    child = ga.mutate(deepcopy(parent))
                    mutated_offspring.append(child)
                main_starting_population.extend(mutated_offspring)

                logger.info(
                    f"Adding {num_mutated} mutated offspring from elites."
                )

                if num_random > 0:
                    random_individuals = ga.initial_population(
                        num_random, strat_bins=5, print_warning=False
                    )
                    main_starting_population.extend(random_individuals)
                    logger.info(
                        f"Adding {num_random} fresh random individuals for diversity."
                    )
            else:
                logger.info(
                    "Calibration disabled. Starting main algorithm with random population."
                )
                main_starting_population = ga.initial_population(
                    main_pop_size, main_strat_bins
                )

        final_population, best_fitness, best_loss, should_stop = run_ga_phase(
            MAIN_PHASE,
            config,
            ga,
            main_starting_population,
            tui,
            session_log_filename,
            start_gen_main,
            outer_fold_k,
            evaluator,
        )

        if should_stop and outer_fold_k != -1:
            return (
                final_population[0] if final_population else {},
                best_fitness,
                best_loss,
            )
    finally:
        if close_evaluator:
            evaluator.__exit__(None, None, None)

    fold_info = f" for Fold {outer_fold_k + 1}" if outer_fold_k != -1 else ""
    logger.info(f"Optimization{fold_info} finished.")
    performance_tracker.print_overall_report()

    return final_population[0], best_fitness, best_loss
