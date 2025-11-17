import os
from typing import Dict, Optional

import numpy as np

from src.ga_runner import run_optimization
from src.logger.logger import logger
from src.tui.tui_screen import TUI
from src.utils.checkpoint_manager import GaState
from src.utils.data_splitter import create_stratified_k_folds
from src.evaluator.create_evaluator import create_evaluator


def run_nested_resampling(
    config: Dict,
    tui: TUI,
    session_log_filename: str,
    loaded_state: Optional[GaState],
    log_queue=None,
):
    """
    Manages the nested resampling process.
    Adapts its behavior based on the SLURM_ARRAY_TASK_ID environment variable.
    """
    nested_config = config["nested_validation_config"]
    outer_k_folds = nested_config["outer_k_folds"]

    fold_to_run_str = os.environ.get("SLURM_ARRAY_TASK_ID")
    is_slurm_array_job = fold_to_run_str is not None

    if is_slurm_array_job:
        try:
            fold_index = int(fold_to_run_str)
            logger.info(
                f"Running as SLURM array task. Executing ONLY Fold {fold_index + 1}/{outer_k_folds}."
            )
            _run_single_fold(
                config, tui, session_log_filename, loaded_state, fold_index, log_queue
            )
        except (ValueError, IndexError):
            logger.error(
                f"Invalid SLURM_ARRAY_TASK_ID: {fold_to_run_str}. Could not run fold."
            )
    else:
        logger.info(
            "Not a SLURM array job. Running all folds sequentially."
        )
        if not nested_config["enabled"] or outer_k_folds <= 1:
            run_optimization(config, tui, session_log_filename, loaded_state)
            return

        logger.info(
            f"--- Starting Sequential Nested Resampling With {outer_k_folds} Folds ---"
        )
        all_fold_scores = []
        start_fold = 0

        if loaded_state and loaded_state.outer_fold_k != -1:
            start_fold = loaded_state.outer_fold_k
            if (
                loaded_state.phase == "main_algorithm"
                and loaded_state.phase_completed
            ):
                logger.info(f"Fold {start_fold + 1} was already completed. Resuming from Fold {start_fold + 2}.")
                start_fold += 1
            else:
                logger.info(f"Resuming nested resampling from Fold {start_fold + 1}/{outer_k_folds}.")

        for k in range(start_fold, outer_k_folds):
            current_fold_loaded_state = (
                loaded_state if k == start_fold else None
            )
            final_fitness, _, _ = _run_single_fold(
                config,
                tui,
                session_log_filename,
                current_fold_loaded_state,
                k,
                log_queue
            )
            if final_fitness is not None:
                all_fold_scores.append(final_fitness)

        logger.info("--- Sequential Nested Resampling Finished ---")
        if all_fold_scores:
            mean_accuracy = np.mean(all_fold_scores)
            std_accuracy = np.std(all_fold_scores)
            logger.info(
                f"Average Accuracy across {outer_k_folds} folds: {mean_accuracy:.4f} (±{std_accuracy:.6f})"
            )


def _run_single_fold(
    config: Dict,
    tui: TUI,
    session_log_filename: str,
    loaded_state: Optional[GaState],
    fold_index: int,
    log_queue=None,
) -> (Optional[float], Optional[float], Optional[dict]):
    """
    Helper function to encapsulate the logic for running a single fold.
    This is now the core logic called by both sequential and parallel modes.
    """
    outer_k_folds = config["nested_validation_config"]["outer_k_folds"]
    tui.update_fold_status(fold_index + 1, outer_k_folds)
    logger.info(f"--- Running Fold {fold_index + 1}/{outer_k_folds} ---")

    base = os.environ.get("SLURM_JOB_ID")
    array_id = os.environ.get("SLURM_ARRAY_TASK_ID", "0")
    data_dir = f"/mnt/lscratch/slurm/{base}/{array_id}"
    os.makedirs(data_dir, exist_ok=True)
    seed = config["project"]["seed"]
    fold_indices_list = create_stratified_k_folds(data_dir, outer_k_folds, seed)
    
    if fold_index >= len(fold_indices_list):
        logger.error(f"Fold index {fold_index} is out of bounds for {len(fold_indices_list)} folds.")
        return None, None, None

    train_idx, test_idx = fold_indices_list[fold_index]
    ga_config = config["genetic_algorithm_config"]
    
    fixed_batch_size = ga_config.get("fixed_batch_size_for_cal", 64)

    with create_evaluator(
        config,
        ga_config["calibration"]["training_epochs"],
        ga_config["calibration"]["stop_conditions"]["early_stop_epochs"],
        ga_config["calibration"].get("data_subset_percentage", 1.0),
        tui,
        session_log_filename,
        train_indices=train_idx,
        test_indices=test_idx,
        fixed_batch_size=fixed_batch_size,
        log_queue=log_queue,
    ) as evaluator:
        best_individual, final_fitness, final_loss = run_optimization(
            config=config,
            tui=tui,
            session_log_filename=session_log_filename,
            loaded_state=loaded_state,
            outer_fold_k=fold_index,
            train_indices=train_idx,
            test_indices=test_idx,
            evaluator=evaluator,
        )

    logger.info(f"--- Fold {fold_index + 1} Finished ---")
    if final_fitness is not None:
        logger.info(
            f"Best Fitness (Accuracy) for Fold {fold_index + 1}: {final_fitness:.4f}"
        )
        logger.info(f"Best Loss for Fold {fold_index + 1}: {final_loss:.4f}")
        logger.info(
            f"Best Hyperparameters for Fold {fold_index + 1}: {best_individual}"
        )

    return final_fitness, final_loss, best_individual
