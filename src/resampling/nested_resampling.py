from typing import Dict

import numpy as np

from src.ga_runner import run_optimization
from src.logger.logger import logger
from src.tui.tui_screen import TUI
from src.utils.checkpoint_manager import GaState, checkpoint_manager
from src.utils.data_splitter import create_stratified_k_folds


def run_nested_resampling(
    config: Dict,
    tui: TUI,
    session_log_filename: str,
    loaded_state: GaState | None,
):
    """
    Manages the nested resampling process (outer K-fold cross-validation).

    If nested resampling is disabled in the config, it runs a single
    optimization process. Otherwise, it iterates through K folds, running
    the full GA optimization within each to find the best hyperparameters,
    and then evaluates the result on the fold's test set.
    """
    nested_config = config["nested_validation_config"]
    outer_k_folds = nested_config["outer_k_folds"]

    if not nested_config["enabled"] or outer_k_folds <= 1:
        run_optimization(config, tui, session_log_filename, loaded_state)
        return

    logger.info(
        f"--- Starting Nested Resampling With {outer_k_folds} Folds ---"
    )
    data_dir = "./model_data"
    seed = config["project"]["seed"]

    fold_indices = create_stratified_k_folds(data_dir, outer_k_folds, seed)

    all_fold_scores = []
    start_fold = 0

    if loaded_state and loaded_state.outer_fold_k != -1:
        start_fold = loaded_state.outer_fold_k

        if loaded_state.phase_completed:
            logger.info(
                f"Fold {start_fold + 1} was already completed. Resuming from Fold {start_fold + 2}."
            )
            start_fold += 1
        else:
            logger.info(
                f"Resuming nested resampling from Fold {start_fold + 1}/{outer_k_folds}."
            )

    for k in range(start_fold, outer_k_folds):
        tui.update_fold_status(k + 1, outer_k_folds)
        logger.info(f"--- Running Fold {k + 1}/{outer_k_folds} ---")

        train_idx, test_idx = fold_indices[k]

        current_fold_loaded_state = loaded_state if k == start_fold else None

        best_individual, final_fitness, final_loss = run_optimization(
            config=config,
            tui=tui,
            session_log_filename=session_log_filename,
            loaded_state=current_fold_loaded_state,
            outer_fold_k=k,
            train_indices=train_idx,
            test_indices=test_idx,
        )

        logger.info(f"--- Fold {k + 1} Finished ---")
        logger.info(
            f"Best Fitness (Accuracy) for Fold {k + 1}: {final_fitness:.4f}"
        )
        logger.info(f"Best Loss for Fold {k + 1}: {final_loss:.4f}")
        logger.info(f"Best Hyperparameters for Fold {k + 1}: {best_individual}")
        all_fold_scores.append(final_fitness)

        if k < outer_k_folds - 1:
            checkpoint_manager.delete_checkpoint()

    logger.info("--- Nested Resampling Finished ---")

    if all_fold_scores:
        mean_accuracy = np.mean(all_fold_scores)
        std_accuracy = np.std(all_fold_scores)
        logger.info(
            f"Average Accuracy across {outer_k_folds} folds: {mean_accuracy:.4f} (±{std_accuracy:.6f})"
        )
    else:
        logger.warning("No fold scores were recorded.")
