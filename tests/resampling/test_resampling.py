from unittest.mock import MagicMock, patch

import pytest

from src.resampling.nested_resampling import (
    _run_single_fold,
    run_nested_resampling,
)


@pytest.fixture
def mock_config():
    """Provides a default configuration dictionary for tests."""
    return {
        "nested_validation_config": {"enabled": True, "outer_k_folds": 5},
        "project": {"seed": 42},
        "genetic_algorithm_config": {
            "calibration": {
                "training_epochs": 1,
                "stop_conditions": {"early_stop_epochs": 1},
                "data_subset_percentage": 1.0,
            },
            "fixed_batch_size_for_cal": 64,
        },
    }


@pytest.fixture
def mock_dependencies():
    """Mocks all external dependencies for nested_resampling."""
    with (
        patch(
            "src.resampling.nested_resampling.run_optimization"
        ) as mock_run_opt,
        patch(
            "src.resampling.nested_resampling.create_stratified_k_folds",
            return_value=[(1, 2)] * 5,
        ) as mock_create_folds,
        patch(
            "src.resampling.nested_resampling.create_evaluator"
        ) as mock_create_evaluator,
        patch("src.resampling.nested_resampling.logger") as mock_logger,
    ):
        mock_create_evaluator.return_value.__enter__.return_value = MagicMock()
        mock_create_evaluator.return_value.__exit__.return_value = None

        yield {
            "run_optimization": mock_run_opt,
            "create_stratified_k_folds": mock_create_folds,
            "create_evaluator": mock_create_evaluator,
            "logger": mock_logger,
        }


def test_run_nested_resampling_slurm_mode(
    mock_config, mock_dependencies, monkeypatch
):
    """Test behavior when running as a SLURM array task."""
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "1")
    mock_dependencies["run_optimization"].return_value = (
        None,
        0.9,
        0.1,
    )  # fitness, loss

    run_nested_resampling(mock_config, MagicMock(), "log.txt", None)

    mock_dependencies["run_optimization"].assert_called_once()
    call_args, call_kwargs = mock_dependencies["run_optimization"].call_args
    assert call_kwargs.get("outer_fold_k") == 1
    mock_dependencies["logger"].info.assert_any_call(
        "Running as SLURM array task. Executing ONLY Fold 2/5."
    )


def test_run_nested_resampling_sequential_mode(mock_config, mock_dependencies):
    """Test behavior when running sequentially (not a SLURM job)."""
    mock_dependencies["run_optimization"].return_value = (None, 0.9, 0.1)

    run_nested_resampling(mock_config, MagicMock(), "log.txt", None)

    assert mock_dependencies["run_optimization"].call_count == 5
    mock_dependencies["logger"].info.assert_any_call(
        "Not a SLURM array job. Running all folds sequentially."
    )
    mock_dependencies["logger"].info.assert_any_call(
        "Average Accuracy across 5 folds: 0.9000 (±0.000000)"
    )


def test_run_nested_resampling_disabled(mock_config, mock_dependencies):
    """Test that it runs the standard optimization if nested resampling is disabled."""
    mock_config["nested_validation_config"]["enabled"] = False

    run_nested_resampling(mock_config, MagicMock(), "log.txt", None)

    mock_dependencies["run_optimization"].assert_called_once()
    call_args, call_kwargs = mock_dependencies["run_optimization"].call_args
    assert "outer_fold_k" not in call_kwargs
    assert "train_indices" not in call_kwargs


def test_run_single_fold_logic(mock_config, mock_dependencies):
    """Test the core logic of running one fold."""
    fold_index = 2
    mock_dependencies["run_optimization"].return_value = (
        "best_chromo",
        0.95,
        0.05,
    )

    final_fitness, final_loss, best_individual = _run_single_fold(
        mock_config, MagicMock(), "log.txt", None, fold_index
    )

    mock_dependencies["create_stratified_k_folds"].assert_called_once_with(
        "./model_data", 5, 42
    )
    mock_dependencies["create_evaluator"].assert_called_once()

    mock_dependencies["run_optimization"].assert_called_once()
    call_args, call_kwargs = mock_dependencies["run_optimization"].call_args
    assert call_kwargs.get("outer_fold_k") == fold_index
    assert call_kwargs.get("train_indices") is not None
    assert call_kwargs.get("test_indices") is not None

    assert final_fitness == 0.95
    assert final_loss == 0.05
    assert best_individual == "best_chromo"

    mock_dependencies["logger"].info.assert_any_call("--- Running Fold 3/5 ---")
    mock_dependencies["logger"].info.assert_any_call(
        "Best Fitness (Accuracy) for Fold 3: 0.9500"
    )
