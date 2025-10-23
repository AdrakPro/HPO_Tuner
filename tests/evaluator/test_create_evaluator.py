from unittest.mock import Mock

import numpy as np
import pytest

from src.evaluator.create_evaluator import create_evaluator

BASE_PATH = "src.evaluator.create_evaluator"
INDIVIDUAL_EVAL_PATH = f"{BASE_PATH}.IndividualEvaluator"
PARALLEL_EVAL_PATH = f"{BASE_PATH}.ParallelEvaluator"
CPU_STRAT_PATH = f"{BASE_PATH}.CPUOnlyStrategy"
GPU_STRAT_PATH = f"{BASE_PATH}.GPUOnlyStrategy"
HYBRID_STRAT_PATH = f"{BASE_PATH}.HybridStrategy"
LOGGER_PATH = f"{BASE_PATH}.logger"


@pytest.fixture
def mock_tui():
    """Provides a mock TUI object with a progress attribute."""
    tui = Mock(name="MockTUI")
    tui.progress = Mock(name="MockProgress")
    return tui


@pytest.fixture
def base_config():
    """Provides a base config dictionary."""
    return {
        "parallel_config": {
            "execution": {
                "enable_parallel": False,
                "evaluation_mode": "CPU",
                "gpu_workers": 1,
                "cpu_workers": 4,
            }
        },
        "neural_network_config": {"model": "test_net", "layers": 3},
    }


@pytest.fixture
def common_args(mock_tui):
    """Provides common arguments for the create_evaluator function."""
    train_indices = np.array([1, 2, 3])
    test_indices = np.array([4, 5, 6])
    return {
        "training_epochs": 100,
        "early_stop_epochs": 10,
        "subset_percentage": 1.0,
        "tui": mock_tui,
        "session_log_filename": "test_session.log",
        "train_indices": train_indices,
        "test_indices": test_indices,
        "fixed_batch_size": 32,
    }


def test_create_evaluator_parallel_disabled(
    base_config, common_args, mock_tui, mocker
):
    """
    Tests that IndividualEvaluator is created when parallel execution is disabled.
    """
    mock_individual_eval = mocker.patch(INDIVIDUAL_EVAL_PATH)
    mock_parallel_eval = mocker.patch(PARALLEL_EVAL_PATH)

    config = base_config
    config["parallel_config"]["execution"]["enable_parallel"] = False
    nn_config = config["neural_network_config"]
    args = common_args

    evaluator = create_evaluator(config=config, **args)

    mock_individual_eval.assert_called_once_with(
        neural_config=nn_config,
        training_epochs=args["training_epochs"],
        early_stop_epochs=args["early_stop_epochs"],
        subset_percentage=args["subset_percentage"],
        progress=mock_tui.progress,
        train_indices=args["train_indices"],
        test_indices=args["test_indices"],
    )
    mock_parallel_eval.assert_not_called()
    assert evaluator == mock_individual_eval.return_value


def test_create_evaluator_parallel_cpu_mode(
    base_config, common_args, mock_tui, mocker
):
    """
    Tests that ParallelEvaluator is created with CPUOnlyStrategy
    for 'CPU' mode.
    """
    mock_individual_eval = mocker.patch(INDIVIDUAL_EVAL_PATH)
    mock_parallel_eval = mocker.patch(PARALLEL_EVAL_PATH)
    mock_cpu_strategy = mocker.patch(CPU_STRAT_PATH)
    mock_gpu_strategy = mocker.patch(GPU_STRAT_PATH)
    mock_hybrid_strategy = mocker.patch(HYBRID_STRAT_PATH)

    config = base_config
    config["parallel_config"]["execution"]["enable_parallel"] = True
    config["parallel_config"]["execution"]["evaluation_mode"] = "CPU"
    args = common_args

    evaluator = create_evaluator(config=config, **args)

    mock_cpu_strategy.assert_called_once_with()
    mock_gpu_strategy.assert_not_called()
    mock_hybrid_strategy.assert_not_called()

    mock_parallel_eval.assert_called_once_with(
        config=config,
        training_epochs=args["training_epochs"],
        early_stop_epochs=args["early_stop_epochs"],
        subset_percentage=args["subset_percentage"],
        strategy=mock_cpu_strategy.return_value,
        progress=mock_tui.progress,
        session_log_filename=args["session_log_filename"],
        train_indices=args["train_indices"],
        test_indices=args["test_indices"],
        fixed_batch_size=args["fixed_batch_size"],
    )
    mock_individual_eval.assert_not_called()
    assert evaluator == mock_parallel_eval.return_value


def test_create_evaluator_parallel_gpu_mode(
    base_config, common_args, mock_tui, mocker
):
    """
    Tests that ParallelEvaluator is created with GPUOnlyStrategy
    for 'GPU' mode.
    """
    mock_individual_eval = mocker.patch(INDIVIDUAL_EVAL_PATH)
    mock_parallel_eval = mocker.patch(PARALLEL_EVAL_PATH)
    mock_cpu_strategy = mocker.patch(CPU_STRAT_PATH)
    mock_gpu_strategy = mocker.patch(GPU_STRAT_PATH)
    mock_hybrid_strategy = mocker.patch(HYBRID_STRAT_PATH)

    config = base_config
    config["parallel_config"]["execution"]["enable_parallel"] = True
    config["parallel_config"]["execution"]["evaluation_mode"] = "GPU"
    args = common_args

    evaluator = create_evaluator(config=config, **args)

    mock_gpu_strategy.assert_called_once_with()
    mock_cpu_strategy.assert_not_called()
    mock_hybrid_strategy.assert_not_called()

    mock_parallel_eval.assert_called_once_with(
        config=config,
        training_epochs=args["training_epochs"],
        early_stop_epochs=args["early_stop_epochs"],
        subset_percentage=args["subset_percentage"],
        strategy=mock_gpu_strategy.return_value,
        progress=mock_tui.progress,
        session_log_filename=args["session_log_filename"],
        train_indices=args["train_indices"],
        test_indices=args["test_indices"],
        fixed_batch_size=args["fixed_batch_size"],
    )
    mock_individual_eval.assert_not_called()
    assert evaluator == mock_parallel_eval.return_value


def test_create_evaluator_parallel_hybrid_mode(
    base_config, common_args, mock_tui, mocker
):
    """
    Tests that ParallelEvaluator is created with HybridStrategy
    for 'HYBRID' mode.
    """
    mock_individual_eval = mocker.patch(INDIVIDUAL_EVAL_PATH)
    mock_parallel_eval = mocker.patch(PARALLEL_EVAL_PATH)
    mock_cpu_strategy = mocker.patch(CPU_STRAT_PATH)
    mock_gpu_strategy = mocker.patch(GPU_STRAT_PATH)
    mock_hybrid_strategy = mocker.patch(HYBRID_STRAT_PATH)

    config = base_config
    config["parallel_config"]["execution"]["enable_parallel"] = True
    config["parallel_config"]["execution"]["evaluation_mode"] = "HYBRID"
    args = common_args

    evaluator = create_evaluator(config=config, **args)

    mock_hybrid_strategy.assert_called_once_with()
    mock_cpu_strategy.assert_not_called()
    mock_gpu_strategy.assert_not_called()

    mock_parallel_eval.assert_called_once_with(
        config=config,
        training_epochs=args["training_epochs"],
        early_stop_epochs=args["early_stop_epochs"],
        subset_percentage=args["subset_percentage"],
        strategy=mock_hybrid_strategy.return_value,
        progress=mock_tui.progress,
        session_log_filename=args["session_log_filename"],
        train_indices=args["train_indices"],
        test_indices=args["test_indices"],
        fixed_batch_size=args["fixed_batch_size"],
    )
    mock_individual_eval.assert_not_called()
    assert evaluator == mock_parallel_eval.return_value


def test_create_evaluator_parallel_invalid_mode_fallback(
    base_config, common_args, mock_tui, mocker
):
    """
    Tests that IndividualEvaluator is created as a fallback when an
    unknown evaluation_mode is provided and an error is logged.
    """
    mock_individual_eval = mocker.patch(INDIVIDUAL_EVAL_PATH)
    mock_parallel_eval = mocker.patch(PARALLEL_EVAL_PATH)
    mock_logger = mocker.patch(LOGGER_PATH)

    invalid_mode = "INVALID_MODE"
    config = base_config
    config["parallel_config"]["execution"]["enable_parallel"] = True
    config["parallel_config"]["execution"]["evaluation_mode"] = invalid_mode
    nn_config = config["neural_network_config"]
    args = common_args

    evaluator = create_evaluator(config=config, **args)

    mock_logger.error.assert_called_once_with(
        f"Unknown evaluation mode: {invalid_mode}. Defaulting to sequential."
    )

    mock_individual_eval.assert_called_once_with(
        neural_config=nn_config,
        training_epochs=args["training_epochs"],
        early_stop_epochs=args["early_stop_epochs"],
        subset_percentage=args["subset_percentage"],
        progress=mock_tui.progress,
        train_indices=args["train_indices"],
        test_indices=args["test_indices"],
    )
    mock_parallel_eval.assert_not_called()
    assert evaluator == mock_individual_eval.return_value
