import dataclasses
import os
import queue
from unittest.mock import MagicMock

import pytest
import torch
from pytest_mock import MockerFixture

from src.evaluator.worker import (
    _is_adamw,
    _is_sgd,
    init_device,
    worker_main,
)
from src.model.chromosome import Chromosome, OptimizerSchedule

# Import the real classes you provided
from src.model.parallel import Result, Task, WorkerConfig
from src.utils.exceptions import CudaOutOfMemoryError


@pytest.fixture
def base_worker_config() -> WorkerConfig:
    """Provides a basic WorkerConfig with real queues."""
    ctx = torch.multiprocessing.get_context("spawn")

    return WorkerConfig(
        worker_id=0,
        device="cpu",
        task_queue=ctx.Queue(),
        result_queue=ctx.Queue(),
        session_log_filename="test_log.log",
        num_dataloader_workers=0,
        fixed_batch_size=None,
    )


@pytest.fixture
def mock_chromosome() -> MagicMock:
    """
    Provides a mock Chromosome object that can be modified
    and inspected.
    """
    chrom = MagicMock(spec=Chromosome)
    chrom.base_lr = 0.01
    chrom.batch_size = 128
    chrom.optimizer_schedule = OptimizerSchedule.ADAMW_COSINE
    chrom.aug_intensity = 0.5
    chrom.weight_decay = 1e-4
    return chrom


@pytest.fixture
def mock_task_data(mock_chromosome: MagicMock) -> dict:
    """Provides a dict of mock task data."""
    hyperparams_dict = {"lr": 0.01, "bs": 128}

    return {
        "index": 1,
        "pop_size": 10,
        "training_epochs": 2,
        "early_stop_epochs": 1,
        "individual_hyperparams": hyperparams_dict,
        "neural_network_config": {"type": "mock_net"},
        "subset_percentage": 1.0,
        "train_indices": None,
        "test_indices": None,
        "is_final": False,
    }


@pytest.fixture
def mock_dependencies(
    mocker: MockerFixture, mock_chromosome: MagicMock
) -> MagicMock:
    """
    Mocks all heavy or external dependencies for a standard worker run.
    Returns the mock for train_and_eval for inspection.
    """
    mocker.patch(
        "src.evaluator.worker.Chromosome.from_dict",
        return_value=mock_chromosome,
    )

    mock_cm = MagicMock()
    mock_cm.__enter__.return_value = (
        ["dummy_train_loader"],
        ["dummy_test_loader"],
    )
    mocker.patch(
        "src.evaluator.worker.get_dataset_loaders", return_value=mock_cm
    )

    mock_train = mocker.patch(
        "src.evaluator.worker.train_and_eval",
        return_value=(0.95, 0.1),
        # (accuracy, loss)
    )

    mocker.patch("src.evaluator.worker.logger")
    mocker.patch("src.evaluator.worker.ThreadOptimizer.enable_tf32")
    mocker.patch("torch.set_num_threads")
    mocker.patch("torch.cuda.is_available", return_value=True)
    mocker.patch("torch.cuda.set_device")
    mocker.patch("torch.cuda.max_memory_allocated", return_value=1024**3)
    mocker.patch("torch.cuda.reset_peak_memory_stats")

    mocker.patch("time.perf_counter", side_effect=[1000.0, 1005.0])

    return mock_train


def test_init_device_cpu(mocker: MockerFixture):
    """Tests device initialization for 'cpu'."""
    mock_enable_tf32 = mocker.patch(
        "src.evaluator.worker.ThreadOptimizer.enable_tf32"
    )

    device = init_device("cpu")

    assert device == torch.device("cpu")
    mock_enable_tf32.assert_called_once()


def test_init_device_gpu_available(mocker: MockerFixture):
    """Tests device initialization for a GPU when CUDA is available."""
    mocker.patch("torch.cuda.is_available", return_value=True)
    mock_set_device = mocker.patch("torch.cuda.set_device")
    mocker.patch("src.evaluator.worker.ThreadOptimizer.enable_tf32")

    device = init_device(0)

    assert device == torch.device("cuda:0")
    mock_set_device.assert_called_once_with(0)


def test_init_device_gpu_not_available(mocker: MockerFixture):
    """Tests device initialization for a GPU when CUDA is NOT available."""
    mocker.patch("torch.cuda.is_available", return_value=False)
    mock_logger_error = mocker.patch("src.evaluator.worker.logger.error")
    mocker.patch("src.evaluator.worker.ThreadOptimizer.enable_tf32")

    device = init_device(0)

    assert device == torch.device("cpu")
    mock_logger_error.assert_called_once_with(
        "CUDA not available for GPU worker 0."
    )


@pytest.mark.parametrize(
    "optimizer, expected_adamw, expected_sgd",
    [
        (OptimizerSchedule.ADAMW_COSINE, True, False),
        (OptimizerSchedule.ADAMW_ONECYCLE, True, False),
        (OptimizerSchedule.ADAMW_EXPONENTIAL, True, False),
        (OptimizerSchedule.SGD_COSINE, False, True),
        (OptimizerSchedule.SGD_ONECYCLE, False, True),
        (OptimizerSchedule.SGD_EXPONENTIAL, False, True),
    ],
)
def test_optimizer_checkers(optimizer, expected_adamw, expected_sgd):
    """Tests the _is_adamw and _is_sgd helper functions."""
    assert _is_adamw(optimizer) == expected_adamw
    assert _is_sgd(optimizer) == expected_sgd


def test_worker_main_success_cpu(
    base_worker_config: WorkerConfig,
    mock_task_data: dict,
    mock_dependencies: MagicMock,
    mocker: MockerFixture,
):
    """Tests a successful run of the worker on a CPU device."""
    mocker.patch("torch.cuda.is_available", return_value=False)
    mock_set_num_threads = mocker.patch("torch.set_num_threads")

    config = dataclasses.replace(base_worker_config, device="cpu")
    task = Task(**mock_task_data)

    config.task_queue.put(task)
    config.task_queue.put(None)

    worker_main(config)

    assert os.environ["OMP_NUM_THREADS"] == "12"
    assert os.environ["MKL_NUM_THREADS"] == "12"
    mock_set_num_threads.assert_called_with(12)

    result = config.result_queue.get(timeout=1)
    assert isinstance(result, Result)
    assert result.status == "SUCCESS"
    assert result.index == task.index
    assert result.fitness == 0.95
    assert result.loss == 0.1
    assert result.duration_seconds == pytest.approx(5.0)
    assert result.worker_type == "cpu"

    assert any("Accuracy: 0.9500" in str(line) for line in result.log_lines)


def test_worker_main_success_gpu(
    base_worker_config: WorkerConfig,
    mock_task_data: dict,
    mock_dependencies: MagicMock,
    mocker: MockerFixture,
):
    """Tests a successful run of the worker on a (mocked) GPU device."""
    mock_set_num_threads = mocker.patch("torch.set_num_threads")
    mocker.patch("torch.cuda.max_memory_allocated", return_value=1024**3 * 2.5)

    config = dataclasses.replace(base_worker_config, device=0)
    task = Task(**mock_task_data)

    config.task_queue.put(task)
    config.task_queue.put(None)

    worker_main(config)

    assert os.environ["OMP_NUM_THREADS"] == "2"
    assert os.environ["MKL_NUM_THREADS"] == "2"
    mock_set_num_threads.assert_called_with(2)

    result = config.result_queue.get(timeout=1)
    assert result.status == "SUCCESS"
    assert result.worker_type == "gpu"

    assert any(
        "GPU memory used: 2.50 GB" in str(line) for line in result.log_lines
    )


def test_worker_main_controlled_failure(
    base_worker_config: WorkerConfig,
    mock_task_data: dict,
    mock_dependencies: MagicMock,
    mocker: MockerFixture,
):
    """Tests that a CudaOutOfMemoryError is caught and handled."""
    mocker.patch(
        "src.evaluator.worker.train_and_eval",
        side_effect=CudaOutOfMemoryError("Test OOM"),
    )

    task = Task(**mock_task_data)
    base_worker_config.task_queue.put(task)
    base_worker_config.task_queue.put(None)

    worker_main(base_worker_config)

    result = base_worker_config.result_queue.get(timeout=1)
    assert result.status == "FAILURE"
    assert result.fitness == 0.0
    assert result.loss == float("inf")
    assert result.error_message == "Test OOM"
    assert result.worker_type == "cpu"

    assert any("Controlled failure" in str(line) for line in result.log_lines)


def test_worker_main_unexpected_failure(
    base_worker_config: WorkerConfig,
    mock_task_data: dict,
    mock_dependencies: MagicMock,
    mocker: MockerFixture,
):
    """Tests that a generic Exception is caught and handled."""
    mocker.patch(
        "src.evaluator.worker.train_and_eval",
        side_effect=ValueError("Unexpected error"),
    )

    task = Task(**mock_task_data)
    base_worker_config.task_queue.put(task)
    base_worker_config.task_queue.put(None)

    worker_main(base_worker_config)

    result = base_worker_config.result_queue.get(timeout=1)
    assert result.status == "FAILURE"
    assert result.error_message == "Unexpected error"
    assert result.worker_type == "cpu"

    assert any("Unexpected error" in str(line) for line in result.log_lines)


def test_worker_main_handles_empty_queue(
    base_worker_config: WorkerConfig, mocker: MockerFixture
):
    """Tests that the worker loops on queue.Empty and stops on None."""
    mock_get = mocker.patch.object(
        base_worker_config.task_queue,
        "get",
        side_effect=[queue.Empty, queue.Empty, None],
    )

    mocker.patch(
        "src.evaluator.worker.init_device", return_value=torch.device("cpu")
    )
    mocker.patch("src.evaluator.worker.ThreadOptimizer.enable_tf32")
    mocker.patch("torch.set_num_threads")

    worker_main(base_worker_config)

    assert mock_get.call_count == 3
    assert base_worker_config.result_queue.empty()


@pytest.mark.parametrize(
    "optimizer, batch_size, base_lr, expected_lr, is_capped",
    [
        (OptimizerSchedule.ADAMW_COSINE, 128, 0.001, 0.001, False),
        (
            OptimizerSchedule.ADAMW_COSINE,
            512,
            0.001,
            0.001 * (512 / 128) ** 0.5,
            False,
        ),
        (OptimizerSchedule.ADAMW_COSINE, 2048, 0.005, 0.01, True),
        (OptimizerSchedule.SGD_COSINE, 128, 0.1, 0.1, False),
        (OptimizerSchedule.SGD_COSINE, 256, 0.1, 0.1 * (256 / 128), False),
        (OptimizerSchedule.SGD_COSINE, 1024, 0.1, 0.5, True),
    ],
)
def test_worker_main_lr_scaling(
    mocker: MockerFixture,
    base_worker_config: WorkerConfig,
    mock_task_data: dict,
    optimizer: OptimizerSchedule,
    batch_size: int,
    base_lr: float,
    expected_lr: float,
    is_capped: bool,
):
    """Tests the learning rate scaling logic inside the worker."""

    mock_chrom = MagicMock(spec=Chromosome)
    mock_chrom.base_lr = base_lr
    mock_chrom.batch_size = batch_size
    mock_chrom.optimizer_schedule = optimizer
    mock_chrom.aug_intensity = 0.5
    mock_chrom.weight_decay = 1e-4

    mocker.patch(
        "src.evaluator.worker.Chromosome.from_dict", return_value=mock_chrom
    )

    mock_train = mocker.patch(
        "src.evaluator.worker.train_and_eval", return_value=(0.9, 0.1)
    )
    mock_cm = MagicMock()
    mock_cm.__enter__.return_value = (["train"], ["test"])
    mocker.patch(
        "src.evaluator.worker.get_dataset_loaders", return_value=mock_cm
    )
    mocker.patch("src.evaluator.worker.logger")
    mocker.patch("src.evaluator.worker.ThreadOptimizer.enable_tf32")
    mocker.patch("torch.set_num_threads")
    mocker.patch("torch.cuda.is_available", return_value=False)
    mocker.patch("time.perf_counter", side_effect=[1000.0, 1001.0])

    task = Task(**mock_task_data)
    base_worker_config.task_queue.put(task)
    base_worker_config.task_queue.put(None)

    worker_main(base_worker_config)

    result = base_worker_config.result_queue.get(timeout=1)
    assert result.status == "SUCCESS"

    mock_train.assert_called_once()
    passed_chromosome = mock_train.call_args.kwargs["chromosome"]

    assert passed_chromosome == mock_chrom
    assert passed_chromosome.base_lr == pytest.approx(expected_lr)

    if batch_size == 128:
        assert any("Using base_lr" in str(line) for line in result.log_lines)
    elif is_capped:
        assert any("CAPPED" in str(line) for line in result.log_lines)
    else:
        assert any("Scaled base_lr" in str(line) for line in result.log_lines)
