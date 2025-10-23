from unittest.mock import MagicMock, patch

import pytest

from src.evaluator.scheduling_strategy import (
    CPUOnlyStrategy,
    GPUOnlyStrategy,
    HybridStrategy,
    _spawn_processes,
)
from src.model.parallel import WorkerConfig


@pytest.fixture
def mock_mp_context():
    """Provides a mock multiprocessing context."""
    context = MagicMock()
    context.Process = MagicMock()
    context.Queue.side_effect = lambda *args, **kwargs: MagicMock()
    return context


@pytest.fixture
def base_kwargs(mock_mp_context):
    """Provides a base dictionary of keyword arguments for launch_workers."""
    return {
        "ctx": mock_mp_context,
        "task_queue": mock_mp_context.Queue(),
        "result_queue": mock_mp_context.Queue(),
        "session_log_filename": "test_session.log",
        "fixed_batch_size": 32,
    }


class TestSpawnProcesses:
    """
    Tests the core _spawn_processes helper function to ensure it creates
    processes with the correct WorkerConfig.
    """

    def test_spawns_gpu_workers_correctly(self, mock_mp_context):
        """Verify GPU worker configurations are correct."""
        _spawn_processes(
            ctx=mock_mp_context,
            task_queue=MagicMock(),
            result_queue=MagicMock(),
            session_log_filename="test.log",
            num_gpu_workers=2,
            dl_workers_per_gpu=4,
            fixed_batch_size=16,
        )

        assert mock_mp_context.Process.call_count == 2

        args_first_call = mock_mp_context.Process.call_args_list[0].kwargs[
            "args"
        ]
        config_1: WorkerConfig = args_first_call[0]
        assert config_1.worker_id == 0
        assert config_1.device == 0
        assert config_1.num_dataloader_workers == 4
        assert config_1.fixed_batch_size == 16
        assert config_1.total_cpu_workers is None

        args_second_call = mock_mp_context.Process.call_args_list[1].kwargs[
            "args"
        ]
        config_2: WorkerConfig = args_second_call[0]
        assert config_2.worker_id == 1
        assert config_2.device == 1

    def test_spawns_cpu_workers_correctly(self, mock_mp_context):
        """Verify CPU worker configurations are correct."""
        _spawn_processes(
            ctx=mock_mp_context,
            task_queue=MagicMock(),
            result_queue=MagicMock(),
            session_log_filename="test.log",
            num_cpu_workers=2,
            dl_workers_per_cpu=1,
        )

        assert mock_mp_context.Process.call_count == 2

        config_1: WorkerConfig = mock_mp_context.Process.call_args_list[
            0
        ].kwargs["args"][0]
        assert config_1.worker_id == 0
        assert config_1.device == "cpu"
        assert config_1.num_dataloader_workers == 1
        assert config_1.total_cpu_workers == 2

    def test_spawns_hybrid_workers_with_offset(self, mock_mp_context):
        """Verify worker IDs are correctly offset for hybrid configurations."""
        _spawn_processes(
            ctx=mock_mp_context,
            task_queue=MagicMock(),
            result_queue=MagicMock(),
            session_log_filename="test.log",
            num_cpu_workers=1,
            num_gpu_workers=2,
            gpu_worker_offset=2,
        )

        assert mock_mp_context.Process.call_count == 3

        gpu_config_1: WorkerConfig = mock_mp_context.Process.call_args_list[
            0
        ].kwargs["args"][0]
        assert gpu_config_1.worker_id == 2  # 0 + offset

        cpu_config_1: WorkerConfig = mock_mp_context.Process.call_args_list[
            2
        ].kwargs["args"][0]
        assert (
            cpu_config_1.worker_id == 4
        )  # 0 + num_gpu_workers(2) + gpu_worker_offset(2)


@patch("src.evaluator.scheduling_strategy._spawn_processes")
@patch("src.evaluator.scheduling_strategy.logger")
class TestSchedulingStrategies:
    """
    Tests the logic of each strategy class, mocking the actual process spawner.
    """

    def test_cpu_only_strategy(self, mock_logger, mock_spawn, base_kwargs):
        """Verify CPUOnlyStrategy calls spawner with correct CPU parameters."""
        strategy = CPUOnlyStrategy()
        exec_config = {"cpu_workers": 4, "dataloader_workers": {"per_cpu": 2}}

        strategy.launch_workers(execution_config=exec_config, **base_kwargs)

        mock_logger.info.assert_called_with(
            "Using CPU-Only scheduling strategy."
        )
        mock_spawn.assert_called_once_with(
            ctx=base_kwargs["ctx"],
            task_queue=base_kwargs["task_queue"],
            result_queue=base_kwargs["result_queue"],
            num_cpu_workers=4,
            session_log_filename=base_kwargs["session_log_filename"],
            dl_workers_per_cpu=2,
            fixed_batch_size=base_kwargs["fixed_batch_size"],
        )

    @patch("torch.cuda.device_count", return_value=4)
    def test_gpu_only_strategy_normal(
        self, mock_device_count, mock_logger, mock_spawn, base_kwargs
    ):
        """Verify GPUOnlyStrategy calls spawner with correct GPU parameters."""
        strategy = GPUOnlyStrategy()
        exec_config = {"gpu_workers": 2, "dataloader_workers": {"per_gpu": 8}}

        result = strategy.launch_workers(
            execution_config=exec_config, **base_kwargs
        )

        mock_logger.info.assert_called_with(
            "Using GPU-Only scheduling strategy."
        )
        mock_spawn.assert_called_once()
        spawn_kwargs = mock_spawn.call_args.kwargs
        assert spawn_kwargs["num_gpu_workers"] == 2
        assert spawn_kwargs["dl_workers_per_gpu"] == 8
        assert "workers" in result
        assert "task_queues" in result and len(result["task_queues"]) == 1

    @patch("torch.cuda.device_count", return_value=1)
    def test_gpu_only_strategy_adjusts_down(
        self, mock_device_count, mock_logger, mock_spawn, base_kwargs
    ):
        """Verify GPUOnlyStrategy adjusts worker count if too many are requested."""
        strategy = GPUOnlyStrategy()
        exec_config = {"gpu_workers": 4}

        strategy.launch_workers(execution_config=exec_config, **base_kwargs)

        mock_logger.warning.assert_called_with(
            "Requested 4 GPUs, but only 1 are available. Adjusting."
        )
        assert mock_spawn.call_args.kwargs["num_gpu_workers"] == 1

    @patch("torch.cuda.device_count", return_value=0)
    def test_gpu_only_strategy_no_gpus(
        self, mock_device_count, mock_logger, mock_spawn, base_kwargs
    ):
        """Verify GPUOnlyStrategy returns empty list and logs error if no GPUs are available."""
        strategy = GPUOnlyStrategy()
        exec_config = {"gpu_workers": 1}

        result = strategy.launch_workers(
            execution_config=exec_config, **base_kwargs
        )

        mock_logger.error.assert_called_with(
            "GPU-Only Strategy selected, but no GPU workers are configured or available."
        )
        mock_spawn.assert_not_called()
        assert result == {"workers": [], "task_queues": []}

    @patch("torch.cuda.device_count", return_value=2)
    def test_hybrid_strategy(
        self, mock_device_count, mock_logger, mock_spawn, base_kwargs
    ):
        """Verify HybridStrategy calls spawner twice with correct, distinct parameters."""
        mock_spawn.side_effect = [[MagicMock()], [MagicMock(), MagicMock()]]

        strategy = HybridStrategy()
        exec_config = {
            "gpu_workers": 1,
            "cpu_workers": 2,
            "dataloader_workers": {"per_gpu": 8, "per_cpu": 2},
        }

        result = strategy.launch_workers(
            execution_config=exec_config, **base_kwargs
        )

        mock_logger.info.assert_called_with(
            "Using main HYBRID strategy with Task Stealing."
        )
        assert mock_spawn.call_count == 2

        # Call 1: GPU workers
        gpu_call_kwargs = mock_spawn.call_args_list[0].kwargs
        assert gpu_call_kwargs["num_gpu_workers"] == 1
        assert "num_cpu_workers" not in gpu_call_kwargs
        assert gpu_call_kwargs["dl_workers_per_gpu"] == 8

        # Call 2: CPU workers
        cpu_call_kwargs = mock_spawn.call_args_list[1].kwargs
        assert cpu_call_kwargs["num_cpu_workers"] == 2
        assert "num_gpu_workers" not in cpu_call_kwargs
        assert cpu_call_kwargs["dl_workers_per_cpu"] == 2
        assert cpu_call_kwargs["gpu_worker_offset"] == 1

        # Verify final result
        assert len(result["workers"]) == 3  # 1 from first call, 2 from second
        assert "gpu_task_queue" in result and "cpu_task_queue" in result
        assert result["gpu_task_queue"] is not result["cpu_task_queue"]
