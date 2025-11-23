from unittest.mock import MagicMock, patch

import pytest

from src.evaluator.scheduling_strategy import (
    CPUOnlyStrategy,
    GPUOnlyStrategy,
    HybridStrategy,
    NumaCoreAllocator,
)


@pytest.fixture
def mock_ctx():
    ctx = MagicMock()
    ctx.Queue.return_value = MagicMock()
    return ctx


@pytest.fixture
def exec_config():
    return {
        "cpu_workers": 2,
        "gpu_workers": 1,
        "dataloader_workers": {"per_cpu": 1, "per_gpu": 2},
    }


class TestStrategies:

    @patch("src.evaluator.scheduling_strategy._spawn_processes")
    def test_cpu_strategy(self, mock_spawn, mock_ctx, exec_config):
        strategy = CPUOnlyStrategy()
        mock_spawn.return_value = [MagicMock(), MagicMock()]

        result = strategy.launch_workers(
            ctx=mock_ctx,
            result_queue=MagicMock(),
            execution_config=exec_config,
            session_log_filename="test.log",
        )

        assert len(result["workers"]) == 2
        mock_spawn.assert_called_once()
        _, kwargs = mock_spawn.call_args
        assert kwargs["num_cpu_workers"] == 2

    @patch("src.evaluator.scheduling_strategy._spawn_processes")
    @patch("torch.cuda.device_count", return_value=4)
    def test_gpu_strategy(self, mock_cuda, mock_spawn, mock_ctx, exec_config):
        strategy = GPUOnlyStrategy()
        mock_spawn.return_value = [MagicMock()]

        result = strategy.launch_workers(
            ctx=mock_ctx,
            result_queue=MagicMock(),
            execution_config=exec_config,
            session_log_filename="test.log",
        )

        assert len(result["workers"]) == 1
        _, kwargs = mock_spawn.call_args
        assert kwargs["num_gpu_workers"] == 1

    @patch("src.evaluator.scheduling_strategy._spawn_processes")
    @patch("torch.cuda.device_count", return_value=4)
    def test_hybrid_strategy(
        self, mock_cuda, mock_spawn, mock_ctx, exec_config
    ):
        strategy = HybridStrategy()
        mock_spawn.side_effect = [[MagicMock()], [MagicMock(), MagicMock()]]

        result = strategy.launch_workers(
            ctx=mock_ctx,
            result_queue=MagicMock(),
            execution_config=exec_config,
            session_log_filename="test.log",
        )

        assert len(result["workers"]) == 3
        assert "gpu_task_queue" in result
        assert "cpu_task_queue" in result


class TestNumaAllocator:

    @patch("src.evaluator.scheduling_strategy.ENABLE_NUMA_SUPPORT", True)
    def test_allocation_logic(self):
        allocator = NumaCoreAllocator(total_cores=8, cores_per_node=4)

        cores = allocator.allocate(0, 2, from_end=False)
        assert cores == [0, 1]

        cores = allocator.allocate(0, 2, from_end=True)
        assert cores == [2, 3]

        cores = allocator.allocate(0, 1)
        assert cores is None
