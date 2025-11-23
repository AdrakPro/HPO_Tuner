import signal
from unittest.mock import MagicMock, patch

import pytest

from src.evaluator.worker import worker_main, pin_worker_to_cores, init_device
from src.model.parallel import Result, WorkerConfig, Task


@pytest.fixture
def mock_queues():
    """Returns mock task and result queues."""
    return MagicMock(), MagicMock()


@pytest.fixture
def worker_config(mock_queues):
    """Creates a standard worker configuration."""
    task_q, result_q = mock_queues
    return WorkerConfig(
        worker_id=1,
        device="cpu",
        task_queue=task_q,
        result_queue=result_q,
        session_log_filename="test.log",
        num_dataloader_workers=0,
        core_ids=[0, 1],
    )


@pytest.fixture
def sample_task():
    """Creates a sample Task object."""
    return Task(
        index=0,
        neural_network_config={},
        individual_hyperparams={
            "base_lr": 0.01,
            "batch_size": 32,
            "optimizer_schedule": "SGD_COSINE",
        },
        training_epochs=1,
        early_stop_epochs=1,
        subset_percentage=1.0,
        pop_size=10,
        is_final=False,
        train_indices=None,
        test_indices=None,
    )


class TestWorkerHelpers:
    def test_pin_worker_to_cores(self):
        """Test core pinning logic."""
        with patch("os.sched_setaffinity", create=True) as mock_affinity:
            pin_worker_to_cores([0, 1])
            mock_affinity.assert_called_once_with(0, [0, 1])

    @patch("src.evaluator.worker.ThreadOptimizer")
    def test_init_device_cpu(self, mock_thread_opt):
        """Test CPU device initialization."""
        dev = init_device("cpu")
        assert dev.type == "cpu"

    @patch("src.evaluator.worker.ThreadOptimizer")
    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.set_device")
    def test_init_device_gpu(
        self, mock_set_device, mock_is_available, mock_thread_opt
    ):
        """Test GPU device initialization."""
        dev = init_device(0)
        assert dev.type == "cuda"
        assert dev.index == 0
        mock_set_device.assert_called_with(0)


class TestWorkerMain:

    @pytest.fixture(autouse=True)
    def mock_sys_calls(self):
        """Automatically mocks threading calls for all tests in this class."""
        with (
            patch("src.evaluator.worker.set_num_interop_threads"),
            patch("src.evaluator.worker.set_num_threads"),
            patch("src.evaluator.worker.pin_worker_to_cores"),
        ):
            yield

    @patch("src.evaluator.worker.train_and_eval")
    @patch("src.evaluator.worker.get_dataset_loaders")
    @patch("src.evaluator.worker.Chromosome.from_dict")
    @patch("src.evaluator.worker.ThreadOptimizer")
    def test_worker_process_success(
        self,
        mock_thread_opt,
        mock_chrom_cls,
        mock_get_loaders,
        mock_train,
        worker_config,
        sample_task,
    ):
        """Verify successful task processing workflow."""
        worker_config.task_queue.get.side_effect = [
            sample_task,
            None,
        ]  # Task then sentinel

        mock_train.return_value = (0.9, 0.1)  # Acc, Loss

        mock_chrom = MagicMock()
        mock_chrom.base_lr = 0.01
        mock_chrom.batch_size = 32
        mock_chrom.aug_intensity = "NONE"
        mock_chrom.optimizer_schedule = "SGD_COSINE"
        mock_chrom_cls.return_value = mock_chrom

        mock_loader_context = MagicMock()
        mock_get_loaders.return_value = mock_loader_context
        mock_loader_context.__enter__.return_value = (MagicMock(), MagicMock())

        worker_main(worker_config)

        assert worker_config.result_queue.put.called
        result = worker_config.result_queue.put.call_args[0][0]

        assert isinstance(result, Result)
        assert result.status == "SUCCESS"
        assert result.fitness == 0.9
        assert result.worker_type == "cpu"

    @patch("src.evaluator.worker.train_and_eval")
    @patch("src.evaluator.worker.get_dataset_loaders")
    @patch("src.evaluator.worker.Chromosome.from_dict")
    @patch("src.evaluator.worker.ThreadOptimizer")
    def test_worker_handles_exception(
        self,
        mock_thread_opt,
        mock_chrom,
        mock_loaders,
        mock_train,
        worker_config,
        sample_task,
    ):
        """Verify exception handling and failure reporting."""
        worker_config.task_queue.get.side_effect = [sample_task, None]

        mock_instance = MagicMock()
        mock_instance.base_lr = 0.01
        mock_instance.batch_size = 32
        mock_instance.optimizer_schedule = "SGD_COSINE"
        mock_chrom.return_value = mock_instance

        mock_loaders.return_value.__enter__.return_value = (None, None)
        mock_train.side_effect = RuntimeError("GPU OOM")

        worker_main(worker_config)

        result = worker_config.result_queue.put.call_args[0][0]
        assert result.status == "FAILURE"
        assert "GPU OOM" in result.error_message

    @patch("src.evaluator.worker.ThreadOptimizer")
    def test_worker_stops_on_sentinel(self, mock_thread_opt, worker_config):
        """Verify worker exits loop when receiving None."""
        worker_config.task_queue.get.return_value = None

        worker_main(worker_config)

        worker_config.result_queue.put.assert_not_called()

    @patch("signal.signal")
    @patch("src.evaluator.worker.ThreadOptimizer")
    def test_signal_handling(self, mock_thread_opt, mock_signal, worker_config):
        """Verify signals are ignored then restored."""
        worker_config.task_queue.get.return_value = None

        worker_main(worker_config)

        assert mock_signal.call_count >= 2

        args_first = mock_signal.call_args_list[0]
        assert args_first[0][0] == signal.SIGINT
        assert args_first[0][1] == signal.SIG_IGN
