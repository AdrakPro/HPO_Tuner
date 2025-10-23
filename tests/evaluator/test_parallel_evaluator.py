import queue
import threading
import time
from unittest.mock import MagicMock

import pytest
import torch.multiprocessing as mp

from src.evaluator.parallel_evaluator import ParallelEvaluator, _task_stealer
from src.evaluator.scheduling_strategy import HybridStrategy, SchedulingStrategy
from src.genetic.stop_conditions import StopConditions
from src.model.parallel import Result


def test_task_stealer_moves_task_when_conditions_met():
    """Verify the stealer moves a task from GPU to CPU queue when appropriate."""
    gpu_queue = MagicMock(spec=mp.Queue, qsize=MagicMock(), get=MagicMock())
    cpu_queue = MagicMock(spec=mp.Queue, put=MagicMock())
    stop_event = threading.Event()
    wakeup_event = threading.Event()
    pending_cpu_tasks = MagicMock(
        spec=threading.Semaphore,
        acquire=MagicMock(return_value=True),
        release=MagicMock(),
    )

    gpu_queue.qsize.return_value = 10  # > threshold
    mock_task = MagicMock()
    gpu_queue.get.return_value = mock_task

    stealer_thread = threading.Thread(
        target=_task_stealer,
        args=(
            gpu_queue,
            cpu_queue,
            2,
            2,
            stop_event,
            wakeup_event,
            pending_cpu_tasks,
        ),
    )
    stealer_thread.start()

    wakeup_event.set()
    time.sleep(0.1)
    stop_event.set()
    wakeup_event.set()
    stealer_thread.join(timeout=1)

    pending_cpu_tasks.acquire.assert_called_with(blocking=False)
    gpu_queue.get.assert_called_once_with(block=False)
    cpu_queue.put.assert_called_once_with(mock_task)


def test_task_stealer_releases_semaphore_if_not_stealing():
    """Verify the semaphore is released if the threshold isn't met."""
    gpu_queue = MagicMock(spec=mp.Queue, qsize=MagicMock(), get=MagicMock())
    cpu_queue = MagicMock(spec=mp.Queue, put=MagicMock())
    stop_event = threading.Event()
    wakeup_event = threading.Event()
    pending_cpu_tasks = MagicMock(
        spec=threading.Semaphore,
        acquire=MagicMock(return_value=True),
        release=MagicMock(),
    )

    gpu_queue.qsize.return_value = 3  # < threshold

    stealer_thread = threading.Thread(
        target=_task_stealer,
        args=(
            gpu_queue,
            cpu_queue,
            2,
            2,
            stop_event,
            wakeup_event,
            pending_cpu_tasks,
        ),
    )
    stealer_thread.start()

    wakeup_event.set()
    time.sleep(0.1)
    stop_event.set()
    wakeup_event.set()
    stealer_thread.join(timeout=1)

    gpu_queue.get.assert_not_called()
    pending_cpu_tasks.release.assert_called_once()


@pytest.fixture
def mock_config():
    """Provides a default configuration dictionary for the evaluator."""
    return {
        "parallel_config": {"execution": {"gpu_workers": 2, "cpu_workers": 2}},
        "neural_network_config": {},
        "early_stop_generations": 10,
        "max_generations": 100,
        "fitness_goal": 0.99,
        "time_limit_minutes": 5,
    }


@pytest.fixture
def mock_deps(mocker):
    """Mocks all major dependencies of ParallelEvaluator."""
    mocker.patch("atexit.register")
    mocker.patch("src.evaluator.parallel_evaluator.signal_manager")
    mock_logger = mocker.patch("src.evaluator.parallel_evaluator.logger")
    mock_thread_cls = mocker.patch(
        "src.evaluator.parallel_evaluator.threading.Thread"
    )

    def create_mock_event(*args, **kwargs):
        event_mock = MagicMock(spec=threading.Event)
        event_mock.set = MagicMock()
        return event_mock

    mock_event_cls = mocker.patch(
        "src.evaluator.parallel_evaluator.threading.Event"
    )
    mock_event_cls.side_effect = create_mock_event

    mock_ctx = MagicMock()

    def create_mock_queue(*args, **kwargs):
        q = MagicMock(spec=mp.Queue)
        q.get = MagicMock()
        q.put = MagicMock()
        q.put_nowait = MagicMock()
        q.qsize = MagicMock(return_value=0)
        q.empty = MagicMock(return_value=True)
        q.close = MagicMock()
        q.join_thread = MagicMock()
        return q

    mock_ctx.Queue.side_effect = create_mock_queue
    mocker.patch("torch.multiprocessing.get_context", return_value=mock_ctx)

    return {"Thread": mock_thread_cls, "mp_context": mock_ctx}


class TestParallelEvaluator:

    def test_evaluate_population_distributes_tasks_hybrid(
        self, mock_config, mock_deps
    ):
        mock_strategy = MagicMock(spec=HybridStrategy)
        mock_gpu_q = mock_deps["mp_context"].Queue()
        mock_cpu_q = mock_deps["mp_context"].Queue()
        mock_strategy.launch_workers.return_value = {
            "workers": [MagicMock()] * 4,
            "gpu_task_queue": mock_gpu_q,
            "cpu_task_queue": mock_cpu_q,
        }

        evaluator = ParallelEvaluator(
            config=mock_config,
            strategy=mock_strategy,
            progress=MagicMock(),
            training_epochs=1,
            early_stop_epochs=1,
            subset_percentage=1.0,
            session_log_filename="test.log",
            train_indices=None,
            test_indices=None,
        )

        results_to_return = [
            Result(
                index=i,
                fitness=0.5,
                loss=0.5,
                duration_seconds=1.0,
                status="SUCCESS",
                log_lines=[],
            )
            for i in range(5)
        ]
        evaluator.result_queue.get.side_effect = results_to_return + [
            queue.Empty
        ]
        evaluator._is_any_worker_alive = lambda: True

        population = [{"id": i} for i in range(5)]
        evaluator.evaluate_population(population, stop_conditions=None)

        assert mock_cpu_q.put.call_count == 2
        assert mock_gpu_q.put.call_count == 3
        assert evaluator.result_queue.get.call_count >= 5

    def test_evaluate_population_stops_early_on_fitness_goal(
        self, mock_config, mock_deps
    ):
        mock_strategy = MagicMock(spec=SchedulingStrategy)
        mock_task_q = mock_deps["mp_context"].Queue()
        mock_strategy.launch_workers.return_value = {
            "workers": [MagicMock()],
            "task_queues": [mock_task_q],
        }

        evaluator = ParallelEvaluator(
            config=mock_config,
            strategy=mock_strategy,
            progress=MagicMock(),
            training_epochs=1,
            early_stop_epochs=1,
            subset_percentage=1.0,
            session_log_filename="test.log",
            train_indices=None,
            test_indices=None,
        )

        evaluator._clear_pending_tasks = MagicMock()
        evaluator._fill_missing_results = MagicMock()

        stop_conditions = StopConditions(config=mock_config)
        stop_conditions.fitness_goal = 0.9
        population = [{"id": i} for i in range(4)]

        successful_result = Result(
            index=0,
            fitness=0.95,
            loss=0.1,
            duration_seconds=1,
            status="SUCCESS",
            log_lines=[],
        )
        evaluator.result_queue.get.side_effect = [
            successful_result,
            queue.Empty,
        ]
        evaluator._is_any_worker_alive = lambda: True

        evaluator.evaluate_population(population, stop_conditions)

        evaluator._clear_pending_tasks.assert_called_once()
        evaluator._fill_missing_results.assert_called_once()

    def test_cleanup_workers_terminates_everything(
        self, mock_config, mock_deps
    ):
        mock_strategy = MagicMock(spec=HybridStrategy)
        mock_worker = MagicMock(spec=mp.Process)
        mock_worker.is_alive.return_value = True

        mock_gpu_q = mock_deps["mp_context"].Queue()
        mock_cpu_q = mock_deps["mp_context"].Queue()
        mock_strategy.launch_workers.return_value = {
            "workers": [mock_worker],
            "gpu_task_queue": mock_gpu_q,
            "cpu_task_queue": mock_cpu_q,
        }

        evaluator = ParallelEvaluator(
            config=mock_config,
            strategy=mock_strategy,
            progress=MagicMock(),
            training_epochs=1,
            early_stop_epochs=1,
            subset_percentage=1.0,
            session_log_filename="test.log",
            train_indices=None,
            test_indices=None,
        )

        evaluator._terminate_process = MagicMock()
        evaluator.cleanup_workers()

        evaluator._stop_stealer_event.set.assert_called_once()
        evaluator._stealer_wakeup_event.set.assert_called_once()
        evaluator._stealer_thread.join.assert_called_once()
        mock_worker.join.assert_called_once()
        evaluator._terminate_process.assert_called_once_with(mock_worker)
