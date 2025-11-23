import queue
from unittest.mock import MagicMock

import pytest

from src.evaluator.parallel_evaluator import ParallelEvaluator
from src.evaluator.scheduling_strategy import HybridStrategy, SchedulingStrategy
from src.genetic.stop_conditions import StopConditions
from src.model.parallel import Result


@pytest.fixture
def mock_mp_context(mocker):
    """Mocks multiprocessing context and Queue."""
    ctx = MagicMock()
    mock_queue = MagicMock()
    mock_queue.qsize.return_value = 0
    ctx.Queue.return_value = mock_queue

    mocker.patch("torch.multiprocessing.get_context", return_value=ctx)
    return ctx


@pytest.fixture
def mock_strategy():
    """Mocks the scheduling strategy."""
    strategy = MagicMock(spec=SchedulingStrategy)
    strategy.launch_workers.return_value = {
        "workers": [MagicMock(), MagicMock()],
        "task_queues": [MagicMock()],
    }
    return strategy


@pytest.fixture
def mock_hybrid_strategy():
    """Mocks a HybridStrategy specifically."""
    strategy = MagicMock(spec=HybridStrategy)
    gpu_q = MagicMock()
    cpu_q = MagicMock()
    gpu_q.qsize.return_value = 0
    cpu_q.qsize.return_value = 0

    strategy.launch_workers.return_value = {
        "workers": [MagicMock(), MagicMock()],
        "gpu_task_queue": gpu_q,
        "cpu_task_queue": cpu_q,
    }
    return strategy


@pytest.fixture
def parallel_config():
    return {
        "parallel_config": {"execution": {"gpu_workers": 1, "cpu_workers": 1}},
        "neural_network_config": {},
    }


class TestParallelEvaluatorInit:

    def test_init_basic_strategy(
        self, mock_mp_context, mock_strategy, parallel_config
    ):
        """Verify initialization with a standard strategy."""
        evaluator = ParallelEvaluator(
            config=parallel_config,
            training_epochs=5,
            early_stop_epochs=2,
            subset_percentage=1.0,
            strategy=mock_strategy,
            progress=MagicMock(),
            session_log_filename="test.log",
            train_indices=None,
            test_indices=None,
        )

        assert len(evaluator._workers) == 2
        assert (
            evaluator._stealer_thread is None
        )  # Standard strategy has no stealer
        mock_strategy.launch_workers.assert_called_once()

        evaluator.cleanup_workers()

    def test_init_hybrid_strategy(
        self, mock_mp_context, mock_hybrid_strategy, parallel_config
    ):
        """Verify initialization with HybridStrategy starts stealer."""
        evaluator = ParallelEvaluator(
            config=parallel_config,
            training_epochs=5,
            early_stop_epochs=2,
            subset_percentage=1.0,
            strategy=mock_hybrid_strategy,
            progress=MagicMock(),
            session_log_filename="test.log",
            train_indices=None,
            test_indices=None,
        )

        assert evaluator._stealer_thread is not None
        assert evaluator._stealer_thread.is_alive()
        assert evaluator.gpu_task_queue is not None
        assert evaluator.cpu_task_queue is not None

        evaluator.cleanup_workers()

    def test_init_raises_if_no_workers(
        self, mock_mp_context, mock_strategy, parallel_config
    ):
        """Should raise RuntimeError if strategy returns no workers."""
        mock_strategy.launch_workers.return_value = {
            "workers": [],
            "task_queues": [],
        }

        with pytest.raises(RuntimeError, match="failed to launch any workers"):
            ParallelEvaluator(
                config=parallel_config,
                training_epochs=5,
                early_stop_epochs=2,
                subset_percentage=1.0,
                strategy=mock_strategy,
                progress=MagicMock(),
                session_log_filename="test.log",
                train_indices=None,
                test_indices=None,
            )


class TestEvaluatePopulation:

    def test_evaluate_populates_tasks(
        self, mock_mp_context, mock_strategy, parallel_config
    ):
        """Verify evaluate_population puts tasks into queue."""
        evaluator = ParallelEvaluator(
            config=parallel_config,
            training_epochs=1,
            early_stop_epochs=1,
            subset_percentage=1.0,
            strategy=mock_strategy,
            progress=MagicMock(),
            session_log_filename="test.log",
            train_indices=None,
            test_indices=None,
        )

        task_queue = mock_strategy.launch_workers.return_value["task_queues"][0]

        res1 = Result(0, 0.5, 0.1, 1.0, "SUCCESS", [], "cpu")
        res2 = Result(1, 0.6, 0.1, 1.0, "SUCCESS", [], "cpu")

        evaluator.result_queue.get.side_effect = [res1, res2]

        population = [{"lr": 0.1}, {"lr": 0.2}]
        results = evaluator.evaluate_population(
            population, stop_conditions=None
        )

        assert task_queue.put.call_count == 2
        assert len(results) == 2
        assert results[0].fitness == 0.5

        evaluator.cleanup_workers()

    def test_early_exit_on_fitness_goal(
        self, mock_mp_context, mock_strategy, parallel_config
    ):
        """Verify evaluation stops early if fitness goal is met."""
        evaluator = ParallelEvaluator(
            config=parallel_config,
            training_epochs=1,
            early_stop_epochs=1,
            subset_percentage=1.0,
            strategy=mock_strategy,
            progress=MagicMock(),
            session_log_filename="test.log",
            train_indices=None,
            test_indices=None,
        )

        res1 = Result(0, 0.99, 0.01, 1.0, "SUCCESS", [], "cpu")
        evaluator.result_queue.get.side_effect = [res1]

        stop_cond = MagicMock(spec=StopConditions)
        stop_cond.fitness_goal = 0.95

        population = [{"lr": 0.1}, {"lr": 0.2}]  # 2 individuals
        results = evaluator.evaluate_population(
            population, stop_conditions=stop_cond
        )

        assert len(results) == 1
        assert evaluator._fitness_goal_met is True

        evaluator.cleanup_workers()

    def test_shutdown_handling(
        self, mock_mp_context, mock_strategy, parallel_config
    ):
        """Verify behavior when shutting down flag is set."""
        evaluator = ParallelEvaluator(
            config=parallel_config,
            training_epochs=1,
            early_stop_epochs=1,
            subset_percentage=1.0,
            strategy=mock_strategy,
            progress=MagicMock(),
            session_log_filename="test.log",
            train_indices=None,
            test_indices=None,
        )

        evaluator._shutting_down = True

        population = [{"lr": 0.1}]
        results = evaluator.evaluate_population(population, None)

        assert len(results) == 1
        assert results[0].status == "SKIPPED"


class TestTaskStealer:

    def test_stealer_logic(
        self, mock_mp_context, mock_hybrid_strategy, parallel_config
    ):
        """
        Verify the mathematical logic of the task stealer.
        We invoke the private _task_stealer method briefly or inspect logic.
        Since it runs in a thread loop, we'll verify the logic block by setting up queue sizes.
        """
        gpu_q = mock_hybrid_strategy.launch_workers.return_value[
            "gpu_task_queue"
        ]
        cpu_q = mock_hybrid_strategy.launch_workers.return_value[
            "cpu_task_queue"
        ]

        gpu_q.qsize.return_value = 50
        cpu_q.qsize.return_value = 0

        evaluator = ParallelEvaluator(
            config=parallel_config,
            training_epochs=1,
            early_stop_epochs=1,
            subset_percentage=1.0,
            strategy=mock_hybrid_strategy,
            progress=MagicMock(),
            session_log_filename="test.log",
            train_indices=None,
            test_indices=None,
        )

        gpu_q.get.side_effect = ["Task1", queue.Empty]

        evaluator._stealer_wakeup_event.set()

        import time

        time.sleep(0.1)  # Let thread run one iteration

        assert cpu_q.put.called

        evaluator.cleanup_workers()


class TestCleanup:

    def test_cleanup_terminates_workers(
        self, mock_mp_context, mock_strategy, parallel_config
    ):
        evaluator = ParallelEvaluator(
            config=parallel_config,
            training_epochs=1,
            early_stop_epochs=1,
            subset_percentage=1.0,
            strategy=mock_strategy,
            progress=MagicMock(),
            session_log_filename="test.log",
            train_indices=None,
            test_indices=None,
        )

        worker_mock = evaluator._workers[0]
        worker_mock.is_alive.return_value = True

        evaluator.cleanup_workers()

        worker_mock.join.assert_called()
        task_q = mock_strategy.launch_workers.return_value["task_queues"][0]
        assert task_q.put.call_args[0][0] is None
