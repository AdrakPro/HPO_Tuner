from unittest.mock import MagicMock, patch

import pytest
from rich.progress import Progress

from src.evaluator.individual_evaluator import IndividualEvaluator


@pytest.fixture
def mock_deps(mocker):
    """Mocks all major external dependencies for the evaluator."""
    mock_train_eval = mocker.patch(
        "src.evaluator.individual_evaluator.train_and_eval"
    )
    mock_get_loaders = mocker.patch(
        "src.evaluator.individual_evaluator.get_dataset_loaders"
    )
    mock_get_loaders.return_value.__enter__.return_value = (
        MagicMock(),
        MagicMock(),
    )

    mocker.patch("src.evaluator.individual_evaluator.logger")
    mocker.patch("src.evaluator.individual_evaluator.Chromosome")
    mocker.patch("time.perf_counter", side_effect=[100.0, 102.5, 200.0, 205.0])

    return {
        "train_and_eval": mock_train_eval,
        "get_dataset_loaders": mock_get_loaders,
    }


@pytest.fixture
def evaluator(mock_deps):
    """Provides a default IndividualEvaluator instance with mocked dependencies."""
    mock_progress = MagicMock(spec=Progress)
    task_id = MagicMock()

    with patch("torch.cuda.is_available", return_value=False):
        if not hasattr(IndividualEvaluator, "set_task_id"):
            IndividualEvaluator.set_task_id = lambda self, tid: setattr(
                self, "task_id", tid
            )

        instance = IndividualEvaluator(
            neural_config={},
            training_epochs=10,
            early_stop_epochs=3,
            subset_percentage=1.0,
            progress=mock_progress,
            task_id=task_id,
            train_indices=None,
            test_indices=None,
        )
    return instance, mock_progress


class TestIndividualEvaluator:

    def test_initialization_on_cpu(self):
        """Verify the evaluator initializes on CPU when CUDA is unavailable."""
        if not hasattr(IndividualEvaluator, "set_task_id"):
            IndividualEvaluator.set_task_id = lambda self, tid: setattr(
                self, "task_id", tid
            )

        with patch("torch.cuda.is_available", return_value=False):
            evaluator = IndividualEvaluator(
                {}, 1, 1, 1.0, MagicMock(), MagicMock(), None, None
            )
            assert evaluator.device.type == "cpu"

    def test_initialization_on_gpu(self):
        """Verify the evaluator initializes on CUDA when it is available."""
        if not hasattr(IndividualEvaluator, "set_task_id"):
            IndividualEvaluator.set_task_id = lambda self, tid: setattr(
                self, "task_id", tid
            )

        with patch("torch.cuda.is_available", return_value=True):
            evaluator = IndividualEvaluator(
                {}, 1, 1, 1.0, MagicMock(), MagicMock(), None, None
            )
            assert evaluator.device.type == "cuda"

    def test_evaluate_population_success_path(self, evaluator, mock_deps):
        """Test a full, successful evaluation of a small population."""
        instance, mock_progress = evaluator
        mock_deps["train_and_eval"].side_effect = [(0.95, 0.1), (0.92, 0.15)]

        population = [{"lr": 0.01}, {"lr": 0.005}]
        results = instance.evaluate_population(population)

        # 1. Check results list
        assert len(results) == 2

        # 2. Check first result details
        result1 = results[0]
        assert result1.status == "SUCCESS"
        assert result1.fitness == 0.95
        assert result1.loss == 0.1
        assert result1.duration_seconds == pytest.approx(2.5)  # 102.5 - 100.0
        assert result1.error_message is None

        # 3. Check second result details
        result2 = results[1]
        assert result2.status == "SUCCESS"
        assert result2.fitness == 0.92

        # 4. Verify dependencies were called correctly
        assert mock_deps["train_and_eval"].call_count == 2
        assert mock_deps["get_dataset_loaders"].call_count == 2
        assert mock_progress.update.call_count == 2

    def test_evaluate_population_handles_exception(self, evaluator, mock_deps):
        """Verify that an exception during training creates a 'FAILURE' result."""
        instance, mock_progress = evaluator
        # First individual succeeds, second fails
        mock_deps["train_and_eval"].side_effect = [
            (0.90, 0.2),
            Exception("CUDA out of memory"),
        ]

        population = [{"lr": 0.01}, {"lr": 0.001}]
        results = instance.evaluate_population(population)

        assert len(results) == 2

        failed_result = results[1]
        assert failed_result.status == "FAILURE"
        assert failed_result.fitness == 0.0
        assert failed_result.loss == float("inf")
        assert "CUDA out of memory" in failed_result.error_message

        assert mock_progress.update.call_count == 2
