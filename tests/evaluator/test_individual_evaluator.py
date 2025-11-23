from unittest.mock import MagicMock, patch

import pytest

from src.evaluator.individual_evaluator import IndividualEvaluator
from src.genetic.stop_conditions import StopConditions


@pytest.fixture
def neural_config():
    return {"input_shape": (3, 32, 32)}


@pytest.fixture
def evaluator(neural_config):
    return IndividualEvaluator(
        neural_config=neural_config,
        training_epochs=1,
        early_stop_epochs=1,
        subset_percentage=1.0,
        progress=MagicMock(),
        train_indices=None,
        test_indices=None,
    )


class TestIndividualEvaluator:

    @patch("src.evaluator.individual_evaluator.train_and_eval")
    @patch("src.evaluator.individual_evaluator.get_dataset_loaders")
    @patch("src.evaluator.individual_evaluator.Chromosome.from_dict")
    def test_evaluate_population_success(
        self, mock_chrom, mock_loaders, mock_train, evaluator
    ):
        """Verify sequential evaluation of a population."""
        mock_train.return_value = (0.85, 0.5)  # Acc, Loss
        mock_loaders.return_value.__enter__.return_value = (
            MagicMock(),
            MagicMock(),
        )

        population = [{"lr": 0.1}, {"lr": 0.2}]
        results = evaluator.evaluate_population(population)

        assert len(results) == 2
        assert results[0].fitness == 0.85
        assert results[0].status == "SUCCESS"
        assert mock_train.call_count == 2

    @patch("src.evaluator.individual_evaluator.train_and_eval")
    @patch("src.evaluator.individual_evaluator.get_dataset_loaders")
    @patch("src.evaluator.individual_evaluator.Chromosome.from_dict")
    def test_evaluate_handles_exception(
        self, mock_chrom, mock_loaders, mock_train, evaluator
    ):
        """Verify individual failure is caught and logged."""
        mock_loaders.return_value.__enter__.return_value = (
            MagicMock(),
            MagicMock(),
        )
        mock_train.side_effect = [(0.9, 0.1), RuntimeError("OOM")]

        population = [{"lr": 0.1}, {"lr": 0.2}]
        results = evaluator.evaluate_population(population)

        assert len(results) == 2

        assert results[0].status == "SUCCESS"
        assert results[0].fitness == 0.9

        assert results[1].status == "FAILURE"
        assert results[1].fitness == 0.0
        assert "OOM" in results[1].error_message

    @patch("src.evaluator.individual_evaluator.train_and_eval")
    @patch("src.evaluator.individual_evaluator.get_dataset_loaders")
    @patch("src.evaluator.individual_evaluator.Chromosome.from_dict")
    def test_early_stop_condition(
        self, mock_chrom, mock_loaders, mock_train, evaluator
    ):
        """Verify loop breaks if StopCondition is met."""
        mock_loaders.return_value.__enter__.return_value = (
            MagicMock(),
            MagicMock(),
        )
        mock_train.side_effect = [(0.99, 0.01), (0.1, 0.9)]

        stop_cond = MagicMock(spec=StopConditions)
        stop_cond.should_stop_evaluation.side_effect = [
            (True, "Goal met"),
            (False, ""),
        ]

        population = [{"lr": 0.1}, {"lr": 0.2}]
        results = evaluator.evaluate_population(
            population, stop_conditions=stop_cond
        )

        assert len(results) == 2
        assert results[0].status == "SUCCESS"
        assert results[0].fitness == 0.99

        assert results[1].status == "CANCELLED"
        assert results[1].fitness == 0.0

    def test_property_accessors(self, evaluator):
        """Test simple setters and properties."""
        assert evaluator.num_workers == 1

        evaluator.set_training_epochs(10)
        assert evaluator.training_epochs == 10

        evaluator.set_subset_percentage(0.5)
        assert evaluator.subset_percentage == 0.5
