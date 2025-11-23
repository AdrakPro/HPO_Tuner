from unittest.mock import patch

import pytest

from src.genetic.stop_conditions import StopConditions


@pytest.fixture
def base_config():
    """Standard configuration for testing."""
    return {
        "early_stop_generations": 3,
        "fitness_goal": 0.95,
        "time_limit_minutes": 10.0,
    }


class TestStopConditions:

    def test_init_sets_attributes(self, base_config):
        """Verify attributes are correctly unpacked from config."""
        stopper = StopConditions(base_config)
        assert stopper.early_stop_generations == 3
        assert stopper.fitness_goal == 0.95
        assert stopper.time_limit_minutes == 10.0
        assert stopper.best_fitness_history == []
        assert isinstance(stopper.start_time, float)

    @pytest.mark.parametrize(
        "fitness,expected_stop",
        [
            (0.50, False),  # Below goal
            (0.94, False),  # Just below goal
            (0.95, True),  # Exactly goal
            (0.99, True),  # Above goal
        ],
    )
    def test_should_stop_evaluation_fitness_goal(
        self, base_config, fitness, expected_stop
    ):
        """Test immediate stop if individual hits target fitness."""
        stopper = StopConditions(base_config)
        stop, reason = stopper.should_stop_evaluation(
            individual_fitness=fitness
        )

        assert stop is expected_stop
        if stop:
            assert "Fitness goal" in reason

    def test_time_limit_not_exceeded(self, base_config):
        """Ensure algorithm continues if time is within limit."""
        stopper = StopConditions(base_config)

        with patch("time.monotonic") as mock_time:
            stopper.start_time = 1000
            mock_time.return_value = 1000 + (5 * 60)  # 5 mins elapsed

            stop, _ = stopper.should_stop_algorithm(
                current_generation=1, best_fitness=0.1
            )

        assert stop is False

    def test_time_limit_exceeded(self, base_config):
        """Ensure algorithm stops if time limit is exceeded."""
        stopper = StopConditions(base_config)

        with patch("time.monotonic") as mock_time:
            stopper.start_time = 1000
            mock_time.return_value = 1000 + (11 * 60)

            stop, reason = stopper.should_stop_algorithm(
                current_generation=1, best_fitness=0.1
            )

        assert stop is True
        assert "Exceeded time limit" in reason

    def test_time_limit_disabled(self, base_config):
        """Test that setting limit to 0 disables the check."""
        base_config["time_limit_minutes"] = 0
        stopper = StopConditions(base_config)

        with patch("time.monotonic") as mock_time:
            stopper.start_time = 1000
            mock_time.return_value = 1000 + (1000 * 60)

            stop, _ = stopper.should_stop_algorithm(
                current_generation=1, best_fitness=0.1
            )

        assert stop is False

    def test_early_stopping_not_triggered_early_generations(self, base_config):
        """Don't stop if we haven't reached 'early_stop_generations' count yet."""
        stopper = StopConditions(base_config)

        stop, _ = stopper.should_stop_algorithm(
            current_generation=1, best_fitness=0.5
        )
        assert stop is False

        stop, _ = stopper.should_stop_algorithm(
            current_generation=2, best_fitness=0.5
        )
        assert stop is False

        stop, _ = stopper.should_stop_algorithm(
            current_generation=3, best_fitness=0.5
        )
        assert stop is False

    def test_early_stopping_triggered_on_stagnation(self, base_config):
        """Stop if fitness remains identical for N generations."""
        stopper = StopConditions(base_config)

        stopper.should_stop_algorithm(1, 0.5)
        stopper.should_stop_algorithm(2, 0.5)
        stopper.should_stop_algorithm(3, 0.5)

        stop, reason = stopper.should_stop_algorithm(
            current_generation=4, best_fitness=0.5
        )

        assert stop is True
        assert "No improvement" in reason
        assert len(stopper.best_fitness_history) == 4

    def test_early_stopping_not_triggered_on_improvement(self, base_config):
        """Continue if fitness is changing/improving."""
        stopper = StopConditions(base_config)

        stopper.should_stop_algorithm(1, 0.5)
        stopper.should_stop_algorithm(2, 0.6)
        stopper.should_stop_algorithm(3, 0.6)  # Stagnant for 1 gen

        stop, _ = stopper.should_stop_algorithm(
            current_generation=4, best_fitness=0.7
        )

        assert stop is False

    def test_early_stopping_fluctuation(self, base_config):
        """Ensure strictly identical values trigger stop (based on current implementation logic)."""
        stopper = StopConditions(base_config)

        stopper.should_stop_algorithm(1, 0.5)
        stopper.should_stop_algorithm(2, 0.50001)
        stopper.should_stop_algorithm(3, 0.5)

        stop, _ = stopper.should_stop_algorithm(
            current_generation=4, best_fitness=0.5
        )
        assert stop is False
