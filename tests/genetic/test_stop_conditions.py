from unittest.mock import patch

import pytest

from src.genetic.stop_conditions import StopConditions


@pytest.fixture
def stop_config():
    """Provides a default configuration for StopConditions."""
    return {
        "early_stop_generations": 3,
        "fitness_goal": 0.95,
        "time_limit_minutes": 10,
    }


@pytest.fixture
def sc_instance(stop_config):
    """Returns an instance of StopConditions."""
    return StopConditions(stop_config)


def test_initialization(sc_instance, stop_config):
    """Test that the class initializes correctly."""
    assert (
        sc_instance.early_stop_generations
        == stop_config["early_stop_generations"]
    )
    assert sc_instance.fitness_goal == stop_config["fitness_goal"]
    assert sc_instance.time_limit_minutes == stop_config["time_limit_minutes"]
    assert sc_instance.start_time is not None
    assert sc_instance.best_fitness_history == []


@pytest.mark.parametrize(
    "fitness, expected_stop, expected_reason_part",
    [
        (0.96, True, "Fitness goal of 0.95 reached"),  # Exceeds goal
        (0.95, True, "Fitness goal of 0.95 reached"),  # Meets goal
        (0.94, False, ""),  # Below goal
    ],
)
def test_should_stop_evaluation(
    sc_instance, fitness, expected_stop, expected_reason_part
):
    """Test the evaluation stop condition based on fitness goal."""
    should_stop, reason = sc_instance.should_stop_evaluation(fitness)
    assert should_stop == expected_stop
    assert expected_reason_part in reason


def test_should_stop_algorithm_time_limit_exceeded(stop_config):
    """Test that the algorithm stops when the time limit is exceeded."""
    stop_config["time_limit_minutes"] = 1
    sc = StopConditions(stop_config)

    with patch("time.monotonic", return_value=sc.start_time + 61):
        should_stop, reason = sc.should_stop_algorithm(
            current_generation=1, best_fitness=0.5
        )
        assert should_stop is True
        assert "Exceeded time limit of 1 minutes" in reason


def test_should_stop_algorithm_time_limit_not_exceeded(sc_instance):
    """Test that the algorithm continues when the time limit is not reached."""
    should_stop, reason = sc_instance.should_stop_algorithm(
        current_generation=1, best_fitness=0.5
    )
    assert should_stop is False
    assert reason == ""


def test_should_stop_algorithm_time_limit_disabled(stop_config):
    """Test that a time limit of 0 is ignored."""
    stop_config["time_limit_minutes"] = 0
    sc = StopConditions(stop_config)
    with patch("time.monotonic", return_value=sc.start_time + 10000):
        should_stop, reason = sc.should_stop_algorithm(
            current_generation=1, best_fitness=0.5
        )
        assert should_stop is False


def test_should_stop_algorithm_early_stopping_triggered(sc_instance):
    """Test early stopping when fitness does not improve."""
    sc_instance.should_stop_algorithm(current_generation=1, best_fitness=0.8)
    sc_instance.should_stop_algorithm(current_generation=2, best_fitness=0.9)
    sc_instance.should_stop_algorithm(current_generation=3, best_fitness=0.9)
    sc_instance.should_stop_algorithm(current_generation=4, best_fitness=0.9)

    should_stop, reason = sc_instance.should_stop_algorithm(
        current_generation=5, best_fitness=0.9
    )

    assert should_stop is True
    assert "No improvement in best fitness for the last 3 generations" in reason


def test_should_stop_algorithm_early_stopping_not_triggered_due_to_improvement(
    sc_instance,
):
    """Test that early stopping is not triggered if fitness improves."""
    sc_instance.should_stop_algorithm(current_generation=1, best_fitness=0.8)
    sc_instance.should_stop_algorithm(current_generation=2, best_fitness=0.9)
    sc_instance.should_stop_algorithm(current_generation=3, best_fitness=0.9)

    should_stop, reason = sc_instance.should_stop_algorithm(
        current_generation=4, best_fitness=0.91
    )
    assert should_stop is False


def test_should_stop_algorithm_early_stopping_not_triggered_due_to_insufficient_generations(
    sc_instance,
):
    """Test that early stopping is not checked before enough generations have passed."""
    sc_instance.should_stop_algorithm(current_generation=1, best_fitness=0.9)
    sc_instance.should_stop_algorithm(current_generation=2, best_fitness=0.9)

    should_stop, reason = sc_instance.should_stop_algorithm(
        current_generation=3, best_fitness=0.9
    )
    assert should_stop is False


def test_should_stop_algorithm_no_condition_met(sc_instance):
    """Test that the algorithm continues when no stop conditions are met."""
    should_stop, reason = sc_instance.should_stop_algorithm(
        current_generation=1, best_fitness=0.5
    )
    assert should_stop is False
    assert reason == ""


def test_best_fitness_history_is_updated(sc_instance):
    """Test that best_fitness_history is correctly appended to on each call."""
    sc_instance.should_stop_algorithm(current_generation=1, best_fitness=0.1)
    sc_instance.should_stop_algorithm(current_generation=2, best_fitness=0.2)
    assert sc_instance.best_fitness_history == [0.1, 0.2]
