import random
from unittest.mock import patch, MagicMock

import pytest

from src.config.default_config import get_default_config
from src.genetic.genetic_algorithm import (
    GeneticAlgorithm,
    DataType,
    get_chromosome_search_space,
)


@pytest.fixture(autouse=True)
def mock_logger(monkeypatch):
    """Automatically mocks the logger for all tests."""
    mock = MagicMock()
    monkeypatch.setattr("src.genetic.genetic_algorithm.logger", mock)
    return mock


@pytest.fixture(scope="module")
def full_config():
    """Fixture to load the entire default configuration once per module."""
    return get_default_config()


@pytest.fixture
def chromosome_space(full_config):
    """Fixture that generates the chromosome search space from the config."""
    return get_chromosome_search_space(full_config)


@pytest.fixture
def ga_config(full_config):
    """Fixture for the genetic operators configuration section."""
    return full_config["genetic_algorithm_config"]["genetic_operators"]


@pytest.fixture
def ga_instance(ga_config, chromosome_space):
    """Fixture for a standard GeneticAlgorithm instance."""
    return GeneticAlgorithm(ga_config, chromosome_space)


def test_initialization(ga_instance, ga_config, chromosome_space):
    """Test if the GA is initialized with the correct attributes from the config."""
    assert ga_instance.config == ga_config
    assert ga_instance.chromosome_space == chromosome_space
    assert ga_instance.active_operators == set(ga_config["active"])
    assert not ga_instance._is_random_mode


def test_get_chromosome_search_space(chromosome_space, full_config):
    """Test if the search space is parsed correctly from the config."""
    nn_hyperparameter_space = full_config["neural_network_config"][
        "hyperparameter_space"
    ]

    # Test a continuous gene
    assert "width_scale" in chromosome_space
    assert chromosome_space["width_scale"]["type"] == DataType.CONTINUOUS
    assert (
        chromosome_space["width_scale"]["min"]
        == nn_hyperparameter_space["width_scale"]["range"][0]
    )
    assert (
        chromosome_space["width_scale"]["max"]
        == nn_hyperparameter_space["width_scale"]["range"][1]
    )

    # Test a categorical gene
    assert "optimizer_schedule" in chromosome_space
    assert (
        chromosome_space["optimizer_schedule"]["type"] == DataType.CATEGORICAL
    )
    assert (
        chromosome_space["optimizer_schedule"]["values"]
        == nn_hyperparameter_space["optimizer_schedule"]["values"]
    )

    # Test a discrete gene
    assert "batch_size" in chromosome_space
    assert chromosome_space["batch_size"]["type"] == DataType.DISCRETE
    assert (
        chromosome_space["batch_size"]["values"]
        == nn_hyperparameter_space["batch_size"]["values"]
    )


def test_initial_population_size(ga_instance):
    """Test if the initial population is generated with the correct size."""
    pop_size = 50
    population = ga_instance.initial_population(pop_size, print_warning=False)
    assert len(population) == pop_size
    assert len({str(p) for p in population}) == pop_size


def test_initial_population_diversity(ga_instance):
    """Test if the initial population is diverse for categorical/discrete genes."""
    pop_size = 100
    population = ga_instance.initial_population(
        pop_size, strat_bins=5, print_warning=False
    )

    optimizers = {p["optimizer_schedule"] for p in population}
    expected_optimizers = set(
        ga_instance.chromosome_space["optimizer_schedule"]["values"]
    )
    assert expected_optimizers.issubset(optimizers)

    batch_sizes = {p["batch_size"] for p in population}
    expected_batch_sizes = set(
        ga_instance.chromosome_space["batch_size"]["values"]
    )
    assert expected_batch_sizes.issubset(batch_sizes)


def test_tournament_selection(ga_instance):
    """Test if tournament selection correctly picks the individual with the highest fitness."""
    population = [{"id": i} for i in range(10)]
    fitness = [0.1, 0.2, 0.9, 0.4, 0.5, 0.6, 0.7, 0.8, 0.3, 0.0]

    with patch("random.sample", return_value=[0, 4, 2]):
        winner = ga_instance.tournament_selection(population, fitness)
        assert winner == {"id": 2}


def test_uniform_crossover(ga_instance):
    """Test if uniform crossover correctly combines genes from two parents."""
    parent1 = {"dropout_rate": 0.2, "batch_size": 128}
    parent2 = {"dropout_rate": 0.5, "batch_size": 256}

    with patch("random.random", side_effect=[0.4, 0.9]):
        ga_instance.config["crossover"]["crossover_prob"] = 0.8
        child = ga_instance.uniform_crossover(parent1, parent2)
        assert child == {"dropout_rate": 0.2, "batch_size": 256}


@patch("random.random", return_value=0.05)
@patch("random.gauss", return_value=0.01)
def test_mutate_continuous_linear(mock_gauss, mock_random, ga_instance):
    """Test mutation for a continuous gene with a linear scale."""
    chromosome = {"dropout_rate": 0.3}
    ga_instance.chromosome_space = {
        "dropout_rate": {"type": DataType.CONTINUOUS, "min": 0.2, "max": 0.5}
    }

    mutated = ga_instance.mutate(chromosome)

    assert mutated["dropout_rate"] == pytest.approx(0.3 + 0.01)


@patch("random.random", return_value=0.05)
@patch("random.choice", return_value="STRONG")
def test_mutate_categorical(mock_choice, mock_random, ga_instance):
    """Test mutation for a categorical gene."""
    chromosome = {"aug_intensity": "MEDIUM"}
    ga_instance.chromosome_space = {
        "aug_intensity": {
            "type": DataType.CATEGORICAL,
            "values": ["MEDIUM", "STRONG"],
        }
    }

    mutated = ga_instance.mutate(chromosome)

    assert mutated["aug_intensity"] == "STRONG"
    mock_choice.assert_called_with(["STRONG"])


def test_elitism(ga_instance):
    """Test if elitism correctly selects the top-performing individuals."""
    population = [{"id": i} for i in range(10)]
    fitness = [0.1, 0.9, 0.3, 0.8, 0.5, 0.4, 0.6, 0.7, 0.2, 0.0]
    ga_instance.config["elitism_percent"] = 0.2

    elite = ga_instance.elitism(population, fitness)

    assert len(elite) == 2
    elite_ids = {e["id"] for e in elite}
    assert elite_ids == {1, 3}


def test_run_generation_preserves_best_individual(ga_instance):
    """Test if run_generation preserves the best individual and maintains population size."""
    pop_size = 20
    population = ga_instance.initial_population(pop_size, print_warning=False)
    fitness = [random.random() for _ in range(pop_size)]

    best_original_idx = max(range(len(fitness)), key=lambda i: fitness[i])
    best_original_individual = population[best_original_idx]

    new_population = ga_instance.run_generation(population, fitness)

    assert len(new_population) == pop_size
    assert best_original_individual in new_population
