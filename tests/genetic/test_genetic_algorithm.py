from unittest.mock import MagicMock, patch

import pytest

MODULE_PATH = "src.genetic.genetic_algorithm"

from src.genetic.genetic_algorithm import (
    GeneticAlgorithm,
    DataType,
    get_chromosome_search_space,
)


@pytest.fixture
def mock_logger(mocker):
    return mocker.patch(f"{MODULE_PATH}.logger")


@pytest.fixture
def ga_config():
    return {
        "active": ["selection", "crossover", "mutation", "elitism"],
        "elitism_percent": 0.1,
        "crossover": {"crossover_prob": 0.5},
        "mutation": {
            "mutation_prob_discrete": 0.5,
            "mutation_prob_categorical": 0.5,
            "mutation_prob_continuous": 1.0,
            "mutation_sigma_continuous": 0.1,
        },
    }


@pytest.fixture
def chromosome_space():
    """Full space for general tests."""
    return {
        "gene_float": {"type": DataType.CONTINUOUS, "min": 0.0, "max": 10.0},
        "gene_log": {
            "type": DataType.CONTINUOUS,
            "min": 1e-4,
            "max": 1e-1,
            "scale": "log",
        },
        "gene_int": {"type": DataType.DISCRETE, "values": [1, 2, 3, 4, 5]},
        "gene_cat": {"type": DataType.CATEGORICAL, "values": ["A", "B", "C"]},
    }


@pytest.fixture
def ga_instance(ga_config, chromosome_space, mock_logger):
    return GeneticAlgorithm(ga_config, chromosome_space)


class TestInitialization:
    def test_init_parses_config(self, ga_instance, ga_config):
        assert ga_instance.mutation_prob_discrete == 0.5
        assert ga_instance.mutation_sigma_continuous == 0.1
        assert "selection" in ga_instance.active_operators

    def test_random_mode(self, chromosome_space, mock_logger):
        config = {
            "active": ["random"],
            "mutation": {
                "mutation_prob_discrete": 0.1,
                "mutation_prob_categorical": 0.1,
                "mutation_prob_continuous": 0.1,
                "mutation_sigma_continuous": 0.1,
            },
        }
        ga = GeneticAlgorithm(config, chromosome_space)
        assert ga._is_random_mode is True
        assert len(ga.active_operators) == 0

    def test_search_space_parser(self):
        raw_config = {
            "neural_network_config": {
                "hyperparameter_space": {
                    "lr": {
                        "type": "float",
                        "range": [0.001, 0.1],
                        "scale": "log",
                    },
                    "layers": {"type": "int", "values": [1, 2, 3]},
                    "activation_fn": {
                        "type": "enum",
                        "values": ["relu", "gelu"],
                    },
                }
            }
        }
        space = get_chromosome_search_space(raw_config)
        assert space["lr"]["type"] == DataType.CONTINUOUS
        assert space["layers"]["type"] == DataType.DISCRETE
        assert space["activation_fn"]["type"] == DataType.CATEGORICAL


class TestInitialPopulation:

    def test_guaranteed_diversity(self, ga_instance):
        pop = ga_instance.initial_population(pop_size=10)

        int_values = {ind["gene_int"] for ind in pop}
        assert int_values == {1, 2, 3, 4, 5}

        cat_values = {ind["gene_cat"] for ind in pop}
        assert cat_values == {"A", "B", "C"}

    def test_lhs_integration(self, ga_instance):
        with patch("scipy.stats.qmc.LatinHypercube") as mock_lhs:
            mock_sampler = MagicMock()
            mock_sampler.random.return_value = [[0.0, 0.0]] * 10
            mock_lhs.return_value = mock_sampler

            ga_instance.initial_population(pop_size=10)

            mock_lhs.assert_called_with(
                d=2, seed=2137, optimization="random-cd"
            )
            assert mock_sampler.random.called

    def test_population_size_and_warning(self, ga_instance, mock_logger):
        pop = ga_instance.initial_population(pop_size=2)
        assert len(pop) == 2
        mock_logger.warning.assert_called()


class TestOperators:

    def test_tournament_selection(self, ga_instance):
        pop = [{"id": 1}, {"id": 2}, {"id": 3}, {"id": 4}]
        fitness = [0.1, 0.2, 0.9, 0.4]

        with patch("random.sample", return_value=[1, 2]):
            winner = ga_instance.tournament_selection(pop, fitness)

        assert winner["id"] == 3

    def test_uniform_crossover(self, ga_instance):
        p1 = {"a": 1, "b": 1, "c": 1}
        p2 = {"a": 2, "b": 2, "c": 2}

        with patch("random.random", side_effect=[0.0, 0.9, 0.0]):
            child = ga_instance.uniform_crossover(p1, p2)

        assert child == {"a": 1, "b": 2, "c": 1}

    def test_mutation_continuous_clamping(self, ga_config, mock_logger):
        """Test that mutation respects min/max bounds."""
        minimal_space = {
            "gene_float": {"type": DataType.CONTINUOUS, "min": 0.0, "max": 10.0}
        }
        ga = GeneticAlgorithm(ga_config, minimal_space)
        chromosome = {"gene_float": 9.9}

        with patch("random.gauss", return_value=5.0):
            with patch("random.random", return_value=0.0):
                mutated = ga.mutate(chromosome)

        assert mutated["gene_float"] == 10.0

    def test_mutation_discrete_change(self, ga_config, mock_logger):
        """Test that discrete mutation picks a DIFFERENT value."""
        minimal_space = {
            "gene_int": {"type": DataType.DISCRETE, "values": [1, 2, 3, 4, 5]}
        }
        ga = GeneticAlgorithm(ga_config, minimal_space)
        chromosome = {"gene_int": 1}

        with patch("random.random", return_value=0.0):
            with patch("random.choice", return_value=5) as mock_choice:
                mutated = ga.mutate(chromosome)

                args, _ = mock_choice.call_args
                assert 1 not in args[0]
                assert mutated["gene_int"] == 5

    def test_elitism(self, ga_instance):
        pop = [{"val": 10}, {"val": 20}, {"val": 30}, {"val": 40}]
        fitness = [0.1, 0.2, 0.3, 0.4]

        elite = ga_instance.elitism(pop, fitness)
        assert len(elite) == 1
        assert elite[0]["val"] == 40


class TestRunGeneration:

    def test_run_generation_flow(self, ga_instance):
        pop_size = 4
        pop = ga_instance.initial_population(pop_size)
        fitness = [0.5] * pop_size

        ga_instance.elitism = MagicMock(return_value=[pop[0]])
        ga_instance.tournament_selection = MagicMock(
            side_effect=lambda p, f: p[0]
        )
        ga_instance.uniform_crossover = MagicMock(return_value=pop[0])
        ga_instance.mutate = MagicMock(return_value=pop[0])

        new_pop = ga_instance.run_generation(pop, fitness)

        assert len(new_pop) == pop_size
        assert ga_instance.elitism.called
        assert ga_instance.tournament_selection.called
        assert ga_instance.uniform_crossover.called
        assert ga_instance.mutate.called

    def test_random_mode_selection(self, ga_instance):
        ga_instance._is_random_mode = True
        pop = [{"a": 1}, {"a": 2}]
        fitness = [1.0, 1.0]

        with patch("random.sample") as mock_sample:
            mock_sample.return_value = ["mutation"]
            ga_instance.mutate = MagicMock(return_value={"a": 3})

            ga_instance.run_generation(pop, fitness)

            assert mock_sample.called
            assert ga_instance.mutate.called


class TestAdaptiveMutation:
    def test_decay(self, ga_instance):
        initial_prob = ga_instance.mutation_prob_discrete
        decay = 0.9
        gen = 2

        ga_instance.set_adaptive_mutation(decay, gen)

        expected = initial_prob * (decay**gen)
        assert ga_instance.mutation_prob_discrete == pytest.approx(expected)
