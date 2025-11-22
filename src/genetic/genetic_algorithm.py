import json
import math
import random
from copy import deepcopy
from enum import Enum, auto
from typing import Any, Dict, List, Optional

from scipy.stats import qmc

from src.logger.logger import logger


# TODO: add chromosome typing for Any
class GeneticAlgorithm:
    def __init__(
        self,
        ga_operators_config: Dict[str, Any],
        chromosome_space: Dict[str, Any],
    ):
        """
        Args:
            ga_operators_config: Genetic operators configuration dictionary.
            chromosome_space: Search space and type for each gene.
        """
        self.config = ga_operators_config
        self.chromosome_space = chromosome_space

        self.available_operators = {
            "selection",
            "crossover",
            "mutation",
            "elitism",
        }

        self.mutation_prob_discrete: float = self.config["mutation"][
            "mutation_prob_discrete"
        ]
        self.mutation_prob_categorical: float = self.config["mutation"][
            "mutation_prob_categorical"
        ]
        self.mutation_prob_continuous: float = self.config["mutation"][
            "mutation_prob_continuous"
        ]
        self.mutation_sigma_continuous: float = self.config["mutation"][
            "mutation_sigma_continuous"
        ]

        self._is_random_mode = self.config["active"] == ["random"]

        if not self._is_random_mode:
            self.active_operators = set(self.config["active"])
        else:
            self.active_operators = set()

    def tournament_selection(self, population: List[Any], fitness: List[float]):
        tournament_size = max(1, int(len(population) ** 0.5))

        if tournament_size > len(population):
            tournament_size = len(population)

        selected_indices = random.sample(
            range(len(population)), tournament_size
        )
        best_idx = max(selected_indices, key=lambda i: fitness[i])

        return deepcopy(population[best_idx])

    def uniform_crossover(self, parent1: Dict, parent2: Dict) -> Dict:
        child = {}
        crossover_prob: int = self.config["crossover"]["crossover_prob"]
        for k in parent1:
            child[k] = (
                parent1[k] if random.random() < crossover_prob else parent2[k]
            )
        return child

    def mutate(self, chromosome: Dict) -> Dict:
        for gene, info in self.chromosome_space.items():
            if info["type"] == DataType.CONTINUOUS:
                if random.random() < self.mutation_prob_continuous:
                    is_valid_log = (
                        info.get("scale") == "log" and info["min"] > 0.0
                    )

                    if is_valid_log:
                        log_min = math.log10(info["min"])
                        log_max = math.log10(info["max"])
                        range_size = log_max - log_min
                        sigma = self.mutation_sigma_continuous * range_size

                        current_val = chromosome[gene]
                        if current_val > 0:
                            log_val = math.log10(current_val)
                            mutated_log = log_val + random.gauss(0, sigma)
                            mutated = 10**mutated_log
                        else:
                            linear_range = info["max"] - info["min"]
                            linear_sigma = (
                                self.mutation_sigma_continuous * linear_range
                            )
                            mutated = current_val + random.gauss(
                                0, linear_sigma
                            )

                    else:
                        range_size = info["max"] - info["min"]
                        sigma = self.mutation_sigma_continuous * range_size
                        mutated = chromosome[gene] + random.gauss(0, sigma)

                    chromosome[gene] = float(
                        max(min(mutated, info["max"]), info["min"])
                    )
            else:
                prob = self.mutation_prob_discrete
                if info["type"] == DataType.CATEGORICAL:
                    prob = self.mutation_prob_categorical

                if random.random() < prob:
                    possible = [
                        v for v in info["values"] if v != chromosome[gene]
                    ]
                    if possible:
                        chromosome[gene] = random.choice(possible)
        return chromosome

    def elitism(self, population: List[Any], fitness: List[float]) -> List[Any]:
        percent = self.config["elitism_percent"]
        elite_num = max(1, int(len(population) * percent))
        elite_indices = sorted(range(len(fitness)), key=lambda i: fitness[i])[
            -elite_num:
        ]

        return [deepcopy(population[i]) for i in elite_indices]

    def set_adaptive_mutation(
        self,
        decay_rate: float,
        gen: int,
    ) -> None:
        self.mutation_prob_discrete *= decay_rate**gen
        self.mutation_prob_categorical *= decay_rate**gen
        self.mutation_prob_continuous *= decay_rate**gen

    def run_generation(
        self, population: List[Dict], fitness: List[float]
    ) -> List[Dict]:
        pop_size = len(population)
        new_pop = []

        if self._is_random_mode:
            numer_of_ga_operators = min(2, len(self.available_operators))
            active_ops = set(
                random.sample(
                    list(self.available_operators), numer_of_ga_operators
                )
            )
        else:
            active_ops = self.active_operators

        if "elitism" in active_ops:
            # Avoid adding the best individual twice
            elite = self.elitism(population, fitness)
            for ind in elite:
                if len(new_pop) < pop_size:
                    new_pop.append(deepcopy(ind))

        # Main loop
        while len(new_pop) < pop_size:
            if "crossover" in active_ops:
                # If crossover is active, we need two parents
                if "selection" in active_ops:
                    # If selection is also active, use tournament selection
                    parent1 = self.tournament_selection(population, fitness)
                    parent2 = self.tournament_selection(population, fitness)
                else:
                    # If crossover is active without selection, pick parents randomly
                    if pop_size >= 2:
                        parents = random.sample(population, 2)
                    else:
                        parents = [
                            deepcopy(population[0]),
                            deepcopy(population[0]),
                        ]
                    parent1, parent2 = deepcopy(parents[0]), deepcopy(
                        parents[1]
                    )

                child = self.uniform_crossover(parent1, parent2)

            elif "selection" in active_ops:
                # If only selection is active (no crossover), the selected individual becomes the child
                child = self.tournament_selection(population, fitness)

            else:
                # If neither crossover nor selection is active, take a random individual
                child = deepcopy(random.choice(population))

            if "mutation" in active_ops:
                child = self.mutate(child)

            new_pop.append(child)

        return new_pop[:pop_size]

    def initial_population(
        self, pop_size: int, print_warning: bool = True
    ) -> List[Dict]:
        """
        Generate a diverse initial population.
        1. Guarantees coverage for all categorical/discrete values.
        2. Fills the remaining population using Latin Hypercube Sampling (LHS)
           for all continuous values to ensure broad, even exploration.
        """
        population: List[Dict[str, Any]] = []

        guaranteed_individuals: List[Dict[str, Any]] = []

        for gene, info in self.chromosome_space.items():
            if info["type"] in [DataType.DISCRETE, DataType.CATEGORICAL]:
                for v in info["values"]:
                    # Use the simplified generator
                    individual = self._generate_individual(
                        forced_gene=gene, forced_value=v
                    )
                    guaranteed_individuals.append(individual)

        # Diversity implies a minimal pop_size
        min_required = len(guaranteed_individuals)

        if print_warning and pop_size < min_required:
            logger.warning(
                f"Requested population size ({pop_size}) is smaller than the minimum required to guarantee diversity ({min_required})."
                " Some values may not be represented."
            )

        population.extend(guaranteed_individuals)

        # De-duplicate guaranteed individuals (in case of overlap)
        existing_population = {
            json.dumps(ind, sort_keys=True): ind for ind in population
        }
        population = list(existing_population.values())

        num_remaining = pop_size - len(population)

        if num_remaining > 0:
            continuous_genes = {
                gene: info
                for gene, info in self.chromosome_space.items()
                if info["type"] == DataType.CONTINUOUS
            }
            num_dimensions = len(continuous_genes)

            if num_dimensions > 0:
                # TODO make seed configurable, now self.config is genetic_config not overall object
                sampler_seed = 2137
                sampler = qmc.LatinHypercube(
                    d=num_dimensions,
                    seed=sampler_seed,
                    optimization="random-cd",
                )

                # Get N unit-scaled samples (values 0.0 to 1.0)
                unit_samples = sampler.random(n=num_remaining)

                # Rescale samples and create individuals
                lhs_individuals = []
                gene_names = list(continuous_genes.keys())

                for i in range(num_remaining):
                    individual = self._generate_individual()
                    sample_row = unit_samples[i]

                    for j, gene in enumerate(gene_names):
                        info = continuous_genes[gene]
                        unit_val = sample_row[j]  # The 0-1 value for this gene

                        if info.get("scale") == "log":
                            log_min = math.log10(info["min"])
                            log_max = math.log10(info["max"])
                            scaled_log = log_min + unit_val * (
                                log_max - log_min
                            )
                            value = 10**scaled_log
                        else:
                            value = info["min"] + unit_val * (
                                info["max"] - info["min"]
                            )

                        individual[gene] = float(value)

                    lhs_individuals.append(individual)

                population.extend(lhs_individuals)

            else:
                pass

        existing_population = {str(ind): ind for ind in population}
        population = list(existing_population.values())

        # 3. Fill up to pop_size with random individuals (if needed)
        while len(population) < pop_size:
            individual = self._generate_individual()
            if str(individual) not in existing_population:
                population.append(individual)
                existing_population[str(individual)] = individual

        random.shuffle(population)

        return population[:pop_size]

    def _generate_individual(
        self,
        forced_gene: Optional[str] = None,
        forced_value: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Generate an individual, optionally forcing a value for a specific gene.
        """
        individual: Dict[str, Any] = {}
        for gene, info in self.chromosome_space.items():
            if forced_gene is not None and gene == forced_gene:
                individual[gene] = self._sample_gene_value(
                    info, forced_value=forced_value
                )
            else:
                individual[gene] = self._sample_gene_value(info)
        return individual

    @staticmethod
    def _sample_gene_value(
        info: Dict[str, Any],
        forced_value: Optional[Any] = None,
    ):
        """
        Helper for sampling a value for a gene.
        If forced_value is provided, use it.
        Otherwise, sample according to type.
        """

        if forced_value is not None:
            return forced_value

        if info["type"] in [DataType.DISCRETE, DataType.CATEGORICAL]:
            return random.choice(info["values"])
        elif info["type"] == DataType.CONTINUOUS:
            if info.get("scale") == "log":
                log_min = math.log10(info["min"])
                log_max = math.log10(info["max"])
                return float(10 ** random.uniform(log_min, log_max))
            else:
                return float(random.uniform(info["min"], info["max"]))
        raise ValueError(f"Unknown gene type: {info['type']}")


class DataType(Enum):
    CONTINUOUS = auto()
    DISCRETE = auto()
    CATEGORICAL = auto()


def get_chromosome_search_space(config: Dict[str, Any]) -> dict:
    """
    Create chromosome search space dictionary from config values and ranges.
    Args:
        config: The configuration dict from settings or elsewhere.
    Returns:
        A dictionary where keys are Chromosome field names and values describe
        type and allowed values/ranges for genetic algorithm.
    """
    nn_conf = config["neural_network_config"]["hyperparameter_space"]

    search_space = {}
    for gene_name, params in nn_conf.items():
        gene_info = {}
        if params["type"] == "float":
            gene_info["type"] = DataType.CONTINUOUS
            gene_info["min"] = params["range"][0]
            gene_info["max"] = params["range"][1]
            if "scale" in params:
                gene_info["scale"] = params["scale"]
        elif params["type"] == "int":
            gene_info["type"] = DataType.DISCRETE
        elif params["type"] == "enum":
            gene_info["type"] = (
                DataType.CATEGORICAL
                if gene_name
                in ("aug_intensity", "activation_fn", "optimizer_schedule")
                else DataType.DISCRETE
            )
            gene_info["values"] = params["values"]

        search_space[gene_name] = gene_info

    return search_space
