from enum import Enum, auto
import numpy as np
import random
from copy import deepcopy
from typing import List, Dict, Any


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
        self._is_random_mode = self.config.get("active") == ["random"]

        if not self._is_random_mode:
            self.active_operators = set(self.config.get("active", []))
        else:
            self.active_operators = set()  # Will be chosen per generation

    def tournament_selection(
        self, population: List[Any], fitness: List[float]
    ) -> Any:
        tournament_size = self.config["selection"]["tournament_size"]
        selected_indices = np.random.choice(
            len(population), tournament_size, replace=False
        )
        best_idx = selected_indices[
            np.argmax([fitness[i] for i in selected_indices])
        ]
        return deepcopy(population[best_idx])

    def uniform_crossover(self, parent1: Dict, parent2: Dict) -> Dict:
        child = {}
        crossover_prob: int = self.config["crossover"]["crossover_prob"]
        for k in parent1:
            child[k] = (
                parent1[k] if np.random.rand() < crossover_prob else parent2[k]
            )
        return child

    def mutate(self, chromosome: Dict) -> Dict:
        mutation_conf = self.config["mutation"]
        for gene, info in self.chromosome_space.items():
            if info["type"] == DataType.CONTINUOUS:
                if np.random.rand() < mutation_conf["mutation_prob_continuous"]:
                    sigma = mutation_conf["mutation_sigma_continuous"]
                    if info.get("scale") == "log":
                        log_val = np.log10(chromosome[gene])
                        mutated_log = log_val + np.random.normal(0, sigma)
                        mutated = 10**mutated_log
                    else:
                        mutated = chromosome[gene] + np.random.normal(0, sigma)
                    chromosome[gene] = float(
                        np.clip(mutated, info["min"], info["max"])
                    )
            else:
                prob = mutation_conf.get("mutation_prob_discrete")
                if info["type"] == DataType.CATEGORICAL:
                    prob = mutation_conf.get("mutation_prob_categorical")

                if np.random.rand() < prob:
                    possible = [
                        v for v in info["values"] if v != chromosome[gene]
                    ]
                    if possible:
                        chromosome[gene] = random.choice(possible)
        return chromosome

    def elitism(self, population: List[Any], fitness: List[float]) -> List[Any]:
        percent = self.config["elitism_percent"]
        elite_num = max(1, int(len(population) * percent))
        elite_indices = np.argsort(fitness)[-elite_num:]
        return [deepcopy(population[i]) for i in elite_indices]

    def run_generation(
        self, population: List[Dict], fitness: List[float]
    ) -> List[Dict]:

        pop_size = len(population)
        new_pop = []

        if self._is_random_mode:
            numer_of_ga_operators = 2
            active_ops = set(
                random.sample(
                    list(self.available_operators), numer_of_ga_operators
                )
            )
            # print(f"Random mode: running with {active_ops}")
        else:
            active_ops = self.active_operators

        # Always preserve the single best individual to avoid losing progress
        best_overall_idx = np.argmax(fitness)
        new_pop.append(deepcopy(population[best_overall_idx]))

        if "elitism" in active_ops:
            # Avoid adding the best individual twice
            elite = self.elitism(population, fitness)
            for ind in elite:
                if len(new_pop) < pop_size and ind not in new_pop:
                    new_pop.append(ind)

        # Main loop
        while len(new_pop) < pop_size:
            if "crossover" in active_ops:
                # If crossover is active, we need two parents
                if "selection" in active_ops:
                    # If selection is also active, use tournament selection.
                    parent1 = self.tournament_selection(population, fitness)
                    parent2 = self.tournament_selection(population, fitness)
                else:
                    # If crossover is active without selection, pick parents randomly
                    parents = random.sample(population, 2)
                    parent1 = deepcopy(parents[0])
                    parent2 = deepcopy(parents[1])

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

    # TODO: now it generators random pop_size based on config. In future this function would go calibration,
    # and result will be top N main algorithm population of M calibration population, where N <= M
    def initial_population(self, pop_size: int) -> List[Dict]:
        population = []
        for _ in range(pop_size):
            indiv = {}
            for gene, info in self.chromosome_space.items():
                if info["type"] == DataType.DISCRETE:
                    indiv[gene] = random.choice(info["values"])
                elif info["type"] == DataType.CATEGORICAL:
                    indiv[gene] = random.choice(info["values"])
                elif info["type"] == DataType.CONTINUOUS:
                    if info.get("scale") == "log":
                        log_min = np.log10(info["min"])
                        log_max = np.log10(info["max"])
                        indiv[gene] = float(
                            10 ** np.random.uniform(log_min, log_max)
                        )
                    else:
                        indiv[gene] = float(
                            np.random.uniform(info["min"], info["max"])
                        )
            population.append(indiv)
        return population


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
                if gene_name in ("aug_intensity", "optimizer_schedule")
                else DataType.DISCRETE
            )
            gene_info["values"] = params["values"]

        search_space[gene_name] = gene_info

    return search_space
