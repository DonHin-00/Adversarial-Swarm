"""
Advanced Genetic Operators for Population-Based Evolution

Implements sophisticated genetic algorithm techniques including crossover,
elitism, fitness-based selection, and adaptive mutation strategies.
"""

import logging
from hive_zero_core.security import SecureRandom

from typing import List, Tuple, Optional, Callable
import numpy as np

logger = logging.getLogger(__name__)


class Individual:
    """Represents a single individual in the population."""

    def __init__(self, genome: str, fitness: float = 0.0, generation: int = 0):
        """
        Initialize an individual.

        Args:
            genome: The code or payload string
            fitness: Fitness score (higher is better)
            generation: Generation number when created
        """
        self.genome = genome
        self.fitness = fitness
        self.generation = generation
        self.gene_seed = SecureRandom.random_int(0, 100000)
        self.parents = []  # Track lineage

    def __repr__(self):
        return (
            f"Individual(gen={self.generation}, fitness={self.fitness:.3f}, seed={self.gene_seed})"
        )


class FitnessFunction:
    """
    Evaluates fitness of individuals for genetic algorithm.
    Supports multiple fitness criteria.
    """

    def __init__(self, weights: Optional[dict] = None):
        """
        Initialize fitness function.

        Args:
            weights: Dictionary of fitness component weights
                     e.g., {'diversity': 0.3, 'validity': 0.5, 'evasion': 0.2}
        """
        self.weights = weights or {
            "diversity": 0.3,  # How different from original
            "validity": 0.5,  # Syntactic correctness
            "evasion": 0.2,  # Potential to evade detection
        }

    def evaluate(
        self, individual: Individual, original: str, validator: Callable[[str], bool]
    ) -> float:
        """
        Evaluate fitness of an individual.

        Args:
            individual: Individual to evaluate
            original: Original genome for comparison
            validator: Function to validate genome

        Returns:
            Fitness score (0.0 to 1.0)
        """
        scores = {}

        # Diversity: Levenshtein-like distance from original
        scores["diversity"] = self._calculate_diversity(individual.genome, original)

        # Validity: Can it compile/execute?
        scores["validity"] = 1.0 if validator(individual.genome) else 0.0

        # Evasion: Estimated evasion potential (length, entropy, uniqueness)
        scores["evasion"] = self._calculate_evasion_potential(individual.genome)

        # Weighted sum
        fitness = sum(scores[k] * self.weights[k] for k in scores)

        logger.debug(f"Fitness evaluation: {scores} → {fitness:.3f}")
        return fitness

    def _calculate_diversity(self, genome1: str, genome2: str) -> float:
        """Calculate diversity score between two genomes."""
        if not genome1 or not genome2:
            return 0.0

        # Simple character-level difference ratio
        max_len = max(len(genome1), len(genome2))
        if max_len == 0:
            return 0.0

        # Count differing characters
        differences = sum(1 for a, b in zip(genome1, genome2) if a != b)
        differences += abs(len(genome1) - len(genome2))

        return min(1.0, differences / max_len)

    def _calculate_evasion_potential(self, genome: str) -> float:
        """Estimate evasion potential based on entropy and characteristics."""
        if not genome:
            return 0.0

        # Calculate Shannon entropy
        if len(genome) < 2:
            entropy = 0.0
        else:
            counts = {}
            for char in genome:
                counts[char] = counts.get(char, 0) + 1

            entropy = 0.0
            total = len(genome)
            for count in counts.values():
                p = count / total
                entropy -= p * np.log2(p) if p > 0 else 0

            # Normalize entropy to 0-1 range (max entropy for ASCII is ~6.5 bits)
            entropy = min(1.0, entropy / 6.5)

        # Unique character ratio
        unique_ratio = len(set(genome)) / len(genome) if genome else 0.0

        # Combined score
        return entropy * 0.7 + unique_ratio * 0.3


class GeneticOperators:
    """
    Advanced genetic operators for evolution.
    Implements crossover, mutation, and selection strategies.
    """

    @staticmethod
    def crossover_single_point(
        parent1: Individual, parent2: Individual, generation: int
    ) -> Tuple[Individual, Individual]:
        """
        Single-point crossover between two parents.

        Args:
            parent1: First parent
            parent2: Second parent
            generation: Current generation number

        Returns:
            Two offspring individuals
        """
        g1, g2 = parent1.genome, parent2.genome

        if len(g1) < 2 or len(g2) < 2:
            # Can't crossover, return copies
            return Individual(g1, 0.0, generation), Individual(g2, 0.0, generation)

        # Choose crossover point
        min_len = min(len(g1), len(g2))
        point = SecureRandom.random_int(1, min_len - 1)

        # Create offspring
        offspring1_genome = g1[:point] + g2[point:]
        offspring2_genome = g2[:point] + g1[point:]

        offspring1 = Individual(offspring1_genome, 0.0, generation)
        offspring2 = Individual(offspring2_genome, 0.0, generation)

        # Track lineage
        offspring1.parents = [parent1.gene_seed, parent2.gene_seed]
        offspring2.parents = [parent2.gene_seed, parent1.gene_seed]

        logger.debug(
            f"Crossover at point {point}: {len(g1)}, {len(g2)} → {len(offspring1_genome)}, {len(offspring2_genome)}"
        )

        return offspring1, offspring2

    @staticmethod
    def crossover_uniform(
        parent1: Individual, parent2: Individual, generation: int, probability: float = 0.5
    ) -> Individual:
        """
        Uniform crossover: each character randomly chosen from either parent.

        Args:
            parent1: First parent
            parent2: Second parent
            generation: Current generation number
            probability: Probability of choosing from parent1 vs parent2

        Returns:
            Offspring individual
        """
        g1, g2 = parent1.genome, parent2.genome
        max_len = max(len(g1), len(g2))

        # Extend shorter genome with padding
        g1_padded = g1.ljust(max_len)
        g2_padded = g2.ljust(max_len)

        # Uniform selection
        offspring_genome = "".join(
            g1_padded[i] if SecureRandom.random_float() < probability else g2_padded[i]
            for i in range(max_len)
        ).rstrip()

        offspring = Individual(offspring_genome, 0.0, generation)
        offspring.parents = [parent1.gene_seed, parent2.gene_seed]

        logger.debug(f"Uniform crossover: {len(g1)}, {len(g2)} → {len(offspring_genome)}")

        return offspring

    @staticmethod
    def mutate_random_insertion(individual: Individual, gene_pool: List[str]) -> Individual:
        """
        Insert random genetic material from gene pool.

        Args:
            individual: Individual to mutate
            gene_pool: Pool of genetic material to insert

        Returns:
            Mutated individual
        """
        if not gene_pool or not individual.genome:
            return individual

        genome = individual.genome
        insertion = SecureRandom.random_choice(gene_pool)
        position = SecureRandom.random_int(0, len(genome))

        mutated_genome = genome[:position] + insertion + genome[position:]
        mutated = Individual(mutated_genome, 0.0, individual.generation)
        mutated.parents = [individual.gene_seed]

        logger.debug(f"Random insertion at {position}: '{insertion}'")

        return mutated

    @staticmethod
    def mutate_random_deletion(individual: Individual, max_delete: int = 5) -> Individual:
        """
        Delete random segment from genome.

        Args:
            individual: Individual to mutate
            max_delete: Maximum number of characters to delete

        Returns:
            Mutated individual
        """
        genome = individual.genome
        if len(genome) < 3:
            return individual

        delete_len = SecureRandom.random_int(1, min(max_delete, len(genome) // 2))
        position = SecureRandom.random_int(0, len(genome) - delete_len)

        mutated_genome = genome[:position] + genome[position + delete_len :]
        mutated = Individual(mutated_genome, 0.0, individual.generation)
        mutated.parents = [individual.gene_seed]

        logger.debug(f"Random deletion: {delete_len} chars at position {position}")

        return mutated

    @staticmethod
    def mutate_swap(individual: Individual) -> Individual:
        """
        Swap two random segments in genome.

        Args:
            individual: Individual to mutate

        Returns:
            Mutated individual
        """
        genome = list(individual.genome)
        if len(genome) < 4:
            return individual

        # Choose two random positions
        pos1 = SecureRandom.random_int(0, len(genome) - 2)
        pos2 = SecureRandom.random_int(pos1 + 1, len(genome) - 1)

        # Swap
        genome[pos1], genome[pos2] = genome[pos2], genome[pos1]

        mutated_genome = "".join(genome)
        mutated = Individual(mutated_genome, 0.0, individual.generation)
        mutated.parents = [individual.gene_seed]

        logger.debug(f"Swap mutation: positions {pos1} ↔ {pos2}")

        return mutated


class SelectionStrategies:
    """
    Selection strategies for choosing parents and survivors.
    """

    @staticmethod
    def tournament_selection(population: List[Individual], tournament_size: int = 3) -> Individual:
        """
        Tournament selection: randomly select k individuals, return best.

        Args:
            population: Current population
            tournament_size: Number of individuals in tournament

        Returns:
            Selected individual
        """
        if not population:
            raise ValueError("Population cannot be empty")

        tournament = SecureRandom.random_sample(population, min(tournament_size, len(population)))
        winner = max(tournament, key=lambda ind: ind.fitness)

        logger.debug(f"Tournament: selected individual with fitness {winner.fitness:.3f}")

        return winner

    @staticmethod
    def roulette_selection(population: List[Individual]) -> Individual:
        """
        Roulette wheel selection: probability proportional to fitness.

        Args:
            population: Current population

        Returns:
            Selected individual
        """
        if not population:
            raise ValueError("Population cannot be empty")

        total_fitness = sum(ind.fitness for ind in population)

        if total_fitness == 0:
            # All zero fitness, random selection
            return SecureRandom.random_choice(population)

        pick = SecureRandom.random_float() * (total_fitness - (0)) + (0)
        current = 0

        for individual in population:
            current += individual.fitness
            if current >= pick:
                logger.debug(f"Roulette: selected individual with fitness {individual.fitness:.3f}")
                return individual

        # Fallback
        return population[-1]

    @staticmethod
    def elitism_selection(population: List[Individual], elite_size: int = 2) -> List[Individual]:
        """
        Elitism: select top-k individuals.

        Args:
            population: Current population
            elite_size: Number of elite individuals to select

        Returns:
            List of elite individuals
        """
        if not population:
            return []

        sorted_pop = sorted(population, key=lambda ind: ind.fitness, reverse=True)
        elites = sorted_pop[:elite_size]

        logger.debug(f"Elitism: selected {len(elites)} individuals")

        return elites
