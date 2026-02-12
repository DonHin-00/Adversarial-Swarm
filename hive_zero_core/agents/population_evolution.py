"""
Population-Based Genetic Algorithm Manager

Implements advanced genetic algorithm with population management, adaptive mutation,
and multi-generational evolution for payload and code optimization.
"""

import logging
from hive_zero_core.security import SecureRandom, InputValidator, AuditLogger, AccessController
from hive_zero_core.security.audit_logger import SecurityEvent
from hive_zero_core.security.access_control import OperationType

import random
from typing import List, Optional, Callable, Dict
import numpy as np

from hive_zero_core.agents.genetic_operators import (
    Individual, FitnessFunction, GeneticOperators, SelectionStrategies
)


logger = logging.getLogger(__name__)


class PopulationManager:
    """
    Manages a population of individuals through multiple generations.
    Implements full genetic algorithm with crossover, mutation, and selection.
    """

    def __init__(self,
                 population_size: int = 20,
                 elite_size: int = 2,
                 mutation_rate: float = 0.3,
                 crossover_rate: float = 0.7,
                 max_generations: int = 50,
                 fitness_weights: Optional[dict] = None):
        """
        Initialize population manager.

        Args:
            population_size: Number of individuals in population
            elite_size: Number of elite individuals to preserve
            mutation_rate: Probability of mutation
            crossover_rate: Probability of crossover
            max_generations: Maximum number of generations
            fitness_weights: Weights for fitness components
        """
        if population_size < 4:
            raise ValueError(f"population_size must be >= 4, got {population_size}")
        if elite_size >= population_size:
            raise ValueError(f"elite_size must be < population_size")
        if not 0.0 <= mutation_rate <= 1.0:
            raise ValueError(f"mutation_rate must be in [0,1], got {mutation_rate}")
        if not 0.0 <= crossover_rate <= 1.0:
            raise ValueError(f"crossover_rate must be in [0,1], got {crossover_rate}")

        self.population_size = population_size
        self.elite_size = elite_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.max_generations = max_generations

        self.fitness_function = FitnessFunction(weights=fitness_weights)
        self.operators = GeneticOperators()
        self.selection = SelectionStrategies()

        self.population: List[Individual] = []
        self.current_generation = 0
        self.best_individual: Optional[Individual] = None
        self.history: List[Dict] = []

        # Adaptive parameters
        self.adaptive_mutation = True
        self.stagnation_counter = 0
        self.last_best_fitness = 0.0

        # Gene pool for diversity
        self.gene_pool: List[str] = self._initialize_gene_pool()

    def _initialize_gene_pool(self) -> List[str]:
        """Initialize gene pool with useful genetic material."""
        return [
            "# NOP",
            "pass",
            "_x = 0",
            "  ",
            "\n",
            "'''docstring'''",
            "__unused__ = None",
            "# fmt: skip",
            "# type: ignore",
            "# noqa",
        ]

    def initialize_population(self, seed_genome: str,
                            validator: Callable[[str], bool]) -> None:
        """
        Initialize population from seed genome.

        Args:
            seed_genome: Initial genome to derive population from
            validator: Function to validate genomes
        """
        from hive_zero_core.agents.genetic_evolution import PolymorphicEngine

        engine = PolymorphicEngine()
        self.population = []

        # Add seed genome as first individual
        seed_individual = Individual(seed_genome, 0.0, 0)
        seed_individual.fitness = self.fitness_function.evaluate(
            seed_individual, seed_genome, validator
        )
        self.population.append(seed_individual)

        # Generate diverse initial population
        attempts = 0
        max_attempts = self.population_size * 10

        while len(self.population) < self.population_size and attempts < max_attempts:
            attempts += 1
            gene_seed = random.randint(0, 100000)

            try:
                # Mutate seed with varying mutation rates
                varied_rate = random.uniform(0.2, 0.8)
                mutated_genome = engine.mutate_code(seed_genome, gene_seed, varied_rate)

                if validator(mutated_genome):
                    individual = Individual(mutated_genome, 0.0, 0)
                    individual.gene_seed = gene_seed
                    individual.fitness = self.fitness_function.evaluate(
                        individual, seed_genome, validator
                    )
                    self.population.append(individual)

            except Exception as e:
                logger.debug(f"Failed to create individual: {e}")
                continue

        # Fill remaining slots with random variations if needed
        while len(self.population) < self.population_size:
            # Add slight variations
            base = random.choice(self.population[:len(self.population)//2 + 1])
            individual = Individual(base.genome + f"\n# GEN_0_{len(self.population)}", 0.0, 0)
            if validator(individual.genome):
                individual.fitness = self.fitness_function.evaluate(
                    individual, seed_genome, validator
                )
                self.population.append(individual)
            else:
                # Just clone if variation fails
                self.population.append(Individual(base.genome, base.fitness, 0))

        self.best_individual = max(self.population, key=lambda ind: ind.fitness)

        logger.info(f"Initialized population of {len(self.population)} individuals")
        logger.info(f"Best initial fitness: {self.best_individual.fitness:.3f}")

    def evolve(self, seed_genome: str, validator: Callable[[str], bool],
              generations: Optional[int] = None) -> Individual:
        """
        Evolve population over multiple generations.

        Args:
            seed_genome: Original genome for comparison
            validator: Function to validate genomes
            generations: Number of generations (default: max_generations)

        Returns:
            Best individual found
        """
        if not self.population:
            self.initialize_population(seed_genome, validator)

        generations = generations or self.max_generations

        for gen in range(generations):
            self.current_generation = gen + 1

            # Create next generation
            next_population = self._create_next_generation(seed_genome, validator)

            # Update population
            self.population = next_population

            # Track best individual
            current_best = max(self.population, key=lambda ind: ind.fitness)
            if current_best.fitness > self.best_individual.fitness:
                self.best_individual = current_best
                self.stagnation_counter = 0
                logger.info(f"Gen {self.current_generation}: New best fitness {current_best.fitness:.3f}")
            else:
                self.stagnation_counter += 1

            # Adaptive mutation rate
            if self.adaptive_mutation:
                self._adapt_mutation_rate()

            # Record history
            self._record_generation_stats()

            # Early stopping if converged
            if self._check_convergence():
                logger.info(f"Converged at generation {self.current_generation}")
                break

        logger.info(f"Evolution complete. Best fitness: {self.best_individual.fitness:.3f}")
        return self.best_individual

    def _create_next_generation(self, seed_genome: str,
                               validator: Callable[[str], bool]) -> List[Individual]:
        """Create next generation through selection, crossover, and mutation."""
        next_gen = []

        # Elitism: preserve best individuals
        elites = self.selection.elitism_selection(self.population, self.elite_size)
        next_gen.extend([Individual(e.genome, e.fitness, self.current_generation + 1)
                        for e in elites])

        # Generate offspring
        while len(next_gen) < self.population_size:
            # Selection
            if random.random() < 0.7:  # Tournament selection 70% of the time
                parent1 = self.selection.tournament_selection(self.population, 3)
                parent2 = self.selection.tournament_selection(self.population, 3)
            else:  # Roulette selection 30% of the time
                parent1 = self.selection.roulette_selection(self.population)
                parent2 = self.selection.roulette_selection(self.population)

            # Crossover
            if random.random() < self.crossover_rate:
                if random.random() < 0.6:  # Single-point 60%
                    offspring1, offspring2 = self.operators.crossover_single_point(
                        parent1, parent2, self.current_generation + 1
                    )
                else:  # Uniform 40%
                    offspring1 = self.operators.crossover_uniform(
                        parent1, parent2, self.current_generation + 1
                    )
                    offspring2 = Individual(parent2.genome, 0.0, self.current_generation + 1)
            else:
                # No crossover, clone parents
                offspring1 = Individual(parent1.genome, 0.0, self.current_generation + 1)
                offspring2 = Individual(parent2.genome, 0.0, self.current_generation + 1)

            # Mutation
            offspring1 = self._apply_mutations(offspring1)
            offspring2 = self._apply_mutations(offspring2)

            # Validate and evaluate
            for idx, offspring in enumerate([offspring1, offspring2]):
                if len(next_gen) >= self.population_size:
                    break

                if validator(offspring.genome):
                    offspring.fitness = self.fitness_function.evaluate(
                        offspring, seed_genome, validator
                    )
                    next_gen.append(offspring)
                else:
                    # Clone the corresponding parent instead of always using parent1
                    fallback_parent = parent1 if idx == 0 else parent2
                    logger.warning(f"Offspring {idx+1} failed validation, cloning corresponding parent")
                    if len(next_gen) < self.population_size:
                        parent_copy = Individual(fallback_parent.genome, fallback_parent.fitness,
                                                self.current_generation + 1)
                        next_gen.append(parent_copy)

        return next_gen[:self.population_size]

    def _apply_mutations(self, individual: Individual) -> Individual:
        """Apply various mutations to an individual."""
        if random.random() > self.mutation_rate:
            return individual

        # Choose mutation type
        mutation_type = random.choice(['insertion', 'deletion', 'swap', 'polymorphic'])

        try:
            if mutation_type == 'insertion':
                individual = self.operators.mutate_random_insertion(individual, self.gene_pool)
            elif mutation_type == 'deletion':
                individual = self.operators.mutate_random_deletion(individual, max_delete=3)
            elif mutation_type == 'swap':
                individual = self.operators.mutate_swap(individual)
            elif mutation_type == 'polymorphic':
                # Apply polymorphic mutation
                from hive_zero_core.agents.genetic_evolution import PolymorphicEngine
                engine = PolymorphicEngine()
                mutated_genome = engine.mutate_code(individual.genome,
                                                   random.randint(0, 100000),
                                                   self.mutation_rate)
                individual = Individual(mutated_genome, 0.0, individual.generation)

        except Exception as e:
            logger.debug(f"Mutation failed: {e}")

        return individual

    def _adapt_mutation_rate(self):
        """Adapt mutation rate based on population progress."""
        if self.stagnation_counter > 5:
            # Increase mutation for exploration
            self.mutation_rate = min(0.8, self.mutation_rate * 1.2)
            logger.debug(f"Increased mutation rate to {self.mutation_rate:.2f}")
        elif self.stagnation_counter == 0:
            # Decrease mutation for exploitation
            self.mutation_rate = max(0.1, self.mutation_rate * 0.9)
            logger.debug(f"Decreased mutation rate to {self.mutation_rate:.2f}")

    def _check_convergence(self) -> bool:
        """Check if population has converged."""
        if self.stagnation_counter > 15:
            return True

        # Check diversity
        unique_genomes = len(set(ind.genome for ind in self.population))
        diversity_ratio = unique_genomes / len(self.population)

        if diversity_ratio < 0.1:  # Less than 10% unique
            logger.info(f"Low diversity detected: {diversity_ratio:.1%}")
            return True

        return False

    def _record_generation_stats(self):
        """Record statistics for current generation."""
        fitnesses = [ind.fitness for ind in self.population]

        stats = {
            'generation': self.current_generation,
            'best_fitness': max(fitnesses),
            'avg_fitness': np.mean(fitnesses),
            'worst_fitness': min(fitnesses),
            'fitness_std': np.std(fitnesses),
            'unique_genomes': len(set(ind.genome for ind in self.population)),
            'mutation_rate': self.mutation_rate,
        }

        self.history.append(stats)

        logger.debug(f"Gen {self.current_generation}: avg={stats['avg_fitness']:.3f}, "
                    f"best={stats['best_fitness']:.3f}, diversity={stats['unique_genomes']}")

    def get_statistics(self) -> Dict:
        """Get comprehensive statistics about evolution process."""
        if not self.history:
            return {'status': 'not_started'}

        return {
            'total_generations': len(self.history),
            'final_best_fitness': self.best_individual.fitness if self.best_individual else 0.0,
            'final_avg_fitness': self.history[-1]['avg_fitness'],
            'convergence_generation': self._find_convergence_generation(),
            'fitness_improvement': (
                self.history[-1]['best_fitness'] - self.history[0]['best_fitness']
                if len(self.history) > 0 else 0.0
            ),
            'final_diversity': self.history[-1]['unique_genomes'],
            'history': self.history,
        }

    def _find_convergence_generation(self) -> Optional[int]:
        """Find generation where fitness plateaued."""
        if len(self.history) < 5:
            return None

        # Find first generation where improvement stopped
        for i in range(len(self.history) - 5):
            recent_best = [h['best_fitness'] for h in self.history[i:i+5]]
            if max(recent_best) - min(recent_best) < 0.01:
                return i

        return None

    def get_best_individuals(self, n: int = 5) -> List[Individual]:
        """Get top-n individuals from population."""
        sorted_pop = sorted(self.population, key=lambda ind: ind.fitness, reverse=True)
        return sorted_pop[:n]
