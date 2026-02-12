"""
Genetic Evolution Engine for Red Team Capabilities

Implements polymorphic code mutation and natural selection for payload evasion.
Inspired by self-modifying malware techniques adapted for legitimate red team operations.
"""

import ast
import logging
import random
from typing import Optional, Tuple


logger = logging.getLogger(__name__)


class PolymorphicEngine:
    """
    Polymorphic Engine: Injects random safe NOPs based on genetic seed.
    Changes code signature without breaking logic (Strong & Silent).
    """

    @staticmethod
    def mutate_code(source_code: str, gene_seed: int, mutation_rate: float = 0.5) -> str:
        """
        Inject genetic markers (junk variables) into source code.

        Args:
            source_code: Python source code as string
            gene_seed: Random seed for reproducible mutations
            mutation_rate: Probability of mutation per eligible line (default: 0.5)

        Returns:
            Mutated source code with same functionality but different signature

        Raises:
            ValueError: If source_code is empty or mutation_rate is invalid
        """
        # Input validation
        if not source_code or not source_code.strip():
            raise ValueError("Source code cannot be empty")
        if not 0.0 <= mutation_rate <= 1.0:
            raise ValueError(f"mutation_rate must be between 0 and 1, got {mutation_rate}")

        try:
            random.seed(gene_seed)
            lines = source_code.split('\n')
            new_lines = []
            mutations_applied = 0

            # Genetic Markers (Junk Variables) - more variety
            junk_vars = [f"_gene_{random.randint(1000, 9999)}" for _ in range(10)]
            junk_types = [
                lambda: f"{random.choice(junk_vars)} = {random.randint(0, 1000)} # GENE: {gene_seed}",
                lambda: f"{random.choice(junk_vars)} = '{chr(random.randint(97, 122))}' * 0 # NOP",
                lambda: f"pass  # GEN_{gene_seed}",
            ]

            for line in lines:
                new_lines.append(line)
                # Inject junk after function definitions, class definitions, or imports
                line_stripped = line.strip()
                if (line_stripped.startswith("def ") or
                    line_stripped.startswith("import ") or
                    line_stripped.startswith("from ") or
                    line_stripped.startswith("class ")):

                    if random.random() < mutation_rate:
                        # Get indentation from original line
                        indent = len(line) - len(line.lstrip())
                        junk_line = " " * (indent + 4) + random.choice(junk_types)()
                        new_lines.append(junk_line)
                        mutations_applied += 1

            # Ensure at least one mutation was applied
            if mutations_applied == 0 and len(lines) > 0:
                # Force at least one mutation at the beginning
                new_lines.insert(1 if len(new_lines) > 1 else 0,
                                f"# GENETIC_MARKER_{gene_seed}")
                mutations_applied = 1

            result = '\n'.join(new_lines)
            logger.debug(f"Applied {mutations_applied} mutations with seed {gene_seed}")
            return result

        except Exception as e:
            logger.error(f"Error during code mutation: {e}")
            # Return original code with minimal mutation as fallback
            return f"# GENE:{gene_seed}\n{source_code}"

    @staticmethod
    def mutate_string(payload: str, gene_seed: int, min_mutations: int = 1) -> str:
        """
        Mutate a string payload by injecting harmless padding/encoding changes.

        Args:
            payload: String payload to mutate
            gene_seed: Random seed for reproducible mutations
            min_mutations: Minimum number of mutations to apply (default: 1)

        Returns:
            Mutated payload with different signature

        Raises:
            ValueError: If payload is empty or min_mutations is negative
        """
        # Input validation
        if not payload:
            raise ValueError("Payload cannot be empty")
        if min_mutations < 0:
            raise ValueError(f"min_mutations must be non-negative, got {min_mutations}")

        try:
            random.seed(gene_seed)
            mutated = payload

            # Apply various string mutation techniques (more robust variations)
            mutations = [
                lambda s: s + f"#{gene_seed}",  # Comment padding
                lambda s: f"/* GEN:{gene_seed} */{s}",  # C-style comment
                lambda s: f"<!-- {gene_seed} -->{s}",  # HTML comment
                lambda s: s.replace(" ", "  " if random.random() > 0.5 else " "),  # Whitespace variation
                lambda s: s + "\x00" * random.randint(1, 2),  # Minimal null byte padding
                lambda s: f"{s}\n-- GENE:{gene_seed}",  # SQL comment style
                lambda s: s + " " * random.randint(1, 3),  # Trailing spaces
            ]

            # Apply mutations
            num_mutations = max(min_mutations, random.randint(min_mutations, min(3, len(mutations))))
            selected_mutations = random.sample(mutations, num_mutations)

            for mutation_func in selected_mutations:
                try:
                    mutated = mutation_func(mutated)
                except Exception as e:
                    logger.warning(f"Mutation function failed: {e}, continuing...")
                    continue

            # Ensure we actually mutated something
            if mutated == payload:
                mutated = payload + f"#{gene_seed}"

            logger.debug(f"Applied {num_mutations} string mutations with seed {gene_seed}")
            return mutated

        except Exception as e:
            logger.error(f"Error during string mutation: {e}")
            # Fallback to minimal mutation
            return payload + f"#{gene_seed}"


class NaturalSelection:
    """
    Natural Selection: Ensures mutated code/payloads remain valid.
    No 'bad genes' allowed.
    """

    @staticmethod
    def validate_python(source_code: str, strict: bool = False) -> bool:
        """
        Validate that mutated Python code is syntactically correct.

        Args:
            source_code: Python code to validate
            strict: If True, also check for empty code and basic structure

        Returns:
            True if code compiles successfully, False otherwise
        """
        if not source_code or not source_code.strip():
            return False

        try:
            # Attempt compilation
            compile(source_code, '<string>', 'exec')

            # Strict mode: additional checks
            if strict:
                # Check for at least some meaningful content
                non_comment_lines = [
                    line for line in source_code.split('\n')
                    if line.strip() and not line.strip().startswith('#')
                ]
                if len(non_comment_lines) == 0:
                    return False

            return True
        except (SyntaxError, ValueError, TypeError) as e:
            logger.debug(f"Python validation failed: {e}")
            return False
        except Exception as e:
            logger.warning(f"Unexpected error during Python validation: {e}")
            return False

    @staticmethod
    def validate_ast(source_code: str) -> Tuple[bool, Optional[ast.AST]]:
        """
        Validate Python code and return its AST if valid.

        Args:
            source_code: Python code to validate

        Returns:
            Tuple of (is_valid, ast_tree or None)
        """
        if not source_code or not source_code.strip():
            return False, None

        try:
            tree = ast.parse(source_code)
            return True, tree
        except (SyntaxError, ValueError) as e:
            logger.debug(f"AST validation failed: {e}")
            return False, None
        except Exception as e:
            logger.warning(f"Unexpected error during AST validation: {e}")
            return False, None

    @staticmethod
    def validate_payload(payload: str, max_length: int = 10000,
                        max_null_bytes: int = 5,
                        allow_empty: bool = False) -> bool:
        """
        Validate that a payload meets basic safety constraints.

        Args:
            payload: String payload to validate
            max_length: Maximum allowed length
            max_null_bytes: Maximum allowed null bytes
            allow_empty: Whether to allow empty payloads

        Returns:
            True if payload is valid, False otherwise
        """
        # Check for empty payload
        if not payload:
            return allow_empty

        try:
            # Check basic constraints
            if len(payload) > max_length:
                logger.debug(f"Payload too long: {len(payload)} > {max_length}")
                return False

            # Check for excessive null termination
            null_count = payload.count('\x00')
            if null_count > max_null_bytes:
                logger.debug(f"Too many null bytes: {null_count} > {max_null_bytes}")
                return False

            # Check for reasonable character distribution
            # (avoid payloads that are mostly null bytes or single character)
            if len(payload) > 10:
                unique_chars = len(set(payload))
                if unique_chars < 3 and null_count == 0:
                    logger.debug(f"Insufficient character diversity: {unique_chars} unique chars")
                    return False

            return True

        except Exception as e:
            logger.warning(f"Error during payload validation: {e}")
            return False

class GenerationTracker:
    """
    Tracks evolution generations for payloads and code mutations.
    """

    def __init__(self):
        self.current_generation = 0
        self.mutation_history = []
        self.max_history_size = 1000  # Prevent unbounded growth

    def increment_generation(self, gene_seed: int, success: bool, metadata: Optional[dict] = None):
        """
        Record a mutation event.

        Args:
            gene_seed: The seed used for this mutation
            success: Whether the mutation was successful
            metadata: Optional additional metadata about the mutation
        """
        self.current_generation += 1

        entry = {
            'generation': self.current_generation,
            'gene_seed': gene_seed,
            'success': success
        }
        if metadata:
            entry['metadata'] = metadata

        self.mutation_history.append(entry)

        # Trim history if it gets too large
        if len(self.mutation_history) > self.max_history_size:
            self.mutation_history = self.mutation_history[-self.max_history_size:]

    def get_generation(self) -> int:
        """Get current generation number."""
        return self.current_generation

    def get_mutation_stats(self) -> dict:
        """Get statistics about mutation history."""
        if not self.mutation_history:
            return {
                'total': 0,
                'successful': 0,
                'success_rate': 0.0,
                'current_generation': self.current_generation
            }

        total = len(self.mutation_history)
        successful = sum(1 for m in self.mutation_history if m['success'])

        # Calculate recent success rate (last 10 mutations)
        recent_history = self.mutation_history[-10:]
        recent_successful = sum(1 for m in recent_history if m['success'])
        recent_rate = recent_successful / len(recent_history) if recent_history else 0.0

        return {
            'total': total,
            'successful': successful,
            'success_rate': successful / total if total > 0 else 0.0,
            'recent_success_rate': recent_rate,
            'current_generation': self.current_generation
        }

    def reset(self):
        """Reset tracking state."""
        self.current_generation = 0
        self.mutation_history.clear()


class GeneticEvolution:
    """
    Main interface for genetic evolution capabilities.
    Combines polymorphic engine, natural selection, and generation tracking.
    """

    def __init__(self, max_generations: int = 100, mutation_rate: float = 0.5):
        """
        Initialize the genetic evolution system.

        Args:
            max_generations: Maximum number of generations to track
            mutation_rate: Base mutation rate for code mutations (0.0 to 1.0)

        Raises:
            ValueError: If parameters are out of valid range
        """
        if max_generations < 1:
            raise ValueError(f"max_generations must be positive, got {max_generations}")
        if not 0.0 <= mutation_rate <= 1.0:
            raise ValueError(f"mutation_rate must be between 0 and 1, got {mutation_rate}")

        self.engine = PolymorphicEngine()
        self.selector = NaturalSelection()
        self.tracker = GenerationTracker()
        self.max_generations = max_generations
        self.mutation_rate = mutation_rate

    def evolve_code(self, source_code: str, max_attempts: int = 10,
                   strict_validation: bool = False) -> Tuple[str, int, bool]:
        """
        Evolve Python source code through multiple generations.

        Args:
            source_code: Original Python source code
            max_attempts: Maximum mutation attempts
            strict_validation: If True, apply stricter validation rules

        Returns:
            Tuple of (mutated_code, gene_seed, success)

        Raises:
            ValueError: If source_code is empty or max_attempts is invalid
        """
        # Input validation
        if not source_code or not source_code.strip():
            raise ValueError("Source code cannot be empty")
        if max_attempts < 1:
            raise ValueError(f"max_attempts must be positive, got {max_attempts}")

        best_mutated = None
        best_seed = 0

        try:
            for attempt in range(max_attempts):
                gene_seed = random.randint(0, 100000)

                try:
                    mutated = self.engine.mutate_code(source_code, gene_seed, self.mutation_rate)
                except Exception as e:
                    logger.warning(f"Mutation attempt {attempt} failed: {e}")
                    self.tracker.increment_generation(gene_seed, False, {'error': str(e)})
                    continue

                # Validate the mutated code
                is_valid = self.selector.validate_python(mutated, strict=strict_validation)

                if is_valid:
                    self.tracker.increment_generation(gene_seed, True)
                    return mutated, gene_seed, True
                else:
                    self.tracker.increment_generation(gene_seed, False)
                    # Keep first valid mutation as backup
                    if best_mutated is None and self.selector.validate_python(mutated, strict=False):
                        best_mutated = mutated
                        best_seed = gene_seed

        except Exception as e:
            logger.error(f"Critical error during code evolution: {e}")

        # If we have a backup mutation (valid but not strict), return it
        if best_mutated is not None:
            logger.info(f"Returning backup mutation (seed={best_seed})")
            return best_mutated, best_seed, True

        # If all mutations failed, return original
        logger.warning("All mutation attempts failed, returning original code")
        return source_code, 0, False

    def evolve_payload(self, payload: str, max_attempts: int = 10,
                      max_length: int = 10000) -> Tuple[str, int, bool]:
        """
        Evolve a string payload through multiple generations.

        Args:
            payload: Original payload string
            max_attempts: Maximum mutation attempts
            max_length: Maximum allowed payload length

        Returns:
            Tuple of (mutated_payload, gene_seed, success)

        Raises:
            ValueError: If payload is empty or max_attempts is invalid
        """
        # Input validation
        if not payload:
            raise ValueError("Payload cannot be empty")
        if max_attempts < 1:
            raise ValueError(f"max_attempts must be positive, got {max_attempts}")

        try:
            for attempt in range(max_attempts):
                gene_seed = random.randint(0, 100000)

                try:
                    mutated = self.engine.mutate_string(payload, gene_seed, min_mutations=1)
                except Exception as e:
                    logger.warning(f"Mutation attempt {attempt} failed: {e}")
                    self.tracker.increment_generation(gene_seed, False, {'error': str(e)})
                    continue

                # Validate the mutated payload
                is_valid = self.selector.validate_payload(mutated, max_length=max_length)

                if is_valid:
                    self.tracker.increment_generation(gene_seed, True)
                    return mutated, gene_seed, True
                else:
                    self.tracker.increment_generation(gene_seed, False)

        except Exception as e:
            logger.error(f"Critical error during payload evolution: {e}")

        # If all mutations failed, apply minimal fallback mutation
        try:
            fallback_seed = random.randint(0, 100000)
            fallback_mutated = payload + f"#{fallback_seed}"
            if self.selector.validate_payload(fallback_mutated, max_length=max_length):
                logger.info("Using fallback mutation")
                self.tracker.increment_generation(fallback_seed, True, {'fallback': True})
                return fallback_mutated, fallback_seed, True
        except Exception:
            pass

        # Absolute fallback: return original
        logger.warning("All mutation attempts failed, returning original payload")
        return payload, 0, False

    def get_stats(self) -> dict:
        """Get evolution statistics."""
        return self.tracker.get_mutation_stats()

    def reset(self):
        """Reset evolution state."""
        self.tracker.reset()
