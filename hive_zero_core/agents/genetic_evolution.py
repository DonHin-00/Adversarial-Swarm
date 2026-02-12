"""
Genetic Evolution Engine for Red Team Capabilities

Implements polymorphic code mutation and natural selection for payload evasion.
Inspired by self-modifying malware techniques adapted for legitimate red team operations.
"""

import ast
import random
from typing import Optional, Tuple


class PolymorphicEngine:
    """
    Polymorphic Engine: Injects random safe NOPs based on genetic seed.
    Changes code signature without breaking logic (Strong & Silent).
    """

    @staticmethod
    def mutate_code(source_code: str, gene_seed: int) -> str:
        """
        Inject genetic markers (junk variables) into source code.

        Args:
            source_code: Python source code as string
            gene_seed: Random seed for reproducible mutations

        Returns:
            Mutated source code with same functionality but different signature
        """
        random.seed(gene_seed)
        lines = source_code.split('\n')
        new_lines = []

        # Genetic Markers (Junk Variables)
        junk_vars = [f"_gene_{random.randint(1000, 9999)}" for _ in range(5)]

        for line in lines:
            new_lines.append(line)
            # Inject junk after function definitions or imports
            if line.strip().startswith("def ") or line.strip().startswith("import "):
                if random.random() < 0.3:
                    junk = f"    {random.choice(junk_vars)} = {random.randint(0, 1000)} # GENE: {gene_seed}"
                    new_lines.append(junk)

        return '\n'.join(new_lines)

    @staticmethod
    def mutate_string(payload: str, gene_seed: int) -> str:
        """
        Mutate a string payload by injecting harmless padding/encoding changes.

        Args:
            payload: String payload to mutate
            gene_seed: Random seed for reproducible mutations

        Returns:
            Mutated payload with different signature
        """
        random.seed(gene_seed)
        mutated = payload

        # Apply various string mutation techniques
        mutations = [
            lambda s: s + f"#{gene_seed}",  # Comment padding
            lambda s: f"/* GEN:{gene_seed} */{s}",  # C-style comment
            lambda s: s.replace(" ", "  " if random.random() > 0.5 else " "),  # Whitespace variation
            lambda s: s + "\x00" * random.randint(1, 3),  # Null byte padding
        ]

        # Apply 1-3 random mutations
        for _ in range(random.randint(1, 3)):
            mutation = random.choice(mutations)
            try:
                mutated = mutation(mutated)
            except Exception:
                pass  # Skip if mutation fails

        return mutated


class NaturalSelection:
    """
    Natural Selection: Ensures mutated code/payloads remain valid.
    No 'bad genes' allowed.
    """

    @staticmethod
    def validate_python(source_code: str) -> bool:
        """
        Validate that mutated Python code is syntactically correct.

        Args:
            source_code: Python code to validate

        Returns:
            True if code compiles successfully, False otherwise
        """
        try:
            compile(source_code, '<string>', 'exec')
            return True
        except SyntaxError:
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
        try:
            tree = ast.parse(source_code)
            return True, tree
        except SyntaxError:
            return False, None

    @staticmethod
    def validate_payload(payload: str, max_length: int = 10000) -> bool:
        """
        Validate that a payload meets basic safety constraints.

        Args:
            payload: String payload to validate
            max_length: Maximum allowed length

        Returns:
            True if payload is valid, False otherwise
        """
        # Check basic constraints
        if len(payload) > max_length:
            return False

        # Check for null termination issues
        if payload.count('\x00') > 10:
            return False

        return True


class GenerationTracker:
    """
    Tracks evolution generations for payloads and code mutations.
    """

    def __init__(self):
        self.current_generation = 0
        self.mutation_history = []

    def increment_generation(self, gene_seed: int, success: bool):
        """Record a mutation event."""
        self.current_generation += 1
        self.mutation_history.append({
            'generation': self.current_generation,
            'gene_seed': gene_seed,
            'success': success
        })

    def get_generation(self) -> int:
        """Get current generation number."""
        return self.current_generation

    def get_mutation_stats(self) -> dict:
        """Get statistics about mutation history."""
        if not self.mutation_history:
            return {'total': 0, 'success_rate': 0.0}

        total = len(self.mutation_history)
        successful = sum(1 for m in self.mutation_history if m['success'])

        return {
            'total': total,
            'successful': successful,
            'success_rate': successful / total if total > 0 else 0.0,
            'current_generation': self.current_generation
        }


class GeneticEvolution:
    """
    Main interface for genetic evolution capabilities.
    Combines polymorphic engine, natural selection, and generation tracking.
    """

    def __init__(self, max_generations: int = 100):
        self.engine = PolymorphicEngine()
        self.selector = NaturalSelection()
        self.tracker = GenerationTracker()
        self.max_generations = max_generations

    def evolve_code(self, source_code: str, max_attempts: int = 10) -> Tuple[str, int, bool]:
        """
        Evolve Python source code through multiple generations.

        Args:
            source_code: Original Python source code
            max_attempts: Maximum mutation attempts

        Returns:
            Tuple of (mutated_code, gene_seed, success)
        """
        for attempt in range(max_attempts):
            gene_seed = random.randint(0, 100000)
            mutated = self.engine.mutate_code(source_code, gene_seed)

            if self.selector.validate_python(mutated):
                self.tracker.increment_generation(gene_seed, True)
                return mutated, gene_seed, True
            else:
                self.tracker.increment_generation(gene_seed, False)

        # If all mutations failed, return original
        return source_code, 0, False

    def evolve_payload(self, payload: str, max_attempts: int = 10) -> Tuple[str, int, bool]:
        """
        Evolve a string payload through multiple generations.

        Args:
            payload: Original payload string
            max_attempts: Maximum mutation attempts

        Returns:
            Tuple of (mutated_payload, gene_seed, success)
        """
        for attempt in range(max_attempts):
            gene_seed = random.randint(0, 100000)
            mutated = self.engine.mutate_string(payload, gene_seed)

            if self.selector.validate_payload(mutated):
                self.tracker.increment_generation(gene_seed, True)
                return mutated, gene_seed, True
            else:
                self.tracker.increment_generation(gene_seed, False)

        # If all mutations failed, return original
        return payload, 0, False

    def get_stats(self) -> dict:
        """Get evolution statistics."""
        return self.tracker.get_mutation_stats()
