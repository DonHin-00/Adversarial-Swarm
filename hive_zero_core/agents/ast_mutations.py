"""
AST-Based Semantic-Preserving Mutations

Implements semantic-preserving code mutations using Abstract Syntax Trees (AST).
These mutations change code structure without altering behavior.

Features:
- Variable renaming
- Control flow obfuscation
- Dead code insertion
- Constant folding reversal
- Expression complexity increase
"""

import ast
import logging
from typing import Optional, List
import astor  # For AST to source conversion

from hive_zero_core.security import SecureRandom

logger = logging.getLogger(__name__)


class SemanticPreservingMutator(ast.NodeTransformer):
    """
    AST transformer that applies semantic-preserving mutations.

    Mutations preserve program semantics while changing code signature.
    """

    def __init__(self, mutation_types: Optional[List[str]] = None):
        self.mutation_types = mutation_types or [
            "rename_vars",
            "obfuscate_control",
            "insert_dead_code",
            "complexify_expressions",
            "reorder_statements",
        ]
        self.var_mapping = {}
        self.mutations_applied = 0

    def visit_Name(self, node: ast.Name) -> ast.Name:
        """Rename variables consistently throughout the AST."""
        if "rename_vars" not in self.mutation_types:
            return node

        # Don't rename builtins or module imports
        if node.id in dir(__builtins__):
            return node

        # Create consistent mapping
        if node.id not in self.var_mapping:
            # Generate cryptographically secure random name
            suffix = SecureRandom.random_id(8)
            self.var_mapping[node.id] = f"_{node.id}_{suffix}"
            self.mutations_applied += 1

        node.id = self.var_mapping[node.id]
        return node

    def visit_If(self, node: ast.If) -> ast.If:
        """Obfuscate control flow by adding equivalent conditions."""
        if "obfuscate_control" not in self.mutation_types:
            return self.generic_visit(node)

        # Add always-true condition wrapped around if statement
        if SecureRandom.random_float() < 0.3:
            # Create: if True: <original_if>
            new_if = ast.If(test=ast.Constant(value=True), body=[node], orelse=[])
            self.mutations_applied += 1
            return self.generic_visit(new_if)

        return self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        """Insert dead code at beginning of functions."""
        if "insert_dead_code" not in self.mutation_types:
            return self.generic_visit(node)

        # Insert dead code that gets optimized away
        if SecureRandom.random_float() < 0.4:
            dead_var = f"_dead_{SecureRandom.random_id(6)}"
            dead_code = ast.Assign(
                targets=[ast.Name(id=dead_var, ctx=ast.Store())], value=ast.Constant(value=0)
            )
            node.body.insert(0, dead_code)
            self.mutations_applied += 1

        return self.generic_visit(node)

    def visit_BinOp(self, node: ast.BinOp) -> ast.BinOp:
        """Complexify expressions by adding identity operations."""
        if "complexify_expressions" not in self.mutation_types:
            return self.generic_visit(node)

        # Add + 0 or * 1 to numeric expressions
        if SecureRandom.random_float() < 0.2:
            if isinstance(node.op, ast.Add):
                # x + y becomes (x + 0) + y
                new_node = ast.BinOp(
                    left=ast.BinOp(left=node.left, op=ast.Add(), right=ast.Constant(value=0)),
                    op=node.op,
                    right=node.right,
                )
                self.mutations_applied += 1
                return self.generic_visit(new_node)

        return self.generic_visit(node)


class ASTMutationEngine:
    """
    High-level engine for AST-based mutations.

    Provides simple interface for semantic-preserving code mutations.
    """

    @staticmethod
    def mutate_source(
        source_code: str, mutation_types: Optional[List[str]] = None, mutation_rate: float = 0.5
    ) -> str:
        """
        Apply semantic-preserving mutations to Python source code.

        Args:
            source_code: Python source code as string
            mutation_types: List of mutation types to apply
            mutation_rate: Probability of applying mutations (not used for AST)

        Returns:
            Mutated source code with same semantics

        Raises:
            SyntaxError: If source code cannot be parsed
            ValueError: If source code is empty
        """
        if not source_code or not source_code.strip():
            raise ValueError("Source code cannot be empty")

        try:
            # Parse source to AST
            tree = ast.parse(source_code)

            # Apply mutations
            mutator = SemanticPreservingMutator(mutation_types)
            mutated_tree = mutator.visit(tree)

            # Fix missing locations
            ast.fix_missing_locations(mutated_tree)

            # Convert back to source
            mutated_source = astor.to_source(mutated_tree)

            logger.info(f"Applied {mutator.mutations_applied} AST mutations")
            return mutated_source

        except SyntaxError as e:
            logger.error(f"Failed to parse source code: {e}")
            raise
        except Exception as e:
            logger.error(f"AST mutation failed: {e}")
            # Return original on failure
            return source_code

    @staticmethod
    def analyze_mutations(original: str, mutated: str) -> dict:
        """
        Analyze differences between original and mutated code.

        Returns:
            Dictionary with mutation statistics
        """
        try:
            orig_tree = ast.parse(original)
            mut_tree = ast.parse(mutated)

            orig_nodes = sum(1 for _ in ast.walk(orig_tree))
            mut_nodes = sum(1 for _ in ast.walk(mut_tree))

            return {
                "original_nodes": orig_nodes,
                "mutated_nodes": mut_nodes,
                "nodes_added": mut_nodes - orig_nodes,
                "complexity_increase": (mut_nodes / orig_nodes - 1.0) * 100,
            }
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            return {}


# Example usage
if __name__ == "__main__":
    sample_code = """
def calculate_sum(a, b):
    result = a + b
    return result

if True:
    x = 10
    y = 20
    z = calculate_sum(x, y)
    print(z)
"""

    print("Original code:")
    print(sample_code)
    print("\n" + "=" * 60 + "\n")

    mutated = ASTMutationEngine.mutate_source(sample_code)
    print("Mutated code:")
    print(mutated)
    print("\n" + "=" * 60 + "\n")

    stats = ASTMutationEngine.analyze_mutations(sample_code, mutated)
    print("Mutation statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
