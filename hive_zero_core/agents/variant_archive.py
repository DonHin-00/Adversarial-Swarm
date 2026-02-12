"""
Variant Archive System

Stores and manages successful variant generations for analysis and reuse.
Implements generational tracking, fitness-based selection, and persistence.

Features:
- Generational tracking with lineage
- Fitness-based archival
- Performance metrics
- Export/import capabilities
- Query and retrieval system
"""

import json
import logging
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
import pickle

from hive_zero_core.security import SecureRandom, AuditLogger
from hive_zero_core.security.audit_logger import SecurityEvent

logger = logging.getLogger(__name__)


@dataclass
class ArchivedVariant:
    """Represents an archived variant with complete metadata."""
    variant_id: str
    genome: str
    fitness: float
    generation: int
    tier: str
    role: str
    parent_ids: List[str] = field(default_factory=list)
    offspring_count: int = 0
    jobs_completed: int = 0
    intelligence_gathered: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    @staticmethod
    def from_dict(data: dict) -> 'ArchivedVariant':
        """Create from dictionary."""
        return ArchivedVariant(**data)


class VariantArchive:
    """
    Archive system for storing and managing successful variants.
    
    Provides:
    - Generational tracking
    - Fitness-based selection
    - Performance analysis
    - Persistence to disk
    """
    
    def __init__(self, max_size: int = 1000, auto_prune: bool = True):
        """
        Initialize variant archive.
        
        Args:
            max_size: Maximum number of variants to store
            auto_prune: Automatically prune low-fitness variants when full
        """
        self.max_size = max_size
        self.auto_prune = auto_prune
        self.variants: Dict[str, ArchivedVariant] = {}
        self.generation_index: Dict[int, List[str]] = {}
        self.fitness_index: List[tuple] = []  # (fitness, variant_id)
        self.audit_logger = AuditLogger()
        
        logger.info(f"Initialized VariantArchive (max_size={max_size})")
    
    def add_variant(self, variant: ArchivedVariant) -> bool:
        """
        Add variant to archive.
        
        Args:
            variant: Variant to archive
            
        Returns:
            True if added successfully
        """
        # Check if at capacity
        if len(self.variants) >= self.max_size:
            if self.auto_prune:
                self._prune_lowest_fitness()
            else:
                logger.warning(f"Archive at capacity ({self.max_size}), variant not added")
                return False
        
        # Add to main storage
        self.variants[variant.variant_id] = variant
        
        # Update generation index
        if variant.generation not in self.generation_index:
            self.generation_index[variant.generation] = []
        self.generation_index[variant.generation].append(variant.variant_id)
        
        # Update fitness index
        self.fitness_index.append((variant.fitness, variant.variant_id))
        self.fitness_index.sort(reverse=True)  # Highest fitness first
        
        # Audit log
        self.audit_logger.log_event(
            event_type=SecurityEvent.VARIANT_CREATED,
            actor_id="archive_system",
            action="archive_variant",
            resource=variant.variant_id,
            result="success",
            metadata={'generation': variant.generation, 'fitness': variant.fitness}
        )
        
        logger.info(f"Archived variant {variant.variant_id} (gen={variant.generation}, fitness={variant.fitness:.4f})")
        return True
    
    def get_variant(self, variant_id: str) -> Optional[ArchivedVariant]:
        """Retrieve variant by ID."""
        return self.variants.get(variant_id)
    
    def get_top_variants(self, n: int = 10) -> List[ArchivedVariant]:
        """Get top N variants by fitness."""
        top_ids = [vid for _, vid in self.fitness_index[:n]]
        return [self.variants[vid] for vid in top_ids if vid in self.variants]
    
    def get_generation(self, generation: int) -> List[ArchivedVariant]:
        """Get all variants from a specific generation."""
        variant_ids = self.generation_index.get(generation, [])
        return [self.variants[vid] for vid in variant_ids if vid in self.variants]
    
    def get_by_role(self, role: str) -> List[ArchivedVariant]:
        """Get all variants with specific role."""
        return [v for v in self.variants.values() if v.role == role]
    
    def get_by_tags(self, tags: List[str]) -> List[ArchivedVariant]:
        """Get variants matching all specified tags."""
        return [v for v in self.variants.values() 
                if all(tag in v.tags for tag in tags)]
    
    def update_offspring_count(self, variant_id: str, count: int = 1):
        """Increment offspring count for a variant."""
        if variant_id in self.variants:
            self.variants[variant_id].offspring_count += count
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get archive statistics."""
        if not self.variants:
            return {'total_variants': 0}
        
        fitnesses = [v.fitness for v in self.variants.values()]
        generations = [v.generation for v in self.variants.values()]
        
        return {
            'total_variants': len(self.variants),
            'total_generations': len(self.generation_index),
            'avg_fitness': sum(fitnesses) / len(fitnesses),
            'max_fitness': max(fitnesses),
            'min_fitness': min(fitnesses),
            'oldest_generation': min(generations),
            'newest_generation': max(generations),
            'roles_count': len(set(v.role for v in self.variants.values())),
            'total_jobs_completed': sum(v.jobs_completed for v in self.variants.values()),
            'total_offspring': sum(v.offspring_count for v in self.variants.values())
        }
    
    def _prune_lowest_fitness(self, prune_count: int = 10):
        """Remove variants with lowest fitness."""
        if len(self.fitness_index) <= prune_count:
            return
        
        # Get IDs to remove (from end of sorted list)
        to_remove = [vid for _, vid in self.fitness_index[-prune_count:]]
        
        for vid in to_remove:
            if vid in self.variants:
                variant = self.variants[vid]
                
                # Remove from main storage
                del self.variants[vid]
                
                # Remove from generation index
                if variant.generation in self.generation_index:
                    self.generation_index[variant.generation].remove(vid)
                
                # Remove from fitness index
                self.fitness_index = [(f, v) for f, v in self.fitness_index if v != vid]
        
        logger.info(f"Pruned {len(to_remove)} low-fitness variants")
    
    def save_to_disk(self, filepath: str) -> bool:
        """
        Save archive to disk.
        
        Args:
            filepath: Path to save archive
            
        Returns:
            True if successful
        """
        try:
            path = Path(filepath)
            path.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert to serializable format
            data = {
                'variants': {vid: v.to_dict() for vid, v in self.variants.items()},
                'generation_index': self.generation_index,
                'max_size': self.max_size,
                'auto_prune': self.auto_prune,
                'saved_at': datetime.now().isoformat()
            }
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Saved archive to {filepath} ({len(self.variants)} variants)")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save archive: {e}")
            return False
    
    @staticmethod
    def load_from_disk(filepath: str) -> Optional['VariantArchive']:
        """
        Load archive from disk.
        
        Args:
            filepath: Path to load archive from
            
        Returns:
            Loaded archive or None if failed
        """
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            archive = VariantArchive(
                max_size=data['max_size'],
                auto_prune=data['auto_prune']
            )
            
            # Restore variants
            for vid, vdata in data['variants'].items():
                variant = ArchivedVariant.from_dict(vdata)
                archive.variants[vid] = variant
            
            # Restore generation index
            archive.generation_index = {int(k): v for k, v in data['generation_index'].items()}
            
            # Rebuild fitness index
            archive.fitness_index = [(v.fitness, vid) for vid, v in archive.variants.items()]
            archive.fitness_index.sort(reverse=True)
            
            logger.info(f"Loaded archive from {filepath} ({len(archive.variants)} variants)")
            return archive
            
        except Exception as e:
            logger.error(f"Failed to load archive: {e}")
            return None
    
    def export_best_genomes(self, n: int = 10, output_file: str = "best_genomes.txt") -> bool:
        """Export top N genomes to text file."""
        try:
            top_variants = self.get_top_variants(n)
            
            with open(output_file, 'w') as f:
                f.write(f"Top {len(top_variants)} Variant Genomes\n")
                f.write("=" * 60 + "\n\n")
                
                for i, variant in enumerate(top_variants, 1):
                    f.write(f"Rank {i}: {variant.variant_id}\n")
                    f.write(f"Fitness: {variant.fitness:.6f}\n")
                    f.write(f"Generation: {variant.generation}\n")
                    f.write(f"Role: {variant.role}\n")
                    f.write(f"Genome:\n{variant.genome}\n")
                    f.write("-" * 60 + "\n\n")
            
            logger.info(f"Exported {len(top_variants)} genomes to {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export genomes: {e}")
            return False


# Example usage
if __name__ == "__main__":
    # Create archive
    archive = VariantArchive(max_size=100)
    
    # Add some test variants
    for i in range(15):
        variant = ArchivedVariant(
            variant_id=SecureRandom.random_id(12),
            genome=f"genome_{i}",
            fitness=SecureRandom.random_float(),
            generation=i // 5,
            tier=f"TIER_{i % 3}",
            role=["RECON", "ATTACK", "STEALTH"][i % 3],
            jobs_completed=SecureRandom.random_int(1, 10)
        )
        archive.add_variant(variant)
    
    # Get statistics
    stats = archive.get_statistics()
    print("Archive Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Get top variants
    print("\nTop 5 Variants:")
    for v in archive.get_top_variants(5):
        print(f"  {v.variant_id}: fitness={v.fitness:.4f}, gen={v.generation}")
    
    # Save to disk
    archive.save_to_disk("variant_archive.json")
    print("\n✓ Archive saved to disk")
