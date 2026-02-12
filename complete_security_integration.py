#!/usr/bin/env python3
"""
Complete Security Integration - Phase 3

Replaces all remaining insecure random usage with SecureRandom across all modules.
This script performs the final security hardening.
"""

import re
from pathlib import Path

def replace_random_usage(file_path: Path, module_name: str):
    """Replace insecure random usage with SecureRandom in a file."""
    content = file_path.read_text()
    original = content
    changes = []
    
    # Pattern 1: random.randint(a, b) -> SecureRandom.random_int(a, b)
    pattern1 = r'random\.randint\(([^)]+)\)'
    matches1 = re.findall(pattern1, content)
    if matches1:
        content = re.sub(pattern1, r'SecureRandom.random_int(\1)', content)
        changes.append(f"  - random.randint() → SecureRandom.random_int() ({len(matches1)} occurrences)")
    
    # Pattern 2: random.random() -> SecureRandom.random_float()
    pattern2 = r'random\.random\(\)'
    matches2 = re.findall(pattern2, content)
    if matches2:
        content = re.sub(pattern2, 'SecureRandom.random_float()', content)
        changes.append(f"  - random.random() → SecureRandom.random_float() ({len(matches2)} occurrences)")
    
    # Pattern 3: random.choice(x) -> SecureRandom.random_choice(x)
    pattern3 = r'random\.choice\(([^)]+)\)'
    matches3 = re.findall(pattern3, content)
    if matches3:
        content = re.sub(pattern3, r'SecureRandom.random_choice(\1)', content)
        changes.append(f"  - random.choice() → SecureRandom.random_choice() ({len(matches3)} occurrences)")
    
    # Pattern 4: random.uniform(a, b) -> SecureRandom.random_float() * (b-a) + a
    pattern4 = r'random\.uniform\(([^,]+),\s*([^)]+)\)'
    matches4 = re.findall(pattern4, content)
    if matches4:
        # For uniform, we need to handle it specially
        for low, high in matches4:
            old = f'random.uniform({low}, {high})'
            new = f'(SecureRandom.random_float() * ({high} - ({low})) + ({low}))'
            content = content.replace(old, new)
        changes.append(f"  - random.uniform() → SecureRandom formula ({len(matches4)} occurrences)")
    
    # Pattern 5: random.sample(x, k) -> SecureRandom.random_sample(x, k)
    pattern5 = r'random\.sample\(([^,]+),\s*([^)]+)\)'
    matches5 = re.findall(pattern5, content)
    if matches5:
        content = re.sub(pattern5, r'SecureRandom.random_sample(\1, \2)', content)
        changes.append(f"  - random.sample() → SecureRandom.random_sample() ({len(matches5)} occurrences)")
    
    if content != original:
        file_path.write_text(content)
        print(f"\n✓ Updated {module_name}:")
        for change in changes:
            print(change)
        return True
    else:
        print(f"  ⊙ {module_name}: No changes needed")
        return False

def main():
    print("=" * 70)
    print("Complete Security Integration - Phase 3")
    print("Replacing all insecure random usage with SecureRandom")
    print("=" * 70)
    
    modules = [
        ('hive_zero_core/agents/variant_breeding.py', 'variant_breeding'),
        ('hive_zero_core/agents/genetic_operators.py', 'genetic_operators'),
        ('hive_zero_core/agents/population_evolution.py', 'population_evolution'),
        ('hive_zero_core/agents/swarm_fusion.py', 'swarm_fusion'),
        ('hive_zero_core/agents/capability_escalation.py', 'capability_escalation'),
    ]
    
    updated = 0
    for file_path_str, module_name in modules:
        file_path = Path(file_path_str)
        if file_path.exists():
            if replace_random_usage(file_path, module_name):
                updated += 1
        else:
            print(f"  ✗ {module_name}: File not found")
    
    print("\n" + "=" * 70)
    print(f"✓ Updated {updated}/{len(modules)} modules")
    print("=" * 70)
    
    # Verify compilation
    print("\nVerifying compilation...")
    import py_compile
    import sys
    
    errors = []
    for file_path_str, module_name in modules:
        try:
            py_compile.compile(file_path_str, doraise=True)
            print(f"  ✓ {module_name}")
        except Exception as e:
            errors.append(f"{module_name}: {e}")
            print(f"  ✗ {module_name}: {e}")
    
    if errors:
        print(f"\n✗ {len(errors)} compilation error(s)")
        return 1
    else:
        print(f"\n✓ All {len(modules)} modules compile successfully!")
        return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
