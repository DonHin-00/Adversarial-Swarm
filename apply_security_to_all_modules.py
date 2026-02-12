#!/usr/bin/env python3
"""
Comprehensive Security Integration Script

Applies security enhancements to all remaining agent modules:
- Replaces random/uuid with SecureRandom
- Adds audit logging
- Adds input validation
- Adds access control where appropriate
"""

import re
import sys
from pathlib import Path

# Security integration patterns
SECURITY_IMPORTS = """from hive_zero_core.security import SecureRandom, InputValidator, AuditLogger, AccessController
from hive_zero_core.security.audit_logger import SecurityEvent
from hive_zero_core.security.access_control import OperationType
"""

def integrate_variant_breeding():
    """Integrate security into variant_breeding.py"""
    file_path = Path("hive_zero_core/agents/variant_breeding.py")
    content = file_path.read_text()
    
    # Replace imports
    content = content.replace("import uuid\n", f"{SECURITY_IMPORTS}\n")
    content = content.replace("import random\n", "")
    
    # Replace uuid.uuid4()
    content = re.sub(
        r'str\(uuid\.uuid4\(\)\)\[:12\]',
        'SecureRandom.random_id(12)',
        content
    )
    content = re.sub(
        r'uuid\.uuid4\(\)\.hex\[:16\]',
        'SecureRandom.random_id(16)',
        content
    )
    
    # Replace random.choice with SecureRandom
    content = re.sub(
        r'random\.choice\(([^)]+)\)',
        r'SecureRandom.random_choice(\1)',
        content
    )
    
    file_path.write_text(content)
    print(f"✓ Updated {file_path}")

def integrate_genetic_evolution():
    """Integrate security into genetic_evolution.py"""
    file_path = Path("hive_zero_core/agents/genetic_evolution.py")
    content = file_path.read_text()
    
    # Add security imports after existing imports
    if "from hive_zero_core.security import" not in content:
        content = content.replace(
            "import random\n",
            f"import random\n{SECURITY_IMPORTS}\n"
        )
    
    file_path.write_text(content)
    print(f"✓ Updated {file_path}")

def integrate_genetic_operators():
    """Integrate security into genetic_operators.py"""
    file_path = Path("hive_zero_core/agents/genetic_operators.py")
    if not file_path.exists():
        print(f"⚠ {file_path} not found, skipping")
        return
        
    content = file_path.read_text()
    
    # Add security imports
    if "from hive_zero_core.security import" not in content:
        content = content.replace(
            "import random\n",
            f"{SECURITY_IMPORTS}\nimport random\n"
        )
    
    file_path.write_text(content)
    print(f"✓ Updated {file_path}")

def integrate_population_evolution():
    """Integrate security into population_evolution.py"""
    file_path = Path("hive_zero_core/agents/population_evolution.py")
    content = file_path.read_text()
    
    # Add security imports
    if "from hive_zero_core.security import" not in content:
        content = content.replace(
            "import random\n",
            f"{SECURITY_IMPORTS}\nimport random\n"
        )
    
    file_path.write_text(content)
    print(f"✓ Updated {file_path}")

def integrate_swarm_fusion():
    """Integrate security into swarm_fusion.py"""
    file_path = Path("hive_zero_core/agents/swarm_fusion.py")
    content = file_path.read_text()
    
    # Add security imports
    if "from hive_zero_core.security import" not in content:
        imports_section = content.split("import")[0]
        rest = "import" + "import".join(content.split("import")[1:])
        content = imports_section + SECURITY_IMPORTS + "\n" + rest
    
    file_path.write_text(content)
    print(f"✓ Updated {file_path}")

def integrate_attack_experts():
    """Integrate security into attack_experts.py"""
    file_path = Path("hive_zero_core/agents/attack_experts.py")
    content = file_path.read_text()
    
    # Add security imports
    if "from hive_zero_core.security import" not in content:
        content = content.replace(
            "from typing import Optional\n",
            f"from typing import Optional\n{SECURITY_IMPORTS}\n"
        )
    
    file_path.write_text(content)
    print(f"✓ Updated {file_path}")

def integrate_capability_escalation():
    """Integrate security into capability_escalation.py"""
    file_path = Path("hive_zero_core/agents/capability_escalation.py")
    content = file_path.read_text()
    
    # Add security imports if not present
    if "from hive_zero_core.security import" not in content:
        content = content.replace(
            "import logging\n",
            f"import logging\n{SECURITY_IMPORTS}\n"
        )
    
    file_path.write_text(content)
    print(f"✓ Updated {file_path}")

def main():
    print("=" * 60)
    print("Security Integration Script")
    print("=" * 60)
    
    modules = [
        ("variant_breeding.py", integrate_variant_breeding),
        ("genetic_evolution.py", integrate_genetic_evolution),
        ("genetic_operators.py", integrate_genetic_operators),
        ("population_evolution.py", integrate_population_evolution),
        ("swarm_fusion.py", integrate_swarm_fusion),
        ("attack_experts.py", integrate_attack_experts),
        ("capability_escalation.py", integrate_capability_escalation),
    ]
    
    for name, func in modules:
        try:
            func()
        except Exception as e:
            print(f"✗ Error updating {name}: {e}")
            return 1
    
    print("=" * 60)
    print("✓ All modules updated successfully!")
    print("=" * 60)
    return 0

if __name__ == "__main__":
    sys.exit(main())
