#!/bin/bash
# Build validation script for Adversarial-Swarm

set -e

echo "========================================="
echo "Adversarial-Swarm Build Validation"
echo "========================================="
echo ""

# 1. Check Python version
echo "1. Checking Python version..."
python3 --version
echo ""

# 2. Verify project structure
echo "2. Verifying project structure..."
for dir in hive_zero_core tests .github/workflows; do
    if [ -d "$dir" ]; then
        echo "   ✓ $dir/"
    else
        echo "   ✗ $dir/ not found"
        exit 1
    fi
done
echo ""

# 3. Check required files
echo "3. Checking required files..."
for file in setup.py pyproject.toml requirements.txt Makefile Dockerfile README.md LICENSE; do
    if [ -f "$file" ]; then
        echo "   ✓ $file"
    else
        echo "   ✗ $file not found"
        exit 1
    fi
done
echo ""

# 4. Python compilation check
echo "4. Checking Python code compilation..."
python3 -m compileall hive_zero_core -q
echo "   ✓ All Python files compile"
echo ""

# 5. Module import check
echo "5. Checking module imports..."
python3 -c "import hive_zero_core"
echo "   ✓ hive_zero_core imports successfully"
echo ""

echo "========================================="
echo "✓ Build validation complete!"
echo "========================================="
