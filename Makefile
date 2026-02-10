.PHONY: help install install-dev test lint format clean docker-build docker-up docker-down

help:
	@echo "Adversarial-Swarm Makefile Commands:"
	@echo ""
	@echo "  make install       - Install production dependencies"
	@echo "  make install-dev   - Install development dependencies"
	@echo "  make test          - Run tests with coverage"
	@echo "  make lint          - Run all linters"
	@echo "  make format        - Format code with black and isort"
	@echo "  make clean         - Clean build artifacts"
	@echo "  make docker-build  - Build Docker image"
	@echo "  make docker-up     - Start Docker containers"
	@echo "  make docker-down   - Stop Docker containers"
	@echo "  make pre-commit    - Install pre-commit hooks"
	@echo ""

install:
	pip install -r requirements.txt
	pip install -e .

install-dev:
	pip install -r requirements.txt
	pip install -e ".[dev]"

test:
	pytest tests/ -v --cov=hive_zero_core --cov-report=term-missing --cov-report=html

lint:
	@echo "Running flake8..."
	flake8 hive_zero_core
	@echo "Running pylint..."
	pylint hive_zero_core --exit-zero
	@echo "Running mypy..."
	mypy hive_zero_core --ignore-missing-imports
	@echo "Checking code format..."
	black --check hive_zero_core
	isort --check-only hive_zero_core

format:
	@echo "Formatting with black..."
	black hive_zero_core
	@echo "Sorting imports with isort..."
	isort hive_zero_core

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf build dist .pytest_cache .coverage htmlcov .mypy_cache .ruff_cache

docker-build:
	docker-compose build

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

pre-commit:
	pip install pre-commit
	pre-commit install
