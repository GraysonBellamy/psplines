# PSplines Makefile
# Common development and documentation tasks

.PHONY: help docs docs-serve docs-build docs-clean test lint format install dev-install

# Default target
help:
	@echo "PSplines Development Commands"
	@echo "============================"
	@echo ""
	@echo "Documentation:"
	@echo "  docs-serve    - Serve documentation locally (auto-reload)"
	@echo "  docs-build    - Build documentation site"
	@echo "  docs-test     - Test documentation build"
	@echo "  docs-clean    - Clean documentation build files"
	@echo ""
	@echo "Development:"
	@echo "  install       - Install package in development mode"
	@echo "  dev-install   - Install with all development dependencies"
	@echo "  test          - Run test suite"
	@echo "  lint          - Run linting with ruff"
	@echo "  format        - Format code with ruff"
	@echo "  typecheck     - Run type checking with mypy"
	@echo ""
	@echo "Examples:"
	@echo "  examples      - Run all example scripts"
	@echo ""

# Documentation commands
docs: docs-serve

docs-serve:
	@echo "🚀 Starting documentation server..."
	@echo "📖 Documentation will be available at http://127.0.0.1:8000"
	uv run mkdocs serve

docs-build:
	@echo "🔨 Building documentation..."
	uv run mkdocs build --verbose

docs-test:
	@echo "🧪 Testing documentation..."
	python scripts/test_docs.py

docs-clean:
	@echo "🧹 Cleaning documentation build files..."
	rm -rf site/
	@echo "✅ Documentation cleaned"

# Development commands
install:
	@echo "📦 Installing PSplines in development mode..."
	pip install -e .

dev-install:
	@echo "📦 Installing PSplines with development dependencies..."
	uv sync --dev

test:
	@echo "🧪 Running tests..."
	uv run pytest tests/ -v

test-cov:
	@echo "🧪 Running tests with coverage..."
	uv run pytest tests/ --cov=psplines --cov-report=html --cov-report=term

lint:
	@echo "🔍 Running linting..."
	uv run ruff check src/ tests/ examples/

format:
	@echo "✨ Formatting code..."
	uv run ruff format src/ tests/ examples/
	uv run ruff check --fix src/ tests/ examples/

typecheck:
	@echo "🔎 Running type checking..."
	uv run mypy src/psplines/

# Quality checks (all at once)
check: lint typecheck test
	@echo "✅ All quality checks passed!"

# Examples
examples:
	@echo "🏃 Running example scripts..."
	@for script in examples/*.py; do \
		echo "Running $$script..."; \
		uv run python "$$script" || exit 1; \
	done
	@echo "✅ All examples completed successfully!"

# Clean up
clean: docs-clean
	@echo "🧹 Cleaning build artifacts..."
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete
	@echo "✅ Cleanup completed"

# Build package
build:
	@echo "📦 Building package..."
	uv build

# Development workflow
dev: dev-install format lint typecheck test docs-test
	@echo "🎉 Development setup complete!"
	@echo "💡 Run 'make docs-serve' to start the documentation server"

# CI simulation
ci: lint typecheck test docs-build
	@echo "✅ CI checks completed successfully!"