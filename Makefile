PYTHON = uv run python
MAIN = src

install:
	uv sync

run:
	$(PYTHON) -m $(MAIN)

debug:
	$(PYTHON) -m pdb -m $(MAIN)

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

lint:
	-flake8 . --exclude=llm_sdk
	mypy . --exclude llm_sdk --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

test:
	uv run pytest

test-strict:
	uv run pytest -v -s

.PHONY: install run debug clean lint test test-strict