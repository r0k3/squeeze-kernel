# Contributing

Thanks for your interest in improving `squeeze-kernel`.

## Local Setup

This project uses `uv` for environment management.

```bash
uv sync --extra full --extra dev
```

## Validation

Please run the full local checks before opening a pull request:

```bash
uv run python -m ruff check .
uv run python -m pytest
uv build
```

## Guidelines

- Keep changes focused and well-scoped.
- Add or update tests for behavior changes.
- Update `README.md` or examples when the public API changes.
- Prefer small pull requests with a clear motivation.
