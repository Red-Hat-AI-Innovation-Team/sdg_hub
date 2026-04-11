# Contributing to SDG Hub

Thank you for your interest in contributing. This page covers quick setup;
for the full development guide, see [docs/development.md](docs/development.md).

## Quick Setup

```bash
git clone https://github.com/Red-Hat-AI-Innovation-Team/sdg_hub.git
cd sdg_hub
uv sync --extra dev
uv run pre-commit install
uv run pre-commit install --hook-type commit-msg
```

## Run Tests and Lint

```bash
uv run pytest tests/blocks tests/connectors tests/flow tests/utils -m "not (examples or slow)"
uv run ruff check --fix src/ tests/
uv run ruff format src/ tests/
```

## Commit Messages

We use [Conventional Commits](https://www.conventionalcommits.org/),
enforced by pre-commit hooks and CI.

Format: `<type>(<scope>): <description>`

Allowed types: `feat`, `fix`, `docs`, `style`, `refactor`, `perf`,
`test`, `build`, `ci`, `chore`, `revert`.

## What to Read Next

- [Development Guide](docs/development.md) -- full setup, testing,
  linting, CI requirements, and contribution workflows for blocks,
  flows, and connectors.

## Community Guidelines

- Be respectful and inclusive.
- Provide constructive feedback.
- Follow the project's coding standards.
- Report issues responsibly.
