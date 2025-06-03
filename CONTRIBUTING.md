# Contributing to SDG Hub

This is a guide for getting started on contributing to SDG Hub.

## Dev Requirements

Install the development dependencies using the optional `dev` group:

```bash
pip install .[dev]
```

If you’re using a fresh virtual environment, this will install both the core and development requirements declared in `pyproject.toml`.


## Linting

SDG Hub uses a Makefile for linting.

- CI changes should pass the Action linter - you can run this via `make actionlint`

- Docs changes should pass the Markdown linter - you can run this via `make md-lint`

- Code changes should pass the Code linter - you can run this via `make verify`

## Testing

SDG Hub uses [tox](https://tox.wiki/) for test automation and [pytest](https://docs.pytest.org/) as a test framework.

You can run all tests by simply running the `tox -e py3-unit` command.
