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

## Documentation Guidelines

### NumPy-Style Docstrings

If you choose to add docstrings to your functions, we recommend following the NumPy docstring format for consistency with the scientific Python ecosystem.

#### Basic Structure

```python
def example_function(param1, param2=None):
    """Brief description of the function.

    Longer description providing more context about what the function does,
    its purpose, and any important behavioral notes.

    Parameters
    ----------
    param1 : str
        Description of the first parameter
    param2 : int, optional
        Description of the second parameter (default: None)

    Returns
    -------
    bool
        Description of what the function returns

    Raises
    ------
    ValueError
        When invalid input is provided

    Examples
    --------
    >>> result = example_function("hello", 42)
    >>> print(result)
    True
    """
```

#### Key Guidelines

- **Summary**: Start with a concise one-line description
- **Parameters**: Document all function parameters with types and descriptions
- **Returns**: Describe return values with types and meaning
- **Types**: Use standard Python types (`str`, `int`, `list`, `dict`, etc.)
- **Optional parameters**: Mark default parameters as "optional"
- **Examples**: Include simple usage examples when helpful

#### When to Add Docstrings

Docstrings are **optional** but recommended for:
- Public API functions and classes
- Complex functions with multiple parameters
- Functions that might be confusing to other developers
- Core framework components

#### When to Skip Docstrings

You may skip docstrings for:
- Simple utility functions with obvious behavior
- Private/internal functions (starting with `_`)
- Functions with self-explanatory names and simple parameters

**Remember**: Quality over quantity. A well-written docstring is better than a verbose one, and no docstring is better than a poor one.
