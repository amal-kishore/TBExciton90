# Contributing to TBExciton90

We love your input! We want to make contributing to TBExciton90 as easy and transparent as possible, whether it's:

- Reporting a bug
- Discussing the current state of the code
- Submitting a fix
- Proposing new features
- Becoming a maintainer

## Development Process

We use GitHub to host code, to track issues and feature requests, as well as accept pull requests.

1. Fork the repo and create your branch from `main`.
2. If you've added code that should be tested, add tests.
3. If you've changed APIs, update the documentation.
4. Ensure the test suite passes.
5. Make sure your code lints.
6. Issue that pull request!

## Code Style

* Use [Black](https://github.com/psf/black) for Python code formatting
* Follow PEP 8 guidelines
* Write descriptive commit messages
* Add type hints where possible
* Document your functions with docstrings

### Pre-commit hooks

```bash
pip install pre-commit
pre-commit install
```

## Testing

```bash
pytest tests/
```

## License

By contributing, you agree that your contributions will be licensed under its MIT License.