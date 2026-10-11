# Hello World Python Project

A simple Hello World Python project created using Poetry for dependency management and pytest for testing.

## Requirements

- Python 3.14 or later (below Python 4.0)
- Poetry

## Installation

Install the project dependencies:

```bash
poetry install
```

## Run

Run the Hello World function:

```bash
poetry run python -c "from hello_poetry.hello import hello; print(hello())"
```

Expected output:

```text
Hello, World!
```

## Testing

Run the automated test:

```bash
poetry run pytest -v
```

The test verifies that the `hello()` function returns the expected message.

## Technologies

- Python
- Poetry
- pytest