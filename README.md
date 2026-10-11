# Hello Poetry 👋

A simple interactive Python web application built with **Streamlit**, using **Poetry** for dependency management and **pytest** for automated testing.

## Features

- Interactive web interface using Streamlit
- Personalized greeting based on the user's name
- Default "Hello, World!" greeting
- Automated tests for the Python function and Streamlit app

## Technologies

- Python 3.14
- Poetry
- Streamlit
- pytest

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/nadeenJaber509/hello-poetry.git
cd hello-poetry
```

### 2. Install dependencies

Make sure Python 3.14 and Poetry are installed.

```bash
poetry install
```

### 3. Run the application

```bash
poetry run streamlit run app.py
```

Open http://localhost:8501 in your browser.

### 4. Run tests

```bash
poetry run pytest -v
```

## Project Structure

```text
hello-poetry/
├── app.py
├── src/
│   └── hello_poetry/
│       ├── __init__.py
│       └── hello.py
├── tests/
│   ├── test_hello.py
│   └── test_app.py
├── pyproject.toml
├── poetry.lock
├── README.md
└── .gitignore
```

## Collaboration

This repository is shared among team members for our AI training program. Project dependencies are managed through Poetry, and all changes are tracked using Git and GitHub.