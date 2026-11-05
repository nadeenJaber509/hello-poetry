
"""Tests for the AI Learning Lab homepage and data analysis."""

from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest

from hello_poetry.data_utils import analyze_data

APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


def test_home_loads():
    """The homepage loads without exceptions."""
    at = AppTest.from_file(APP_PATH).run()

    assert not at.exception

    content = " ".join(
        element.value for element in at.markdown
    )

    assert "From Data to" in content
    assert "Intelligent Systems" in content


def test_learning_path_is_displayed():
    """The homepage explains all seven learning stages."""
    at = AppTest.from_file(APP_PATH).run()

    assert not at.exception

    content = " ".join(
        element.value for element in at.markdown
    )

    expected_descriptions = [
        "Understand datasets",
        "Handle missing values",
        "Train models",
        "text classification",
        "similar text using embeddings",
        "chunking and vector database",
        "document retrieval",
    ]

    for description in expected_descriptions:
        assert description.lower() in content.lower()


def test_sample_dataset_analysis():
    """The sample data produces the expected overview."""
    df = pd.DataFrame({
        "Name": ["Sara", "Ahmad", "Lina", "Omar", "Maya"],
        "Age": [21, 23, 22, 25, 24],
        "Score": [85, 92, 78, 88, 95],
        "Department": [
            "CS",
            "Engineering",
            "CS",
            "Business",
            "Engineering",
        ],
    })

    result = analyze_data(df)

    assert result["shape"] == (5, 4)
    assert df["Age"].notna().all()
    assert df["Score"].notna().all()
