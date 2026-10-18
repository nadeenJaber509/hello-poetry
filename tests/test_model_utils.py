
import pandas as pd
import pytest

from hello_poetry.model_utils import train_model


@pytest.fixture
def sample_data():
    """Small classification dataset for testing."""
    return pd.DataFrame({
        "age": [
            21, 25, 30, 35, 40, 45, 50, 55, 60, 65,
            22, 27, 32, 37, 42, 47, 52, 57, 62, 67,
        ],
        "plan": ["Basic", "Premium"] * 10,
        "churned": [
            0, 0, 1, 0, 0, 1, 0, 0, 1, 0,
            0, 1, 0, 0, 1, 0, 0, 1, 0, 0,
        ],
    })


@pytest.mark.parametrize(
    "model_name",
    [
        "Logistic Regression",
        "Decision Tree",
        "Random Forest",
    ],
)
def test_model_training(sample_data, model_name):
    """Verify each supported model trains successfully."""

    result = train_model(
        df=sample_data,
        target_column="churned",
        feature_columns=["age", "plan"],
        model_name=model_name,
        test_size=0.2,
    )

    # Verify train/test split
    assert result["train_rows"] == 16
    assert result["test_rows"] == 4

    # Accuracy must be between 0 and 1
    assert 0 <= result["train_accuracy"] <= 1
    assert 0 <= result["test_accuracy"] <= 1

    # Verify confusion matrix
    assert result["confusion_matrix"].shape == (2, 2)
    assert result["confusion_matrix"].sum() == 4

    # Verify a fitted model exists
    assert result["model"].named_steps["model"] is not None


def test_majority_class_baseline(sample_data):
    """Verify baseline uses the most common training class."""

    result = train_model(
        df=sample_data,
        target_column="churned",
        feature_columns=["age", "plan"],
        model_name="Logistic Regression",
        test_size=0.2,
    )

    # Class 0 is the majority class
    assert result["majority_class"] == 0

    # Stratified test set has 3 zeros out of 4 rows
    assert result["baseline_accuracy"] == pytest.approx(0.75)


def test_invalid_target(sample_data):
    """Verify an invalid target column raises an error."""

    with pytest.raises(
        ValueError,
        match="Target column not found",
    ):
        train_model(
            df=sample_data,
            target_column="unknown",
            feature_columns=["age", "plan"],
        )
