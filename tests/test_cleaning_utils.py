
import pandas as pd
import pytest

from hello_poetry.cleaning_utils import (
    get_data_quality_report,
    handle_missing_values,
    remove_duplicates,
    detect_outliers,
    convert_numeric_column,
    prepare_train_test_data,
)


def test_data_quality_report():
    df = pd.DataFrame({
        "age": [20, None, 20],
        "city": ["Ramallah", "Nablus", "Ramallah"],
    })

    report = get_data_quality_report(df)

    assert report["shape"] == (3, 2)
    assert report["missing_values"]["age"] == 1
    assert report["duplicates"] == 1


def test_fill_missing_values_with_mode():
    df = pd.DataFrame({
        "city": ["Ramallah", None, "Ramallah", "Nablus"],
    })

    cleaned = handle_missing_values(df, "mode")

    assert cleaned["city"].isna().sum() == 0
    assert cleaned.loc[1, "city"] == "Ramallah"


def test_remove_duplicates():
    df = pd.DataFrame({
        "age": [20, 20, 25],
        "city": ["Nablus", "Nablus", "Hebron"],
    })

    cleaned = remove_duplicates(df)

    assert len(cleaned) == 2
    assert cleaned.duplicated().sum() == 0


def test_detect_outliers():
    df = pd.DataFrame({
        "age": [20, 21, 22, 23, 24, 150],
    })

    result = detect_outliers(df)

    assert result["age"] == 1


def test_convert_numeric_column():
    df = pd.DataFrame({
        "income": ["1,928", "4,924", None],
    })

    cleaned = convert_numeric_column(df, "income")

    assert cleaned.loc[0, "income"] == 1928
    assert cleaned.loc[1, "income"] == 4924
    assert cleaned["income"].isna().sum() == 1


def test_train_test_split_and_scaling():
    df = pd.DataFrame({
        "age": [20, 22, 24, 26, 28, 30, 32, 34, 36, 38],
        "churned": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    })

    result = prepare_train_test_data(
        df,
        target_column="churned",
        test_size=0.2,
        scale=True,
    )

    assert result["X_train"].shape[0] == 8
    assert result["X_test"].shape[0] == 2

    # StandardScaler learns only from the training set.
    scaler = result["preprocessor"].named_transformers_[
        "numeric"
    ].named_steps["scaler"]

    assert scaler.n_samples_seen_ == 8
