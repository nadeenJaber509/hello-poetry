import pandas as pd

from cleaning_utils import clean_dataframe


def test_fill_missing_numeric_and_text():
    df = pd.DataFrame({"Age": [10, None, 30], "City": ["Ramallah", None, "Nablus"]})
    result = clean_dataframe(df)
    assert result.isna().sum().sum() == 0
    assert result.loc[1, "Age"] == 20
    assert result.loc[1, "City"] in ["Ramallah", "Nablus"]

def test_remove_duplicate_rows():
    df = pd.DataFrame({"x": [1, 1, 2]})
    result = clean_dataframe(df)
    assert len(result) == 2
    assert not result.duplicated().any()

def test_drop_missing_rows_when_requested():
    df = pd.DataFrame({"x": [1, None, 3]})
    result = clean_dataframe(df, fill_missing=False)
    assert len(result) == 2
    assert result.isna().sum().sum() == 0

def test_returns_copy_and_keeps_original():
    df = pd.DataFrame({"x": [1, None]})
    result = clean_dataframe(df)
    assert df.isna().sum().sum() == 1
    assert result.isna().sum().sum() == 0
