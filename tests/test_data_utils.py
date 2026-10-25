from io import StringIO

import pandas as pd

from hello_poetry.data_utils import analyze_data, load_csv


def test_load_valid_csv():
    csv_data = StringIO("Name,Age\nSara,21\nAhmad,23")

    df = load_csv(csv_data)

    assert df.shape == (2, 2)
    assert list(df.columns) == ["Name", "Age"]
    assert df["Age"].iloc[0] == 21


def test_analyze_missing_values():
    df = pd.DataFrame({
        "Name": ["Sara", "Ahmad", "Lina"],
        "Age": [21, None, 23],
    })

    result = analyze_data(df)

    assert result["shape"] == (3, 2)
    assert result["missing_values"]["Age"] == 1
    assert result["missing_values"]["Name"] == 0