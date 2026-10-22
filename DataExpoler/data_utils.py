import pandas as pd

def load_csv(source):
    """Load CSV data from a path or uploaded file-like object."""
    df = pd.read_csv(source)
    if df.empty:
        raise ValueError("The uploaded CSV file has no data rows.")
    if len(df.columns) == 0:
        raise ValueError("The uploaded CSV file has no columns.")
    return df

def dataset_summary(df):
    """Return basic information useful for exploring a dataset."""
    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_values": df.isna().sum(),
        "dtypes": df.dtypes.astype(str),
        "summary": df.describe(include="all").transpose(),
    }
