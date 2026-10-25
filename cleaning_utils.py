import pandas as pd
from pandas.api.types import is_numeric_dtype


def clean_dataframe(df, fill_missing=True, remove_duplicates=True):
    """Return a cleaned copy of a DataFrame.

    Numeric missing values are filled with the median; text/category values
    are filled with the mode, or 'Unknown' if the column has no mode.
    """
    cleaned = df.copy()

    if remove_duplicates:
        cleaned = cleaned.drop_duplicates()

    if fill_missing:
        for column in cleaned.columns:
            if not cleaned[column].isna().any():
                continue
            if is_numeric_dtype(cleaned[column]):
                median = cleaned[column].median()
                cleaned[column] = cleaned[column].fillna(median if pd.notna(median) else 0)
            else:
                mode = cleaned[column].mode(dropna=True)
                fill_value = mode.iloc[0] if not mode.empty else "Unknown"
                cleaned[column] = cleaned[column].fillna(fill_value)
    else:
        cleaned = cleaned.dropna()

    return cleaned.reset_index(drop=True)
