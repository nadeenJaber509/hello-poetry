import pandas as pd 

def load_csv(file):
    try:
        df = pd.read_csv(file)

        if df.empty:
            raise ValueError("The CSV file is empty.")
        
        return df
    except (pd.errors.EmptyDataError, pd.errors.ParserError,
            UnicodeDecodeError, ValueError) as error:
        
        raise ValueError(
                    "Unable to read this CSV file. Please upload a valid CSV."
                ) from error

def analyze_data(df):
    return {
        "shape": df.shape,
        "column_types": df.dtypes.astype(str),
        "statistics": df.describe(include="all"),
        "missing_values": df.isnull().sum(),
    }