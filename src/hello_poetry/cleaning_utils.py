
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

def get_data_quality_report(df):
    """Return a report about dataset quality."""
    return {
        "shape": df.shape,
        "missing_values": df.isna().sum(),
        "duplicates": int(df.duplicated().sum()),
        "data_types": df.dtypes.astype(str),
    }


def handle_missing_values(df, strategy="drop"):
    """Handle missing values using the selected strategy."""
    cleaned_df = df.copy()

    if strategy == "drop":
        return cleaned_df.dropna()

    if strategy == "mean":
        numeric_columns = cleaned_df.select_dtypes(
            include="number"
        ).columns

        for column in numeric_columns:
            cleaned_df[column] = cleaned_df[column].fillna(
                cleaned_df[column].mean()
            )

    elif strategy == "median":
        numeric_columns = cleaned_df.select_dtypes(
            include="number"
        ).columns

        for column in numeric_columns:
            cleaned_df[column] = cleaned_df[column].fillna(
                cleaned_df[column].median()
            )

    elif strategy == "mode":
        for column in cleaned_df.columns:
            modes = cleaned_df[column].mode(dropna=True)

            if not modes.empty:
                cleaned_df[column] = cleaned_df[column].fillna(
                    modes.iloc[0]
                )
    else:
        raise ValueError("Invalid missing value strategy.")

    return cleaned_df


def remove_duplicates(df):
    """Remove duplicate rows from the dataset."""
    return df.drop_duplicates().copy()


def detect_outliers(df):
    """Count outliers in numeric columns using the IQR method."""
    outlier_counts = {}

    numeric_columns = df.select_dtypes(
        include="number"
    ).columns

    for column in numeric_columns:
        q1 = df[column].quantile(0.25)
        q3 = df[column].quantile(0.75)

        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outliers = (
            (df[column] < lower_bound)
            | (df[column] > upper_bound)
        )

        outlier_counts[column] = int(outliers.sum())

    return outlier_counts


def convert_numeric_column(df, column):
    """Convert numeric-looking text values into numbers."""
    cleaned_df = df.copy()

    if column not in cleaned_df.columns:
        raise ValueError(f"Column '{column}' does not exist.")

    cleaned_df[column] = pd.to_numeric(
        cleaned_df[column]
        .astype("string")
        .str.replace(",", "", regex=False)
        .str.strip(),
        errors="coerce",
    )

    return cleaned_df


def prepare_train_test_data(
    df,
    target_column="churned",
    test_size=0.2,
    scale=True,
    encode=True,
):
    """Split the data and fit preprocessing on training data only."""

    if target_column not in df.columns:
        raise ValueError("Target column not found.")

    if not 0 < test_size < 1:
        raise ValueError("Test size must be between 0 and 1.")

    features = df.drop(columns=[target_column, "customer_id"], errors="ignore")
    target = df[target_column]

    if features.empty or features.shape[1] == 0:
        raise ValueError("No feature columns available.")

    numeric_columns = features.select_dtypes(
        include="number"
    ).columns.tolist()

    categorical_columns = features.select_dtypes(
        include=["object", "str", "category", "string"]
    ).columns.tolist()

    # 1. Split BEFORE learning preprocessing parameters
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=42,
    )

    numeric_steps = [
        ("imputer", SimpleImputer(strategy="median"))
    ]

    if scale:
        numeric_steps.append(("scaler", StandardScaler()))

    transformers = []

    if numeric_columns:
        transformers.append(
            ("numeric", Pipeline(numeric_steps), numeric_columns)
        )

    if categorical_columns and encode:
        transformers.append(
            (
                "categorical",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OneHotEncoder(handle_unknown="ignore")),
                ]),
                categorical_columns,
            )
        )

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )

    # 2. Learn median, categories, mean and std from TRAIN ONLY
    X_train_processed = preprocessor.fit_transform(X_train)

    # 3. Apply the learned transformations to TEST
    X_test_processed = preprocessor.transform(X_test)

    return {
        "X_train": X_train_processed,
        "X_test": X_test_processed,
        "y_train": y_train,
        "y_test": y_test,
        "preprocessor": preprocessor,
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
    }

