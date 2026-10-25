

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


def train_model(
    df,
    target_column,
    feature_columns,
    model_name="Logistic Regression",
    test_size=0.2,
):
    """Train and evaluate a classification model."""

    available_models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=42,
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=5,
            random_state=42,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=42,
        ),
    }

    if model_name not in available_models:
        raise ValueError("Invalid model selection.")

    if target_column not in df.columns:
        raise ValueError("Target column not found.")

    if not feature_columns:
        raise ValueError("Select at least one feature.")

    if target_column in feature_columns:
        raise ValueError("Target cannot also be a feature.")

    if any(column not in df.columns for column in feature_columns):
        raise ValueError("One or more features were not found.")

    if not 0 < test_size < 1:
        raise ValueError("Test size must be between 0 and 1.")

    X = df[feature_columns].copy()
    y = df[target_column].copy()

    if y.isna().any():
        raise ValueError("Target column contains missing values.")

    if y.nunique() < 2:
        raise ValueError("Target must have at least two classes.")

    numeric_columns = X.select_dtypes(
        include="number"
    ).columns.tolist()

    categorical_columns = X.select_dtypes(
        include=["object", "str", "string", "category"]
    ).columns.tolist()

    if len(numeric_columns) + len(categorical_columns) != len(
        feature_columns
    ):
        raise ValueError("Unsupported feature data types.")

    # Split data BEFORE fitting preprocessing.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=42,
        stratify=y,
    )

    transformers = []

    if numeric_columns:
        numeric_steps = [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]

        transformers.append((
            "numeric",
            Pipeline(numeric_steps),
            numeric_columns,
        ))

    if categorical_columns:
        categorical_steps = [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]

        transformers.append((
            "categorical",
            Pipeline(categorical_steps),
            categorical_columns,
        ))

    preprocessor = ColumnTransformer(
        transformers=transformers
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", available_models[model_name]),
    ])

    # Fit preprocessing and classifier on TRAIN ONLY.
    pipeline.fit(X_train, y_train)


    train_predictions = pipeline.predict(X_train)
    test_predictions = pipeline.predict(X_test)

    train_accuracy = accuracy_score(
        y_train,
        train_predictions,
    )

    test_accuracy = accuracy_score(
        y_test,
        test_predictions,
    )

    # Baseline: always predict the majority TRAIN class.
    majority_class = y_train.mode().iloc[0]

    baseline_predictions = [majority_class] * len(y_test)

    baseline_accuracy = accuracy_score(
        y_test,
        baseline_predictions,
    )

    labels = sorted(y.unique().tolist())

    matrix = confusion_matrix(
        y_test,
        test_predictions,
        labels=labels,
    )

    return {
        "model": pipeline,
        "model_name": model_name,
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy,
        "baseline_accuracy": baseline_accuracy,
        "majority_class": majority_class,
        "confusion_matrix": matrix,
        "labels": labels,
        "train_rows": len(X_train),
        "test_rows": len(X_test),
    }
