import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


def train_and_evaluate(df, target, features, model_name, test_size=0.2, random_state=42):
    data = df[features + [target]].dropna(subset=[target]).copy()
    X = data[features]
    y = data[target].astype(str)

    if y.nunique() < 2:
        raise ValueError("Target must contain at least two classes.")
    if len(data) < 5:
        raise ValueError("Please provide at least five complete rows.")
    counts = y.value_counts()
    stratify = y if counts.min() >= 2 and int(len(y) * test_size) >= y.nunique() else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )

    numeric_features = X.select_dtypes(include="number").columns.tolist()
    categorical_features = [c for c in features if c not in numeric_features]

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    transformers = []
    if numeric_features:
        transformers.append(("numeric", numeric_pipe, numeric_features))
    if categorical_features:
        transformers.append(("categorical", categorical_pipe, categorical_features))
    preprocess = ColumnTransformer(transformers=transformers)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=random_state),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=random_state),
    }
    if model_name not in models:
        raise ValueError("Unsupported model selected.")

    model = Pipeline([("preprocess", preprocess), ("classifier", models[model_name])])
    model.fit(X_train, y_train)
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(X_train, y_train)
    baseline_pred = baseline.predict(X_test)

    labels = sorted(y.unique())
    cm = confusion_matrix(y_test, test_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=[f"Actual: {x}" for x in labels], columns=[f"Predicted: {x}" for x in labels])

    return {
        "train_accuracy": accuracy_score(y_train, train_pred),
        "test_accuracy": accuracy_score(y_test, test_pred),
        "baseline_accuracy": accuracy_score(y_test, baseline_pred),
        "confusion_matrix": cm_df,
        "report": classification_report(y_test, test_pred, zero_division=0),
    }
