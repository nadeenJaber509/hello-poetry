
import pandas as pd
import streamlit as st

from hello_poetry.cleaning_utils import (
    convert_numeric_column,
    get_data_quality_report,
    remove_duplicates,
)
from hello_poetry.data_utils import load_csv
from hello_poetry.model_utils import train_model

st.set_page_config(
    page_title="Train Model",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 Train Model")
st.caption(
    "Train and evaluate classification models using cleaned data."
)

# ------------------------------------------
# 1. Upload data
# ------------------------------------------

uploaded_file = st.file_uploader(
    "Upload a CSV dataset",
    type=["csv"],
)

if uploaded_file is None:
    st.info("Upload a CSV file to start training.")
    st.stop()

try:
    original_df = load_csv(uploaded_file)
except ValueError as error:
    st.error(str(error))
    st.stop()


# ------------------------------------------
# 2. Reuse Task 3 cleaning utilities
# ------------------------------------------

df = original_df.copy()

text_columns = df.select_dtypes(
    include=["object", "str", "string"]
).columns.tolist()

convert_columns = st.multiselect(
    "Convert numeric-looking text columns",
    text_columns,
    default=["income"] if "income" in text_columns else [],
)

for column in convert_columns:
    df = convert_numeric_column(df, column)

df = remove_duplicates(df)

st.subheader("1. Dataset Overview")

col1, col2, col3 = st.columns(3)

col1.metric("Original Rows", len(original_df))
col2.metric("Rows After Cleaning", len(df))
col3.metric("Duplicates Removed", len(original_df) - len(df))

st.dataframe(df.head(), use_container_width=True)

report = get_data_quality_report(df)

st.caption(
    f"Remaining missing values: "
    f"{int(report['missing_values'].sum())}. "
    "These will be imputed using training data only."
)


# ------------------------------------------
# 3. Select target and features
# ------------------------------------------

st.subheader("2. Select Target and Features")

columns = df.columns.tolist()

target_column = st.selectbox(
    "Target column",
    columns,
    index=columns.index("churned") if "churned" in columns else 0,
)

available_features = [
    column
    for column in columns
    if column != target_column
]

default_features = [
    column
    for column in available_features
    if column != "customer_id"
]

selected_features = st.multiselect(
    "Feature columns",
    available_features,
    default=default_features,
)

st.write("Selected features:", len(selected_features))


# ------------------------------------------
# 4. Select model
# ------------------------------------------

st.subheader("3. Model Settings")

model_name = st.selectbox(
    "Choose a classification model",
    [
        "Logistic Regression",
        "Decision Tree",
        "Random Forest",
    ],
)

test_percent = st.slider(
    "Test set size (%)",
    min_value=10,
    max_value=40,
    value=20,
    step=5,
)


# ------------------------------------------
# 5. Train model
# ------------------------------------------

st.subheader("4. Model Evaluation")

if st.button("🚀 Train Model", type="primary"):

    if not selected_features:
        st.warning("Please select at least one feature.")
        st.stop()

    try:
        with st.spinner("Training model..."):
            result = train_model(
                df=df,
                target_column=target_column,
                feature_columns=selected_features,
                model_name=model_name,
                test_size=test_percent / 100,
            )

        st.success(f"{model_name} trained successfully!")

        # ----------------------------------
        # Accuracy metrics
        # ----------------------------------

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Train Accuracy",
            f"{result['train_accuracy']:.2%}",
        )

        col2.metric(
            "Test Accuracy",
            f"{result['test_accuracy']:.2%}",
        )

        col3.metric(
            "Baseline Accuracy",
            f"{result['baseline_accuracy']:.2%}",
        )

        st.caption(
            f"Training rows: {result['train_rows']} | "
            f"Testing rows: {result['test_rows']}"
        )

        # ----------------------------------
        # Baseline comparison
        # ----------------------------------

        st.subheader("5. Model vs. Baseline")

        comparison = pd.DataFrame({
            "Accuracy": [
                result["train_accuracy"],
                result["test_accuracy"],
                result["baseline_accuracy"],
            ],
        }, index=[
            "Train",
            "Test",
            "Baseline",
        ])

        st.bar_chart(comparison)

        st.write(
            "Baseline predicts the most common "
            "training class:",
            result["majority_class"],
        )

        # ----------------------------------
        # Confusion matrix
        # ----------------------------------

        st.subheader("6. Confusion Matrix")

        labels = result["labels"]

        matrix_df = pd.DataFrame(
            result["confusion_matrix"],
            index=[f"Actual {label}" for label in labels],
            columns=[f"Predicted {label}" for label in labels],
        )

        st.dataframe(
            matrix_df,
            use_container_width=True,
        )

        st.caption(
            "Rows represent actual classes; "
            "columns represent predicted classes."
        )

    except ValueError as error:
        st.error(str(error))

    except Exception as error:  # noqa: BLE001
        st.error(
            f"Training failed: {error}"
        )
