
import streamlit as st
import pandas as pd

from hello_poetry.data_utils import load_csv
from hello_poetry.cleaning_utils import (
    get_data_quality_report,
    convert_numeric_column,
    handle_missing_values,
    remove_duplicates,
    detect_outliers,
    prepare_train_test_data,
)


st.set_page_config(
    page_title="Clean Data",
    page_icon="🧹",
    layout="wide",
)

st.title("🧹 Clean Data")
st.caption("Inspect, clean, and prepare datasets for machine learning.")

# --------------------------------------------------
# 1. Upload CSV
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload a CSV file",
    type=["csv"],
)

if uploaded_file is None:
    st.info("Upload a CSV file to start cleaning.")
    st.stop()

try:
    original_df = load_csv(uploaded_file)
except ValueError as error:
    st.error(str(error))
    st.stop()


# --------------------------------------------------
# 2. Original Dataset
# --------------------------------------------------

st.subheader("1. Original Dataset")

original_report = get_data_quality_report(original_df)

col1, col2, col3 = st.columns(3)

col1.metric("Rows", original_report["shape"][0])
col2.metric("Columns", original_report["shape"][1])
col3.metric("Duplicates", original_report["duplicates"])

st.dataframe(original_df.head(), use_container_width=True)


# --------------------------------------------------
# 3. Data Quality Report
# --------------------------------------------------

st.subheader("2. Data Quality Report")

col1, col2 = st.columns(2)

with col1:
    st.write("Missing Values")
    st.dataframe(original_report["missing_values"])

with col2:
    st.write("Column Types")
    st.dataframe(original_report["data_types"])

st.write("Potential Outliers (IQR)")
st.write(detect_outliers(original_df))


# --------------------------------------------------
# 4. Cleaning Options
# --------------------------------------------------

st.subheader("3. Cleaning Options")

cleaned_df = original_df.copy()

text_columns = cleaned_df.select_dtypes(
    include=["object", "str", "string"]
).columns.tolist()

convert_columns = st.multiselect(
    "Convert numeric-looking text columns",
    text_columns,
    default=["income"] if "income" in text_columns else [],
)

for column in convert_columns:
    cleaned_df = convert_numeric_column(cleaned_df, column)


# IMPORTANT:
# Keep a separate copy for ML preprocessing.
# We remove duplicates but do not fill missing values yet.
# The ML pipeline will fit imputers using training data only.
ml_df = remove_duplicates(cleaned_df)


missing_strategy = st.selectbox(
    "Handle missing values",
    [
        "Keep for ML pipeline",
        "Drop rows",
        "Fill with mean",
        "Fill with median",
        "Fill with mode",
    ],
    index=4,
)


if missing_strategy == "Drop rows":
    cleaned_df = handle_missing_values(cleaned_df, "drop")

elif missing_strategy == "Fill with mean":
    cleaned_df = handle_missing_values(cleaned_df, "mean")

elif missing_strategy == "Fill with median":
    cleaned_df = handle_missing_values(cleaned_df, "median")

elif missing_strategy == "Fill with mode":
    cleaned_df = handle_missing_values(cleaned_df, "mode")


remove_dupes = st.checkbox(
    "Remove duplicate rows",
    value=True,
)

if remove_dupes:
    cleaned_df = remove_duplicates(cleaned_df)


# --------------------------------------------------
# 5. Before / After Comparison
# --------------------------------------------------

st.subheader("4. Before / After Comparison")

cleaned_report = get_data_quality_report(cleaned_df)

col1, col2 = st.columns(2)

with col1:
    st.write("Before")

    st.dataframe(
        original_df.head(),
        use_container_width=True,
    )

    st.write("Rows:", original_report["shape"][0])
    st.write("Missing:", int(original_df.isna().sum().sum()))
    st.write("Duplicates:", original_report["duplicates"])

with col2:
    st.write("After")

    st.dataframe(
        cleaned_df.head(),
        use_container_width=True,
    )

    st.write("Rows:", cleaned_report["shape"][0])
    st.write("Missing:", int(cleaned_df.isna().sum().sum()))
    st.write("Duplicates:", cleaned_report["duplicates"])


# --------------------------------------------------
# 6. Download Cleaned CSV
# --------------------------------------------------

st.download_button(
    "Download Cleaned CSV",
    data=cleaned_df.to_csv(index=False).encode("utf-8"),
    file_name="cleaned_data.csv",
    mime="text/csv",
)


# --------------------------------------------------
# 7. Machine Learning Preparation
# --------------------------------------------------

st.subheader("5. Machine Learning Preparation")

st.caption(
    "ML preprocessing uses a separate dataset. "
    "Missing-value imputation and scaling are fitted "
    "on the training set only."
)

target_columns = ml_df.columns.tolist()

target = st.selectbox(
    "Select target column",
    target_columns,
    index=(
        target_columns.index("churned")
        if "churned" in target_columns
        else 0
    ),
)

test_percent = st.slider(
    "Test set size (%)",
    min_value=10,
    max_value=40,
    value=20,
    step=5,
)

scale = st.checkbox(
    "Scale numeric features",
    value=True,
)

encode = st.checkbox(
    "One-Hot Encode categories",
    value=True,
)

if st.button("Prepare Train / Test Data"):
    try:
        result = prepare_train_test_data(
            ml_df,
            target_column=target,
            test_size=test_percent / 100,
            scale=scale,
            encode=encode,
        )

        st.success("Dataset prepared successfully!")

        col1, col2 = st.columns(2)

        col1.metric(
            "Training Rows",
            result["X_train"].shape[0],
        )

        col2.metric(
            "Testing Rows",
            result["X_test"].shape[0],
        )

        st.write(
            "Training Features:",
            result["X_train"].shape[1],
        )

        st.write(
            "Testing Features:",
            result["X_test"].shape[1],
        )

    except ValueError as error:
        st.error(str(error))
