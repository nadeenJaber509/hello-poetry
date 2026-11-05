
"""Educational Streamlit page for data cleaning and ML preparation."""

import streamlit as st

from hello_poetry.cleaning_utils import (
    convert_numeric_column,
    detect_outliers,
    get_data_quality_report,
    handle_missing_values,
    prepare_train_test_data,
    remove_duplicates,
)
from hello_poetry.data_utils import load_csv

st.set_page_config(
    page_title="Data Cleaning | AI Learning Lab",
    page_icon="🧹",
    layout="wide",
)


# --------------------------------------------------
# Educational introduction
# --------------------------------------------------

st.caption("TASK 03 · DATA & MACHINE LEARNING")
st.title("Data Cleaning & Preparation")

st.write(
    "Inspect dataset quality, fix common data problems, "
    "and prepare data for machine learning."
)

with st.container(border=True):
    st.markdown("#### What is Data Cleaning?")
    st.write(
        "Data cleaning is the process of finding and "
        "handling missing values, duplicate rows, "
        "incorrect data types, and potential outliers. "
        "It helps make datasets more reliable for "
        "analysis and machine learning."
    )

col1, col2 = st.columns(2, gap="medium")

with col1, st.container(border=True):
    st.markdown("**Try it**")
    st.write(
        "Upload a CSV file, review the data quality "
        "report, apply cleaning options, and download "
        "the cleaned dataset."
    )

with col2, st.container(border=True):
    st.markdown("**What to notice**")
    st.write(
        "Compare missing values and duplicates before "
        "and after cleaning. Notice how ML preparation "
        "uses training data to fit preprocessing steps."
    )

st.divider()


# --------------------------------------------------
# Interactive Demo
# --------------------------------------------------

st.subheader("Interactive Demo")
st.caption("Upload a CSV file to explore cleaning options.")


# --------------------------------------------------
# 1. Upload CSV
# --------------------------------------------------

st.subheader("1. Upload Dataset")

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

st.subheader("2. Original Dataset")

original_report = get_data_quality_report(original_df)

col1, col2, col3 = st.columns(3)

col1.metric("Rows", original_report["shape"][0])
col2.metric("Columns", original_report["shape"][1])
col3.metric("Duplicates", original_report["duplicates"])

st.dataframe(
    original_df.head(),
    width="stretch",
)


# --------------------------------------------------
# 3. Data Quality Report
# --------------------------------------------------

st.subheader("3. Data Quality Report")

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

st.subheader("4. Cleaning Options")

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
    cleaned_df = convert_numeric_column(
        cleaned_df,
        column,
    )


# Keep ML preprocessing separate to avoid data leakage.
# Remove duplicates, but let the ML pipeline fit
# missing-value imputers using training data only.
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
    cleaned_df = handle_missing_values(
        cleaned_df,
        "drop",
    )

elif missing_strategy == "Fill with mean":
    cleaned_df = handle_missing_values(
        cleaned_df,
        "mean",
    )

elif missing_strategy == "Fill with median":
    cleaned_df = handle_missing_values(
        cleaned_df,
        "median",
    )

elif missing_strategy == "Fill with mode":
    cleaned_df = handle_missing_values(
        cleaned_df,
        "mode",
    )


remove_dupes = st.checkbox(
    "Remove duplicate rows",
    value=True,
)

if remove_dupes:
    cleaned_df = remove_duplicates(cleaned_df)


# --------------------------------------------------
# 5. Before / After Comparison
# --------------------------------------------------

st.subheader("5. Before / After Comparison")

cleaned_report = get_data_quality_report(cleaned_df)

col1, col2 = st.columns(2)

with col1:
    st.write("Before")

    st.dataframe(
        original_df.head(),
        width="stretch",
    )

    st.write("Rows:", original_report["shape"][0])
    st.write(
        "Missing:",
        int(original_df.isna().sum().sum()),
    )
    st.write(
        "Duplicates:",
        original_report["duplicates"],
    )

with col2:
    st.write("After")

    st.dataframe(
        cleaned_df.head(),
        width="stretch",
    )

    st.write("Rows:", cleaned_report["shape"][0])
    st.write(
        "Missing:",
        int(cleaned_df.isna().sum().sum()),
    )
    st.write(
        "Duplicates:",
        cleaned_report["duplicates"],
    )


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

st.subheader("6. Machine Learning Preparation")

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

        st.success(
            "Dataset prepared successfully!"
        )

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
