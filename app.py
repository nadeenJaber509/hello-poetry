
import streamlit as st
import pandas as pd

from hello_poetry.data_utils import load_csv, analyze_data


st.set_page_config(
    page_title="Data Explorer",
    page_icon="📊",
    layout="wide",
)


# Sample dataset
@st.cache_data
def get_sample_data():
    return pd.DataFrame({
        "Name": ["Sara", "Ahmad", "Lina", "Omar", "Maya"],
        "Age": [21, 23, 22, 25, 24],
        "Score": [85, 92, 78, 88, 95],
        "Department": ["CS", "Engineering", "CS", "Business", "Engineering"],
    })


st.title("📊 Data Explorer")
st.caption("Upload a CSV file or explore the sample dataset.")

# Upload section
uploaded_file = st.file_uploader(
    "Upload your CSV file",
    type=["csv"],
)

if uploaded_file is not None:
    try:
        df = load_csv(uploaded_file)
        st.success("CSV file loaded successfully!")
    except (ValueError, UnicodeError) as error:
        st.error(str(error))
        st.stop()
else:
    df = get_sample_data()
    st.info("Using sample dataset. Upload a CSV to explore your own data.")


# Dataset overview
result = analyze_data(df)

st.subheader("Dataset Overview")

col1, col2 = st.columns(2)
col1.metric("Rows", result["shape"][0])
col2.metric("Columns", result["shape"][1])


# Data preview
st.subheader("Data Preview")
st.dataframe(df.head(), use_container_width=True)


# Column types
st.subheader("Column Types")
st.dataframe(
    result["column_types"].rename("Data Type"),
    use_container_width=True,
)


# Summary statistics
st.subheader("Summary Statistics")
st.dataframe(
    result["statistics"],
    use_container_width=True,
)


# Missing values
st.subheader("Missing Values")
st.dataframe(
    result["missing_values"].rename("Missing Count"),
    use_container_width=True,
)


# Interactive charts
st.sidebar.header("Chart Settings")

numeric_columns = df.select_dtypes(include="number").columns.tolist()

if numeric_columns:
    selected_column = st.sidebar.selectbox(
        "Choose a column",
        numeric_columns,
    )

    chart_type = st.sidebar.selectbox(
        "Chart type",
        ["Histogram", "Line Chart", "Bar Chart"],
    )

    st.subheader(f"{selected_column} Visualization")

    if chart_type == "Histogram":
        st.bar_chart(
            df[selected_column].dropna().value_counts().sort_index()
        )
    elif chart_type == "Line Chart":
        st.line_chart(df[selected_column])
    else:
        st.bar_chart(df[selected_column])

else:
    st.warning("No numeric columns available for visualization.")
