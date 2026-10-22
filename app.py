import streamlit as st
import pandas as pd
import os

st.set_page_config(page_title="Progineer Training", page_icon="", layout="wide")

st.title("Progineer Training — Tasks 1 to 4")
st.write("A combined Streamlit project: Hello World, Data Explorer, Data Cleaning, and Model Training.")

task = st.sidebar.radio(
    "Choose a task",
    ["Task 1 — Hello World", "Task 2 — Data Explorer", "Task 3 — Data Cleaning", "Task 4 — Train Model"],
)

if task == "Task 1 — Hello World":
    st.header("Task 1: Hello World")
    st.success("Hello World!")
    st.write("This is the first Streamlit task.")

elif task == "Task 2 — Data Explorer":
    from DataExpoler.data_utils import load_csv, dataset_summary
    st.header("Task 2: Data Explorer")
    uploaded = st.file_uploader("Upload a CSV file", type=["csv"], key="explorer_upload")
    default_path = os.path.join(os.path.dirname(__file__), "DataExpoler", "student-dataset.csv")
    if uploaded is not None:
        try:
            df = load_csv(uploaded)
        except Exception as e:
            st.error(f"Could not read this CSV file: {e}")
            st.stop()
    else:
        st.caption("Showing the included sample dataset. Upload your own CSV to explore it.")
        try:
            df = load_csv(default_path)
        except Exception as e:
            st.error(f"Could not load the sample dataset: {e}")
            st.stop()

    st.subheader("Preview")
    st.dataframe(df.head())
    c1, c2 = st.columns(2)
    c1.metric("Rows", df.shape[0])
    c2.metric("Columns", df.shape[1])
    st.subheader("Column data types")
    st.dataframe(df.dtypes.astype(str).rename("Data type"))
    st.subheader("Summary statistics")
    st.dataframe(df.describe(include="all").transpose())
    st.subheader("Missing values")
    st.dataframe(df.isna().sum().rename("Missing values"))

    st.sidebar.subheader("Explore a column")
    col = st.sidebar.selectbox("Choose column", list(df.columns))
    st.write(f"### Distribution of {col}")
    if pd.api.types.is_numeric_dtype(df[col]):
        st.bar_chart(df[col].dropna().value_counts().sort_index())
    else:
        st.bar_chart(df[col].fillna("(Missing)").astype(str).value_counts().head(20))

elif task == "Task 3 — Data Cleaning":
    from cleaning_utils import clean_dataframe
    st.header("Task 3: Data Cleaning")
    uploaded = st.file_uploader("Upload a CSV file to clean", type=["csv"], key="clean_upload")
    default_path = os.path.join(os.path.dirname(__file__), "DataExpoler", "messy_customers.csv")
    if uploaded is not None:
        try:
            source = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Could not read this CSV file: {e}")
            st.stop()
    else:
        source = pd.read_csv(default_path)
        st.caption("Using the included sample dataset. Upload a different CSV if needed.")

    st.subheader("Before cleaning")
    a, b, c = st.columns(3)
    a.metric("Rows", len(source))
    b.metric("Missing cells", int(source.isna().sum().sum()))
    c.metric("Duplicate rows", int(source.duplicated().sum()))

    fill_method = st.selectbox("How should missing values be handled?", ["Fill (recommended)", "Drop rows with missing values"])
    remove_duplicates = st.checkbox("Remove duplicate rows", value=True)
    cleaned = clean_dataframe(source, fill_missing=(fill_method == "Fill (recommended)"), remove_duplicates=remove_duplicates)

    st.subheader("After cleaning")
    a, b, c = st.columns(3)
    a.metric("Rows", len(cleaned))
    b.metric("Missing cells", int(cleaned.isna().sum().sum()))
    c.metric("Duplicate rows", int(cleaned.duplicated().sum()))
    st.dataframe(cleaned, use_container_width=True)
    st.download_button("Download cleaned CSV", cleaned.to_csv(index=False).encode("utf-8"), "cleaned_data.csv", "text/csv")

elif task == "Task 4 — Train Model":
    from cleaning_utils import clean_dataframe
    from model_utils import train_and_evaluate
    st.header("Task 4: Train a Machine Learning Model")
    uploaded = st.file_uploader("Upload a CSV dataset", type=["csv"], key="model_upload")
    default_path = os.path.join(os.path.dirname(__file__), "DataExpoler", "messy_customers.csv")
    if uploaded is not None:
        try:
            raw = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Could not read this CSV file: {e}")
            st.stop()
    else:
        raw = pd.read_csv(default_path)
        st.info("The included student dataset is for demonstrating exploration and cleaning. For meaningful classification, upload a dataset with a categorical target column.")

    df = clean_dataframe(raw)
    st.write("Cleaned data preview")
    st.dataframe(df.head())
    if len(df.columns) < 2:
        st.warning("The dataset needs at least two columns: one target and at least one feature.")
        st.stop()

    target = st.selectbox("Choose the target column (what you want to predict)", list(df.columns), index=len(df.columns)-1)
    features = st.multiselect("Choose feature columns", [c for c in df.columns if c != target], default=[c for c in df.columns if c != target])
    model_name = st.selectbox("Choose model", ["Logistic Regression", "Decision Tree", "Random Forest"])
    test_size = st.slider("Test data percentage", min_value=10, max_value=40, value=20, step=5)

    if st.button("Train model", type="primary"):
        if not features:
            st.error("Choose at least one feature column.")
        elif df[target].nunique(dropna=True) < 2:
            st.error("The target must contain at least two different classes.")
        else:
            try:
                result = train_and_evaluate(df, target, features, model_name, test_size / 100)
                x1, x2, x3 = st.columns(3)
                x1.metric("Training accuracy", f"{result['train_accuracy']:.3f}")
                x2.metric("Test accuracy", f"{result['test_accuracy']:.3f}")
                x3.metric("Majority-class baseline", f"{result['baseline_accuracy']:.3f}")
                st.subheader("Classification report")
                st.code(result["report"])
                st.subheader("Confusion matrix")
                st.dataframe(result["confusion_matrix"])
                st.caption("Accuracy is a useful starting metric, but it can be misleading when classes are imbalanced. Compare test accuracy with the majority-class baseline.")
            except Exception as e:
                st.error(f"Could not train this model: {e}")
