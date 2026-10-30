
import os

import pandas as pd
import streamlit as st

from task6.llm import ask_llm, parse_label
from task6.prompts import PROMPT_A, PROMPT_B, PROMPT_C

st.set_page_config(page_title="Progineer Training", page_icon="", layout="wide")

st.title("Progineer Training — Tasks 1 to 8")
st.write("A combined Streamlit project with data exploration, cleaning, model training, LLM classification, and search.")

task = st.sidebar.radio(
    "Choose a task",
    [
        "Task 1 — Hello World",
        "Task 2 — Data Explorer",
        "Task 3 — Data Cleaning",
        "Task 4 — Train Model",
        "Task 6 — Classifier Lab",
        "Task 7 — Semantic Search",
        "Task 8 — Docs Search",
    ],
)

if task == "Task 1 — Hello World":
    st.header("Task 1: Hello World")
    st.success("Hello World!")
    st.write("This is the first Streamlit task.")

elif task == "Task 2 — Data Explorer":
    from DataExpoler.data_utils import load_csv
    st.header("Task 2: Data Explorer")
    uploaded = st.file_uploader("Upload a CSV file", type=["csv"], key="explorer_upload")
    default_path = os.path.join(os.path.dirname(__file__), "DataExpoler", "student-dataset.csv")
    if uploaded is not None:
        try:
            df = load_csv(uploaded)
        except (ValueError, OSError, pd.errors.ParserError) as e:
            st.error(f"Could not read this CSV file: {e}")
            st.stop()
    else:
        st.caption("Showing the included sample dataset. Upload your own CSV to explore it.")
        try:
            df = load_csv(default_path)
        except (ValueError, OSError, pd.errors.ParserError) as e:
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
        except (ValueError, OSError, pd.errors.ParserError) as e:
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
        except (ValueError, OSError, pd.errors.ParserError) as e:
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
            except (ValueError, OSError, pd.errors.ParserError) as e:
                st.error(f"Could not train this model: {e}")

elif task == "Task 6 — Classifier Lab":
    st.header("Task 6: LLM Sentiment Classifier")
    st.write("Enter a comment, choose a prompt, and predict its sentiment.")

    prompt_options = {
        "Prompt A — Zero-Shot": PROMPT_A,
        "Prompt B — Few-Shot": PROMPT_B,
        "Prompt C — Improved": PROMPT_C,
    }

    selected_prompt = st.selectbox(
        "Choose a prompt",
        list(prompt_options.keys()),
    )

    model = st.selectbox(
        "Choose model",
        ["openai/gpt-oss-120b", "openai/gpt-oss-20b"],
    )

    comment = st.text_area(
        "Enter a comment",
        placeholder="The delivery was fast and the product was excellent.",
    )

    if st.button("Predict sentiment", type="primary"):
        if not comment.strip():
            st.warning("Please enter a comment.")
        else:
            try:
                with st.spinner("Classifying..."):
                    raw_result = ask_llm(
                        system_prompt=prompt_options[selected_prompt],
                        user_message=comment,
                        model=model,
                        temperature=0,
                    )
                    label = parse_label(raw_result)

                if label == "positive":
                    st.success("Sentiment: Positive")
                elif label == "negative":
                    st.error("Sentiment: Negative")
                elif label == "neutral":
                    st.info("Sentiment: Neutral")
                else:
                    st.warning("The model returned an invalid response.")
                    st.code(raw_result)

            except (ValueError, OSError, RuntimeError) as error:
                st.error(
                    "Classification failed. Check your Groq API key, "
                    "internet connection, and model availability."
                )
                st.caption(f"Error details: {error}")

    st.divider()
    st.subheader("Evaluation Results")

    results_files = {
        "120b — Prompt A": "results_A_zero_shot_openai_gpt-oss-120b.csv",
        "120b — Prompt B": "results_B_few_shot_openai_gpt-oss-120b.csv",
        "120b — Prompt C": "results_C_improved_openai_gpt-oss-120b.csv",
        "20b — Prompt B": "results_B_few_shot_openai_gpt-oss-20b.csv",
    }

    results_dir = os.path.join(os.path.dirname(__file__), "task6")
    summaries = []
    loaded_results = {}

    for experiment, filename in results_files.items():
        path = os.path.join(results_dir, filename)

        if not os.path.exists(path):
            st.warning(f"Results file not found: {filename}")
            continue

        try:
            result_df = pd.read_csv(path)

            required_columns = {"label", "prediction"}
            if not required_columns.issubset(result_df.columns):
                st.warning(
                    f"{filename} is missing required columns: "
                    "'label' and 'prediction'."
                )
                continue

            # Normalize labels to avoid case and whitespace mismatches.
            result_df["label"] = (
                result_df["label"].astype(str).str.strip().str.lower()
            )
            result_df["prediction"] = (
                result_df["prediction"].astype(str).str.strip().str.lower()
            )

            valid_labels = {"positive", "negative", "neutral"}
            valid = result_df["prediction"].isin(valid_labels)
            accuracy = (
                (
                    result_df.loc[valid, "label"]
                    == result_df.loc[valid, "prediction"]
                ).mean()
                if valid.any()
                else 0.0
            )

            loaded_results[experiment] = result_df
            summaries.append({
                "Experiment": experiment,
                "Accuracy": f"{accuracy:.1%}",
                "Invalid predictions": int((~valid).sum()),
                "Comments": len(result_df),
            })

        except (ValueError, OSError, pd.errors.ParserError) as error:
            st.warning(f"Could not load {filename}: {error}")

    if summaries:
        st.subheader("Accuracy Summary")
        st.dataframe(
            pd.DataFrame(summaries),
            hide_index=True,
            use_container_width=True,
        )

        selected_experiment = st.selectbox(
            "Inspect an experiment",
            list(loaded_results.keys()),
        )
        inspected = loaded_results[selected_experiment]

        labels = ["negative", "neutral", "positive"]
        valid_labels = {"positive", "negative", "neutral"}
        valid_rows = (
            inspected["label"].isin(valid_labels)
            & inspected["prediction"].isin(valid_labels)
        )
        evaluated = inspected.loc[valid_rows].copy()

        st.subheader("Confusion Matrix")
        if evaluated.empty:
            st.info("There are no valid predictions to evaluate.")
        else:
            matrix = pd.crosstab(
                evaluated["label"],
                evaluated["prediction"],
            ).reindex(
                index=labels,
                columns=labels,
                fill_value=0,
            )
            st.dataframe(matrix, use_container_width=True)

        st.subheader("Incorrect Predictions")
        wrong = evaluated[
            evaluated["label"] != evaluated["prediction"]
        ]

        if wrong.empty:
            st.success("No incorrect predictions in this experiment.")
        else:
            display_columns = [
                column for column in
                ["id", "comment", "label", "prediction"]
                if column in wrong.columns
            ]
            st.dataframe(
                wrong[display_columns],
                hide_index=True,
                use_container_width=True,
            )

    else:
        st.warning(
            "No readable evaluation results were found. "
            "Check that the CSV files exist and contain 'label' "
            "and 'prediction' columns."
        )

elif task == "Task 7 — Semantic Search":
    import numpy as np
    from sentence_transformers import SentenceTransformer

    from semantic_search import (
        DATA_PATH,
        EMBEDDINGS_PATH,
        MODEL_NAME,
    )

    st.header("Task 7: Semantic Search")
    st.write("Search customer comments by meaning, not only exact words.")

    @st.cache_resource
    def load_search_data():
        df = pd.read_csv(DATA_PATH)
        comments = df["comment"].fillna("").tolist()
        model = SentenceTransformer(MODEL_NAME)

        if EMBEDDINGS_PATH.exists():
            embeddings = np.load(EMBEDDINGS_PATH)
            if embeddings.shape != (len(comments), 384):
                embeddings = model.encode(
                    comments,
                    normalize_embeddings=True,
                )
                np.save(EMBEDDINGS_PATH, embeddings)
        else:
            embeddings = model.encode(
                comments,
                normalize_embeddings=True,
            )
            np.save(EMBEDDINGS_PATH, embeddings)

        return df, model, embeddings

    from semantic_search import semantic_search

    df, model, embeddings = load_search_data()

    query = st.text_input("Enter your search query")
    top_k = st.slider("Number of results", min_value=1, max_value=10, value=5)
    min_score = st.slider(
        "Minimum similarity",
        min_value=0.0,
        max_value=1.0,
        value=0.30,
        step=0.05,
    )

    if query.strip():
        results = semantic_search(
            query,
            df,
            embeddings,
            model,
            top_k=top_k,
            min_score=min_score,
        )

        if results.empty:
            st.info("No good match found.")
        else:
            results = results.rename(
                columns={"similarity": "score"}
            )
            st.dataframe(results, use_container_width=True)
    else:
        st.caption("Enter a query to search the comments.")

# --------------------------------------------------
# Task 8 — Docs Search
# --------------------------------------------------
elif task == "Task 8 — Docs Search":

    from store import build_vector_store, hybrid_search

    st.header("Task 8: Poetry Documentation Search")
    st.write("Search Poetry documentation using semantic and keyword search.")

    @st.cache_resource
    def load_docs_search():
        collection, model = build_vector_store(
            chunk_size=1500,
            overlap=300,
        )
        return collection, model

    with st.spinner("Loading Poetry documentation..."):
        collection, model = load_docs_search()

    st.success("Poetry documentation is ready!")

    query = st.text_input(
        "Enter your question about Poetry",
        placeholder="Example: Which command creates a new Poetry project?",
        key="docs_search_query",
    )

    top_k = st.slider(
        "Number of results",
        min_value=1,
        max_value=10,
        value=5,
        key="docs_top_k",
    )

    if st.button("Search Documentation", type="primary"):

        if not query.strip():
            st.warning("Please enter a question first.")

        else:
            with st.spinner("Searching Poetry documentation..."):
                results = hybrid_search(
                    query=query,
                    collection=collection,
                    model=model,
                    top_k=top_k,
                )

            st.subheader("Search Results")

            if not results:
                st.info("No matching documentation was found.")

            else:
                for index, result in enumerate(results, start=1):

                    title = result.get("title", "Untitled")
                    source = result.get("source", "Unknown")
                    score = result.get("rrf_score", 0)

                    with st.expander(
                        f"Result {index}: {title} — {source}",
                        expanded=(index == 1),
                    ):
                        st.write("**Source:**", source)
                        st.write("**Title:**", title)
                        st.write(
                            "**Hybrid Search Score:**",
                            round(float(score), 6),
                        )

                        st.divider()

                        st.write(result["text"])
