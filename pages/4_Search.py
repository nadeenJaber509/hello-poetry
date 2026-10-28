
import streamlit as st
from sentence_transformers import SentenceTransformer

from search import (
    MODEL_NAME,
    get_embeddings,
    load_comments,
    search,
)

st.set_page_config(
    page_title="Semantic Search",
    page_icon="🔎",
    layout="wide",
)

st.title("Semantic Search")
st.write("Search customer comments by meaning, not only keywords.")


@st.cache_resource
def load_model():
    return SentenceTransformer(MODEL_NAME)


@st.cache_data
def load_search_data():
    df = load_comments()
    return df


@st.cache_resource
def load_cached_embeddings():
    df = load_comments()
    comments = df["comment"].fillna("").tolist()

    model = load_model()
    return get_embeddings(comments, model)


df = load_search_data()
comments = df["comment"].fillna("").tolist()

model = load_model()
embeddings = load_cached_embeddings()

query = st.text_input(
    "Enter your search query",
    placeholder="e.g. money back problem",
)

k = st.slider(
    "Number of results (Top-k)",
    min_value=1,
    max_value=10,
    value=5,
)

min_score = st.slider(
    "Minimum similarity score",
    min_value=0.0,
    max_value=1.0,
    value=0.30,
    step=0.05,
)

if st.button("Search", type="primary"):
    if not query.strip():
        st.warning("Please enter a search query.")
    else:
        results = search(
            query,
            comments,
            embeddings,
            model,
            k=k,
        )

        filtered_results = [
            result
            for result in results
            if result["score"] >= min_score
        ]

        if not filtered_results:
            st.info("No good match found.")
        else:
            table_data = []

            for result in filtered_results:
                row = df.iloc[result["index"]]

                table_data.append({
                    "ID": row["id"],
                    "Comment": result["comment"],
                    "Label": row["label"],
                    "Similarity": round(result["score"], 4),
                })

            st.dataframe(
                table_data,
                use_container_width=True,
                hide_index=True,
            )
