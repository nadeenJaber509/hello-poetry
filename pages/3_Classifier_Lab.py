
"""Educational Streamlit page for LLM text classification."""

import os
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from sklearn.metrics import confusion_matrix

from hello_poetry.llm import ask_llm, parse_label
from hello_poetry.prompts import PROMPT_A, PROMPT_B, PROMPT_C

st.set_page_config(
    page_title="LLM Classification | AI Learning Lab",
    layout="wide",
)

PROMPTS = {
    "A - Zero-Shot": PROMPT_A,
    "B - Few-Shot": PROMPT_B,
    "C - Improved": PROMPT_C,
}

LABELS = ["positive", "negative", "neutral", "invalid"]

RESULTS_FILES = {
    "Prompt A — 20B": "results_A_openai_gpt-oss-20b.csv",
    "Prompt B — 20B": "results_B_openai_gpt-oss-20b.csv",
    "Prompt C — 20B": "results_C_openai_gpt-oss-20b.csv",
    "Prompt C — 120B": "results_C_openai_gpt-oss-120b.csv",
}


# --------------------------------------------------
# Educational introduction
# --------------------------------------------------

st.caption("TASK 06 · LANGUAGE AI")
st.title("LLM Classification Lab")

st.write(
    "Classify customer comments using language models "
    "and compare how prompts and models affect the results."
)

with st.container(border=True):
    st.markdown("#### What is LLM Classification?")
    st.write(
        "LLM Classification uses a language model to assign "
        "a label to a piece of text. In this example, the "
        "model classifies customer comments as positive, "
        "negative, or neutral."
    )

col1, col2 = st.columns(2, gap="medium")

with col1, st.container(border=True):
    st.markdown("**Try it**")
    st.write(
        'Enter a comment such as "The delivery was fast '
        'and the product works perfectly." '
        "Then compare different prompts and models."
    )

with col2, st.container(border=True):
    st.markdown("**What to notice**")
    st.write(
        "See how the prediction changes with different "
        "prompt styles. Use the evaluation dashboard "
        "to compare accuracy and invalid outputs."
    )

st.divider()


# --------------------------------------------------
# 1. Live Classification
# --------------------------------------------------

st.subheader("1. Live Classification")

comment = st.text_area(
    "Customer Comment",
    placeholder="Write a customer comment here...",
    height=120,
)

col1, col2 = st.columns(2)

with col1:
    selected_prompt = st.selectbox(
        "Choose a Prompt",
        list(PROMPTS.keys()),
        index=2,
    )

with col2:
    selected_model = st.selectbox(
        "Choose a Model",
        ["openai/gpt-oss-20b", "openai/gpt-oss-120b"],
    )

if st.button("Classify Comment", type="primary"):
    if not comment.strip():
        st.warning("Please enter a customer comment.")
    else:
        try:
            load_dotenv()

            # Support both local .env and Streamlit Cloud secrets.
            if not os.getenv("GROQ_API_KEY"):
                try:
                    api_key = st.secrets.get("GROQ_API_KEY")
                except (FileNotFoundError, KeyError):
                    api_key = None

                if api_key:
                    os.environ["GROQ_API_KEY"] = api_key

            if not os.getenv("GROQ_API_KEY"):
                st.error(
                    "Groq API key is missing. "
                    "Please configure it first."
                )
            else:
                with st.spinner("Classifying comment..."):
                    response = ask_llm(
                        system_prompt=PROMPTS[selected_prompt],
                        user_message=comment,
                        model=selected_model,
                        temperature=0,
                    )

                prediction = parse_label(response)

                if prediction == "invalid":
                    st.warning(
                        "The model did not return a valid JSON label."
                    )
                    st.code(response)
                else:
                    st.success(
                        f"Predicted sentiment: {prediction.upper()}"
                    )

        except Exception as error:  # noqa: BLE001
            st.error(f"Classification failed: {error}")


st.divider()


# --------------------------------------------------
# 2. Evaluation Dashboard
# --------------------------------------------------

st.subheader("2. Evaluation Dashboard")

st.write(
    "Compare saved evaluation results without making "
    "new API calls."
)

available_results = {}

for display_name, filename in RESULTS_FILES.items():
    path = Path(filename)

    if path.exists():
        available_results[display_name] = path

if not available_results:
    st.info(
        "No evaluation results found. "
        "Run evaluate.py first."
    )

else:
    summary_rows = []

    for display_name, path in available_results.items():
        df = pd.read_csv(path)

        accuracy = (
            df["label"] == df["prediction"]
        ).mean()

        invalid_count = (
            df["prediction"] == "invalid"
        ).sum()

        summary_rows.append({
            "Configuration": display_name,
            "Accuracy": accuracy,
            "Invalid": invalid_count,
            "Total": len(df),
        })

    summary_df = pd.DataFrame(summary_rows)

    st.subheader("Accuracy Comparison")

    st.dataframe(
        summary_df.style.format({
            "Accuracy": "{:.2%}"
        }),
        width="stretch",
        hide_index=True,
    )

    st.bar_chart(
        summary_df.set_index("Configuration")[["Accuracy"]]
    )

    st.subheader("Detailed Results")

    selected_result = st.selectbox(
        "Select Evaluation",
        list(available_results.keys()),
    )

    results = pd.read_csv(
        available_results[selected_result]
    )

    accuracy = (
        results["label"] == results["prediction"]
    ).mean()

    invalid_count = (
        results["prediction"] == "invalid"
    ).sum()

    m1, m2, m3 = st.columns(3)

    m1.metric("Accuracy", f"{accuracy:.2%}")
    m2.metric("Invalid Predictions", int(invalid_count))
    m3.metric("Evaluated Comments", len(results))

    st.subheader("Confusion Matrix")

    matrix = confusion_matrix(
        results["label"],
        results["prediction"],
        labels=LABELS,
    )

    matrix_df = pd.DataFrame(
        matrix,
        index=[
            f"Actual {label}" for label in LABELS
        ],
        columns=[
            f"Predicted {label}" for label in LABELS
        ],
    )

    st.dataframe(
        matrix_df,
        width="stretch",
    )

    st.subheader("Wrong Predictions")

    wrong = results[
        results["label"] != results["prediction"]
    ]

    if wrong.empty:
        st.success(
            "All comments were classified correctly!"
        )
    else:
        st.dataframe(
            wrong,
            width="stretch",
            hide_index=True,
        )
