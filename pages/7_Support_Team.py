"""Poetry Support Team page with the full agent trace."""

import os
from pathlib import Path

import chromadb
import streamlit as st
from dotenv import load_dotenv
from groq import GroqError
from streamlit.errors import StreamlitSecretNotFoundError

from hello_poetry.chunking import chunk_documents
from hello_poetry.coordinator import answer_with_team
from hello_poetry.documents import load_documents
from hello_poetry.rag import get_embedding_model
from hello_poetry.store import build_collection

ROOT = Path(__file__).resolve().parents[1]
DB_DIR = ROOT / "chroma_db"
COLLECTION = "poetry_800_150"


@st.cache_resource
def ensure_collection():
    client = chromadb.PersistentClient(path=str(DB_DIR))
    try:
        if client.get_collection(COLLECTION).count():
            return
    except chromadb.errors.NotFoundError:
        pass
    documents = load_documents(ROOT / "data" / "poetry_docs")
    chunks = chunk_documents(documents, 800, 150)
    if not chunks:
        raise ValueError("Poetry documentation is missing or empty.")
    build_collection(COLLECTION, chunks, get_embedding_model(), path=DB_DIR)


st.set_page_config(
    page_title="Support Team | AI Learning Lab", page_icon="🤖", layout="wide"
)
st.title("Poetry Support Team")
st.write("Ask a question about Poetry and inspect how the agents reached their answer.")
question = st.text_area("Question", placeholder="How do I add a dependency?")

if st.button("Ask the team"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        try:
            load_dotenv(ROOT / ".env")
            if not os.getenv("GROQ_API_KEY"):
                try:
                    secret = st.secrets.get("GROQ_API_KEY")
                except StreamlitSecretNotFoundError:
                    secret = None
                if secret:
                    os.environ["GROQ_API_KEY"] = secret
            if not os.getenv("GROQ_API_KEY"):
                raise ValueError("GROQ_API_KEY is not configured.")
            with st.spinner("Checking the documentation..."):
                ensure_collection()
                result = answer_with_team(question.strip())
            st.subheader("Answer")
            st.write(result["answer"])
            for command in result["commands"]:
                st.code(command, language="bash")
            st.write("Sources:", ", ".join(result["sources"]) or "None")
            st.write("Confidence:", result["confidence"])
            if result["blocked_by"]:
                st.info(f"Blocked by: {result['blocked_by']}")
            with st.expander("Trace"):
                for step in result["trace"]:
                    st.markdown(f"**{step['agent'].title()}**")
                    st.write("Input:", step.get("input", ""))
                    st.write("Output:", step.get("output", ""))
                    st.caption(
                        f"Tokens: {step.get('tokens', 0)} · Seconds: {step.get('seconds', 0):.2f}"
                    )
                st.write("Model calls:", result["model_calls"])
                st.write("Total tokens:", result["total_tokens"])
                st.write("Total seconds:", round(result["total_seconds"], 2))
        except (GroqError, ValueError, RuntimeError, OSError) as exc:
            st.error(f"The team could not complete this request: {exc}")
