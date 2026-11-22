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

with st.expander("How Our AI Team Works"):
    st.markdown(
        """
        <style>
        .team-flow {
            display: flex;
            align-items: stretch;
            gap: 0.3rem;
            margin: 0.4rem 0 1rem;
        }
        .team-flow__card {
            flex: 1 1 0;
            min-width: 0;
            min-height: 132px;
            padding: 0.75rem 0.65rem;
            border: 1px solid #DFE4E8;
            border-top: 3px solid #547998;
            border-radius: 12px;
            background: #FFFFFF;
            box-shadow: 0 3px 12px rgba(18, 38, 64, 0.04);
        }
        .team-flow__card--end { border-top-color: #678B74; }
        .team-flow__number {
            display: block;
            margin-bottom: 0.35rem;
            color: #66758A;
            font-size: 0.68rem;
            font-weight: 700;
            letter-spacing: 0.09em;
        }
        .team-flow__card strong {
            display: block;
            color: #122640;
            font-size: 0.9rem;
            line-height: 1.2;
        }
        .team-flow__card span:last-child {
            display: block;
            margin-top: 0.45rem;
            color: #56677A;
            font-size: 0.78rem;
            line-height: 1.35;
        }
        .team-flow__arrow {
            align-self: center;
            flex: 0 0 auto;
            color: #8AA0B4;
            font-size: 1.1rem;
            font-weight: 700;
        }
        .team-flow__notes {
            display: flex;
            flex-wrap: wrap;
            gap: 0.6rem;
        }
        .team-flow__note {
            flex: 1 1 260px;
            padding: 0.65rem 0.85rem;
            border: 1px solid #D9E3EB;
            border-radius: 10px;
            background: #F3F7FA;
            color: #40566D;
            font-size: 0.82rem;
            line-height: 1.4;
        }
        .team-flow__note strong { color: #122640; }
        @media (max-width: 900px) {
            .team-flow { flex-direction: column; }
            .team-flow__card { flex: none; min-height: auto; }
            .team-flow__arrow {
                align-self: center;
                line-height: 1;
                transform: rotate(90deg);
            }
        }
        </style>
        <div class="team-flow" aria-label="Poetry Support Team workflow">
            <div class="team-flow__card">
                <span class="team-flow__number">START</span>
                <strong>User Question</strong>
                <span>Ask about Poetry.</span>
            </div>
            <span class="team-flow__arrow" aria-hidden="true">→</span>
            <div class="team-flow__card">
                <span class="team-flow__number">01 · ROUTE</span>
                <strong>Router</strong>
                <span>Classifies the question.</span>
            </div>
            <span class="team-flow__arrow" aria-hidden="true">→</span>
            <div class="team-flow__card">
                <span class="team-flow__number">02 · FIND</span>
                <strong>Researcher</strong>
                <span>Searches docs with tools and gathers evidence.</span>
            </div>
            <span class="team-flow__arrow" aria-hidden="true">→</span>
            <div class="team-flow__card">
                <span class="team-flow__number">03 · DRAFT</span>
                <strong>Writer</strong>
                <span>Creates a structured JSON answer from evidence.</span>
            </div>
            <span class="team-flow__arrow" aria-hidden="true">→</span>
            <div class="team-flow__card">
                <span class="team-flow__number">04 · REVIEW</span>
                <strong>Reviewer</strong>
                <span>Checks the draft and can request a rewrite.</span>
            </div>
            <span class="team-flow__arrow" aria-hidden="true">→</span>
            <div class="team-flow__card">
                <span class="team-flow__number">05 · VERIFY</span>
                <strong>Python Guardrails</strong>
                <span>Final citations and commands; earlier scope and evidence gates.</span>
            </div>
            <span class="team-flow__arrow" aria-hidden="true">→</span>
            <div class="team-flow__card team-flow__card--end">
                <span class="team-flow__number">RESULT</span>
                <strong>Final Answer</strong>
                <span>Supported answer or a clear refusal.</span>
            </div>
        </div>
        <div class="team-flow__notes">
            <div class="team-flow__note">
                <strong>Reviewer ↶ Writer:</strong> The Reviewer can send a draft back
                to the Writer for one revision.
            </div>
            <div class="team-flow__note">
                <strong>Router → Refusal:</strong> Out-of-scope questions stop after
                the Router. Insufficient evidence stops after the Researcher.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

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
