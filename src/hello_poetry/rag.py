
"""Build grounded prompts for the Poetry RAG chatbot."""

SYSTEM_PROMPT = (
    "You answer questions about Poetry, the Python dependency manager. "
    "Use the numbered sources below to answer. "
    "After each fact, add the number of the source in square brackets, like [1]. "
    "Keep the answer short, and include the command when there is one."
)

from functools import lru_cache

from sentence_transformers import SentenceTransformer

from hello_poetry.llm import ask_llm
from hello_poetry.store import query_collection

MODEL_NAME = "openai/gpt-oss-20b"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

@lru_cache(maxsize=1)
def get_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL_NAME)

def format_sources(results: list[dict]) -> str:
    """Format retrieved chunks as numbered sources."""

    formatted = []

    for index, result in enumerate(results, start=1):
        source = (
            f"[{index}]\n"
            f"Title: {result['title']}\n"
            f"File: {result['source']}\n"
            f"Text:\n{result['text']}"
        )
        formatted.append(source)

    return "\n\n".join(formatted)


def build_messages(
    question: str,
    results: list[dict],
) -> tuple[str, str]:
    """Build the system prompt and user message."""

    sources_text = format_sources(results)

    user_message = (
        f"Sources:\n{sources_text}\n\n"
        f"Question: {question}"
    )

    return SYSTEM_PROMPT, user_message


def answer_question(
    question: str,
    collection_name: str,
    k: int = 3,
) -> dict:
    """Retrieve relevant chunks and generate a grounded answer."""

    if k < 1:
        raise ValueError("k must be at least 1.")

    if not question.strip():
        raise ValueError("Question cannot be empty.")

    embedding_model = get_embedding_model()

    results = query_collection(
        name=collection_name,
        query=question,
        model=embedding_model,
        k=k,
    )

    system_prompt, user_message = build_messages(
        question,
        results,
    )

    if not results:
        return {
            "answer": "No relevant documentation was found.",
            "sources": [],
            "prompt": (
                f"SYSTEM:\n{system_prompt}\n\n"
                f"USER:\n{user_message}"
            ),
        }

    answer = ask_llm(
        system_prompt=system_prompt,
        user_message=user_message,
        model=MODEL_NAME,
        temperature=0,
    )

    return {
        "answer": answer,
        "sources": results,
        "prompt": (
            f"SYSTEM:\n{system_prompt}\n\n"
            f"USER:\n{user_message}"
        ),
    }


