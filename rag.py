
from store import build_vector_store, hybrid_search
from task6.llm import ask_llm

MODEL = "openai/gpt-oss-20b"

SYSTEM_PROMPT = """You answer questions about Poetry, the Python dependency manager.

Use only information explicitly supported by the provided sources.
Do not use outside knowledge or invent commands.

If the sources do not contain enough information to answer,
clearly say that you cannot answer based on the available documentation.

After each fact, add the number of the source in square brackets, like [1].
Keep the answer short, and include the command when there is one."""


def format_sources(results):
    """Format retrieved chunks as numbered sources."""

    formatted = []

    for index, result in enumerate(results, start=1):
        source = (
            f"[{index}]\n"
            f"Title: {result['title']}\n"
            f"File: {result['source']}\n"
            f"Text: {result['text']}"
        )

        formatted.append(source)

    return "\n\n".join(formatted)


def build_messages(question, results):
    """Build the system prompt and user message."""

    sources_text = format_sources(results)

    user_message = (
        f"Sources:\n{sources_text}\n\n"
        f"Question: {question}"
    )

    return SYSTEM_PROMPT, user_message


def answer_question(question, collection, model, k=3):
    """Retrieve relevant chunks and generate an answer."""

    results = hybrid_search(
        query=question,
        collection=collection,
        model=model,
        top_k=k,
    )
    if not results:
        return {
            "answer": (
                "I could not find relevant information "
                "in the available Poetry documentation."
            ),
            "sources": [],
            "prompt": None,
        }
    system_prompt, user_message = build_messages(
        question,
        results,
    )

    answer = ask_llm(
        system_prompt=system_prompt,
        user_message=user_message,
        model=MODEL,
        temperature=0,
    )

    sources = [
        {
            "title": result["title"],
            "source": result["source"],
            "similarity": result.get("similarity"),
            "text": result["text"],
        }
        for result in results
    ]

    return {
        "answer": answer,
        "sources": sources,
        "prompt": {
            "system": system_prompt,
            "user": user_message,
        },
    }


if __name__ == "__main__":
    collection, embedding_model = build_vector_store()

    question = "How can I run a Python script inside the project's Poetry environment?"

    result = answer_question(
        question=question,
        collection=collection,
        model=embedding_model,
        k=3,
    )

    print("\n--- ANSWER ---")
    print(result["answer"])

    print("\n--- SOURCES ---")
    for index, source in enumerate(result["sources"], start=1):
        print(f"\n[{index}] {source['title']}")
        print("File:", source["source"])
        print("Similarity:", source["similarity"])

    print("\n--- FULL PROMPT ---")
    print(result["prompt"]["system"])
    print(result["prompt"]["user"])
