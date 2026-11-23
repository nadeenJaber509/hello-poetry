

from agent import run_agent
from store import build_vector_store, hybrid_search

MODEL = "openai/gpt-oss-20b"

MAX_SEARCH_RESULTS = 3
MAX_RESULT_CHARS = 2500

SYSTEM_PROMPT = """You answer questions about Poetry, the Python dependency manager.

Use only information explicitly supported by the provided sources.
Do not use outside knowledge or invent commands.

Call search_docs once for each user question.
After receiving search results, answer using those results.

If the sources do not contain enough information to answer,
clearly say that you cannot answer based on the available documentation.

After each fact, add the source number in square brackets, like [1].
Keep the answer short and include commands when available.

Never invent source numbers.
Do not include unsupported facts.
"""


TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "search_docs",
            "description": (
                "Search Poetry documentation for information "
                "relevant to the user's question."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query.",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of results to retrieve.",
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    }
]


def search_docs(query, collection, model, k=3):
    """Search Poetry documentation and return limited results."""

    if not isinstance(query, str) or not query.strip():
        raise ValueError("Search query must be a non-empty string.")

    k = max(1, min(k, MAX_SEARCH_RESULTS))

    results = hybrid_search(
        query=query,
        collection=collection,
        model=model,
        top_k=k,
    )

    limited_results = []

    for item in results[:k]:
        text = str(item.get("text") or "")

        limited_results.append({
            "source": item["source"],
            "title": item["title"],
            "similarity": item["similarity"],
            "text": text[:MAX_RESULT_CHARS],
            "truncated": len(text) > MAX_RESULT_CHARS,
        })

    return limited_results


def answer_question(question, collection, model, k=3):
    """
    Retrieve information and generate an answer using Agent Loop.

    model is the embedding model.
    The LLM model is configured separately using MODEL.
    """

    retrieved_sources = []

    def search_docs_tool(query, top_k=3):
        """Tool wrapper that gives Agent Loop access to the store."""

        results = search_docs(
            query=query,
            collection=collection,
            model=model,
            k=top_k,
        )

        # Keep the original results for the application response.
        for source in results:
            retrieved_sources.append({
                **source,
                "source_number": len(retrieved_sources) + 1,
            })

        return retrieved_sources[-len(results):] if results else []

    tool_functions = {
        "search_docs": search_docs_tool,
    }

    result = run_agent(
        question=question,
        system_prompt=SYSTEM_PROMPT,
        tool_schemas=TOOL_SCHEMAS,
        tool_functions=tool_functions,
        model=MODEL,
        max_steps=5,
    )

    return {
        "answer": result["final_answer"],
        "sources": retrieved_sources,
        "prompt": {
            "system": SYSTEM_PROMPT,
            "user": None,
        },
        "steps": result["steps"],
        "total_tokens": result["total_tokens"],
        "stopped_by_limit": result["stopped_by_limit"],
    }


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


if __name__ == "__main__":
    collection, embedding_model = build_vector_store()

    question = "How can I run a Python script using Poetry?"

    result = answer_question(
        question=question,
        collection=collection,
        model=embedding_model,
        k=3,
    )

    print("\n--- ANSWER ---")
    print(result["answer"])

    print("\n--- SOURCES ---")

    for source in result["sources"]:
        print(f"\n[{source['source_number']}] {source['title']}")
        print("File:", source["source"])
        print("Similarity:", source["similarity"])

    print("\n--- AGENT STEPS ---")

    for step in result["steps"]:
        print(step)

    print("\n--- USAGE ---")
    print("Total tokens:", result["total_tokens"])
    print("Stopped by limit:", result["stopped_by_limit"])
