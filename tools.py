
from pathlib import Path

from store import build_vector_store, hybrid_search

BASE_DIR = Path(__file__).resolve().parent
DOCS_PATH = BASE_DIR / "data" / "poetry_docs"

# Limits to control the amount of text sent to the LLM
MAX_SEARCH_RESULTS = 3
MAX_RESULT_CHARS = 1200
MAX_READ_CHARS = 6000

_collection = None
_model = None


def _get_store():
    """
    Initialize the vector store once and reuse it.
    """
    global _collection, _model

    if _collection is None or _model is None:
        _collection, _model = build_vector_store()

    return _collection, _model

def expand_search_query(query: str) -> str:
    normalized = query.casefold()

    # Convert the legacy verification option to the current Poetry syntax.
    if (
        "poetry" in normalized
        and "show" in normalized
        and any(option in normalized for option in ("--dev", "-d"))
    ):
        return "poetry show --only dev"

    # Preserve explicit options in other targeted queries.
    if any(
        option in normalized
        for option in ("--dev", "--group", "--only")
    ):
        return query

    # Verification must be checked before adding.
    verification_terms = (
        "verify",
        "verification",
        "check",
        "confirm",
        "installed",
        "installation",
        "added correctly",
    )

    if "poetry" in normalized and any(
        term in normalized for term in verification_terms
    ):
        return "poetry show list installed packages"

    is_add = any(
        word in normalized
        for word in ("add", "adding")
    )

    is_development = any(
        word in normalized
        for word in ("development", "dev")
    )

    if "poetry" in normalized and is_add and is_development:
        return (
            "poetry add package --group dev "
            "development dependency"
        )

    return query

def get_search_queries(query: str) -> list[str]:
    normalized = query.casefold()

    verification_terms = (
        "verify",
        "verification",
        "check",
        "confirm",
        "installed",
        "installation",
        "added correctly",
    )

    has_verification = (
        "poetry" in normalized
        and any(term in normalized for term in verification_terms)
    )

    has_add = (
        "poetry" in normalized
        and any(word in normalized for word in ("add", "adding"))
        and any(word in normalized for word in ("development", "dev"))
    )

    if has_add and has_verification:
        return [
            "poetry add --dev -D --group dev development dependency",
            "poetry show --only dev list installed development packages",
        ]

    return [expand_search_query(query)]


def search_docs(query: str, k: int = 3):
    """
    Search Poetry documentation and return limited relevant results.
    """
    collection, model = _get_store()

    # Enforce a maximum number of search results
    k = max(1, min(k, MAX_SEARCH_RESULTS))

    queries = get_search_queries(query)
    search_results = []

    for search_query in queries:
        results = hybrid_search(
            query=search_query,
            collection=collection,
            model=model,
            top_k=k,
        )
        search_results.append(results)

       # Merge results and keep the strongest evidence.
    merged_results = {}

    for results in search_results:
        for rank, item in enumerate(results, start=1):
            key = (item["source"], item["text"])

            if key not in merged_results:
                merged_results[key] = {
                    **item,
                    "merge_score": 0,
                }

            merged_results[key]["merge_score"] += 1 / (60 + rank)

    # Prioritize direct command documentation.
    for item in merged_results.values():
        text = item["text"].casefold()
        bonus = 0

        if "--dev (-d)" in text or "--dev" in text:
            bonus += 0.03

        if "poetry show" in text and "--only" in text:
            bonus += 0.03

        if "poetry add" in text and "--group" in text:
            bonus += 0.02

        item["merge_score"] += bonus

    merged_results = sorted(
        merged_results.values(),
        key=lambda item: item["merge_score"],
        reverse=True,
    )[:k]
    limited_results = []

    for item in merged_results:
        text = str(item.get("text") or "")
        truncated = len(text) > MAX_RESULT_CHARS

        limited_results.append({
            "source": item["source"],
            "title": item["title"],
            "similarity": item["similarity"],
            "text": text[:MAX_RESULT_CHARS],
            "truncated": truncated,
        })

    return limited_results

def read_doc(
    source: str,
    start: int = 0,
    max_chars: int = MAX_READ_CHARS,
):
    """
    Read a specific portion of a Poetry documentation file.
    """
    # Validate reading parameters
    if start < 0:
        return {"error": "start must be zero or greater."}

    if max_chars < 1:
        return {"error": "max_chars must be greater than zero."}

    # Enforce the maximum reading limit
    max_chars = min(max_chars, MAX_READ_CHARS)

    docs_folder = DOCS_PATH.resolve()
    file_path = (docs_folder / source).resolve()

    # Prevent access outside the documentation folder
    if not file_path.is_relative_to(docs_folder):
        return {"error": "Invalid file path."}

    if not file_path.is_file():
        return {"error": f"Document not found: {source}"}

    try:
        content = file_path.read_text(encoding="utf-8")

        end = min(start + max_chars, len(content))
        selected_content = content[start:end]
        truncated = end < len(content)

        return {
            "source": source,
            "content": selected_content,
            "start": start,
            "end": end,
            "total_chars": len(content),
            "truncated": truncated,
        }

    except OSError as error:
        return {"error": str(error)}


def list_docs():
    """
    Return the names of available Poetry documentation files.
    """
    if not DOCS_PATH.is_dir():
        return {"error": "Documentation folder not found."}

    return sorted(
        file.name
        for file in DOCS_PATH.glob("*.md")
        if file.is_file()
    )


TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "search_docs",
            "description": (
                "Search Poetry documentation for information "
                "relevant to the user's question. "
                "Returns up to 3 relevant results with limited text."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The question or topic to search for."
                    },
                    "k": {
                        "type": "integer",
                        "description": "Maximum number of results to return.",
                        "default": 3
                    }
                },
                "required": ["query"]
            }
        }
    },
   {
    "type": "function",
    "function": {
        "name": "read_doc",
        "description": (
            "Read a specific portion of a Poetry documentation file. "
            "The maximum reading size is 6000 characters per call. "
            "If truncated is true, more content is available."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "source": {
                    "type": "string",
                    "description": "The documentation filename, such as basic-usage.md."
                },
                "start": {
                    "type": "integer",
                    "description": "Character position to start reading from. Defaults to 0."
                },
                "max_chars": {
                    "type": "integer",
                    "description": "Number of characters to read, maximum 6000. Defaults to 6000."
                }
            },
            "required": ["source"]
        }
    }
},
    {
        "type": "function",
        "function": {
            "name": "list_docs",
            "description": (
                "List all available Poetry documentation files."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    }
]


TOOL_FUNCTIONS = {
    "search_docs": search_docs,
    "read_doc": read_doc,
    "list_docs": list_docs,
}
