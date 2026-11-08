
"""Tools for searching and reading Poetry documentation."""

from pathlib import Path

import chromadb

from hello_poetry.rag import get_embedding_model
from hello_poetry.store import query_collection

ROOT = Path(__file__).resolve().parents[2]
DOCS_DIR = ROOT / "data" / "poetry_docs"
DB_DIR = ROOT / "chroma_db"
COLLECTION_NAME = "poetry_800_150"


def search_docs(query: str, k: int = 3) -> list[dict]:
    """Search for relevant passages in the Poetry documentation."""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Search query must not be empty.")

    if not isinstance(k, int) or isinstance(k, bool) or not 1 <= k <= 10:
        raise ValueError("k must be an integer between 1 and 10.")

    client = chromadb.PersistentClient(path=str(DB_DIR))
    collection = client.get_collection(name=COLLECTION_NAME)

    if collection.count() == 0:
        raise ValueError("The documentation collection is empty.")

    model = get_embedding_model()

    results = query_collection(
        name=COLLECTION_NAME,
        query=query.strip(),
        model=model,
        k=k,
        path=DB_DIR,
    )

    return [
        {
            "source": result["source"],
            "title": result["title"],
            "similarity": float(result["similarity"]),
            "text": result["text"],
        }
        for result in results
    ]


def read_doc(source: str) -> str:
    """Read up to 6000 characters from one documentation file."""
    if not isinstance(source, str) or not source.strip():
        return "Error: Please provide a document name."

    docs_root = DOCS_DIR.resolve()
    path = (docs_root / source).resolve()

    if not path.is_relative_to(docs_root):
        return "Error: Access outside the documentation folder is not allowed."

    if not path.is_file():
        return f"Error: Document '{source}' was not found."

    try:
        return path.read_text(encoding="utf-8")[:6000]
    except (OSError, UnicodeError) as error:
        return f"Error: Could not read document: {error}"


def list_docs() -> list[str]:
    """Return available Poetry documentation file names."""
    if not DOCS_DIR.is_dir():
        return []

    return sorted(
        str(path.relative_to(DOCS_DIR))
        for path in DOCS_DIR.rglob("*")
        if path.is_file()
    )


TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "search_docs",
            "description": (
                "Search Poetry documentation by meaning. "
                "Use this to find relevant passages for a question."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The documentation search query.",
                    },
                    "k": {
                        "type": "integer",
                        "description": "Number of results (1 to 10).",
                        "default": 3,
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_doc",
            "description": (
                "Read a specific Poetry documentation file. "
                "Use when more detail is needed from a known source."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "source": {
                        "type": "string",
                        "description": "Document path inside poetry_docs.",
                    },
                },
                "required": ["source"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_docs",
            "description": (
                "List available Poetry documentation files. "
                "Use to discover document names before reading."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
]

TOOL_FUNCTIONS = {
    "search_docs": search_docs,
    "read_doc": read_doc,
    "list_docs": list_docs,
}
