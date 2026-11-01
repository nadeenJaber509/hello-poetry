"""Explore the Poetry documentation before preprocessing."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DOCS_DIR = PROJECT_ROOT / "data" / "poetry_docs"


def explore_documents(folder: Path) -> list[dict]:
    """Collect basic statistics for each Markdown document."""
    documents = []

    for file_path in sorted(folder.glob("*.md")):
        text = file_path.read_text(encoding="utf-8")

        documents.append(
            {
                "source": file_path.name,
                "characters": len(text),
                "lines": len(text.splitlines()),
                "text": text,
            }
        )

    return sorted(
        documents,
        key=lambda document: document["characters"],
        reverse=True,
    )


def main() -> None:
    if not DOCS_DIR.is_dir():
        raise FileNotFoundError(f"Documentation directory not found: {DOCS_DIR}")

    documents = explore_documents(DOCS_DIR)

    if not documents:
        raise ValueError("No Markdown documents found.")

    print(f"{'File':40} {'Characters':>12} {'Lines':>10}")
    print("-" * 65)

    for document in documents:
        print(
            f"{document['source']:40} "
            f"{document['characters']:>12} "
            f"{document['lines']:>10}"
        )

    total_characters = sum(document["characters"] for document in documents)

    print(f"\nTotal files: {len(documents)}")
    print(f"Total characters: {total_characters:,}")

    longest = documents[0]

    print(f"\nLongest document: {longest['source']}")
    print("First 15 lines:\n")

    for line in longest["text"].splitlines()[:15]:
        print(line)

    print("\nTwo longest documents:")
    for document in documents[:2]:
        print(f"- {document['source']}")


if __name__ == "__main__":
    main()
