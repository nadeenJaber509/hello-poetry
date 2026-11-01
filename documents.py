
"""Load and clean Markdown documentation files."""

import re
from pathlib import Path


def clean_blank_lines(text: str) -> str:
    """Replace repeated blank lines with a single blank line."""
    text = text.strip()
    return re.sub(r"\n[ \t]*\n(?:[ \t]*\n)+", "\n\n", text)


def extract_front_matter(text: str, filename: str) -> tuple[str, str]:
    """Extract the document title and remove YAML front matter."""
    title = filename

    lines = text.splitlines()

    if not lines or lines[0].strip() != "---":
        return title, text

    # Find the closing --- delimiter.
    for index in range(1, len(lines)):
        if lines[index].strip() == "---":
            header_lines = lines[1:index]
            content = "\n".join(lines[index + 1:])

            for line in header_lines:
                if line.startswith("title:"):
                    title = line.split(":", 1)[1].strip().strip("\"'")
                    break

            return title, content

    # No closing delimiter: keep the original text.
    return title, text


def load_documents(folder: str | Path) -> list[dict]:
    """Load Markdown files with source, title, and cleaned text."""
    folder = Path(folder)

    if not folder.is_dir():
        raise FileNotFoundError(
            f"Documentation folder not found: {folder}"
        )

    documents = []

    for file_path in sorted(folder.glob("*.md")):
        raw_text = file_path.read_text(encoding="utf-8")

        title, content = extract_front_matter(
            raw_text,
            file_path.name,
        )

        documents.append({
            "source": file_path.name,
            "title": title,
            "text": clean_blank_lines(content),
        })

    return documents
