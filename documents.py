
from pathlib import Path
import re


def load_documents(folder):
    documents = []
    folder = Path(folder)

    for file_path in sorted(folder.glob("*.md")):
        content = file_path.read_text(encoding="utf-8")

        title = file_path.stem
        text = content

        # Extract YAML frontmatter when present.
        frontmatter_match = re.match(
            r"\A---\s*\n(.*?)\n---\s*(?:\n|\Z)",
            content,
            flags=re.DOTALL,
        )

        if frontmatter_match:
            frontmatter = frontmatter_match.group(1)

            title_match = re.search(
                r"""^title:\s*["']?(.*?)["']?\s*$""",
                frontmatter,
                flags=re.MULTILINE,
            )

            if title_match:
                title = title_match.group(1)

            text = content[frontmatter_match.end():]

        # Keep paragraph breaks but remove excessive blank lines.
        text = re.sub(r"\n[ \t]*\n(?:[ \t]*\n)+", "\n\n", text)
        text = text.strip()

        documents.append({
            "source": file_path.name,
            "title": title,
            "text": text,
        })

    return documents


if __name__ == "__main__":
    docs = load_documents("data/poetry_docs")

    print("Number of documents:", len(docs))

    for doc in docs[:2]:
        print("\nSource:", doc["source"])
        print("Title:", doc["title"])
        print("First 200 characters:", doc["text"][:200])
