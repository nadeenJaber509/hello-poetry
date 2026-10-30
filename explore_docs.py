
from pathlib import Path


def main():
    docs_folder = Path("data/poetry_docs")

    # Find all Markdown files
    files = sorted(docs_folder.glob("*.md"))

    if not files:
        print("No Markdown files found.")
        return

    documents_info = []

    total_characters = 0

    for file_path in files:
        text = file_path.read_text(encoding="utf-8")

        characters = len(text)
        lines = len(text.splitlines())

        total_characters += characters

        documents_info.append({
            "name": file_path.name,
            "characters": characters,
            "lines": lines,
        })

    # Sort files from longest to shortest
    documents_info.sort(
        key=lambda item: item["characters"],
        reverse=True,
    )

    # Print the table
    print(f"{'File Name':<35} {'Characters':>12} {'Lines':>10}")
    print("-" * 60)

    for document in documents_info:
        print(
            f"{document['name']:<35} "
            f"{document['characters']:>12} "
            f"{document['lines']:>10}"
        )

    print("\nTotal files:", len(files))
    print("Total characters:", total_characters)

    # Print the first 15 lines of the longest file
    longest_file = docs_folder / documents_info[0]["name"]
    longest_text = longest_file.read_text(encoding="utf-8")

    print(f"\nFirst 15 lines of {longest_file.name}:")
    print("-" * 60)

    for line in longest_text.splitlines()[:15]:
        print(line)


if __name__ == "__main__":
    main()
