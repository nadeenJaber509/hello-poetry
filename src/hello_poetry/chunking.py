
"""Split documents into overlapping text chunks."""


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into chunks with a fixed character overlap."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")

    if overlap < 0:
        raise ValueError("overlap cannot be negative.")

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size.")

    if not text:
        return []

    chunks = []
    step = chunk_size - overlap

    for start in range(0, len(text), step):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])

        if end == len(text):
            break

    return chunks


def chunk_documents(
    documents: list[dict],
    chunk_size: int,
    overlap: int,
) -> list[dict]:
    """Split documents and preserve their source metadata."""
    all_chunks = []

    for document in documents:
        text_chunks = chunk_text(
            document["text"],
            chunk_size,
            overlap,
        )

        for index, text in enumerate(text_chunks):
            all_chunks.append({
                "id": f"{document['source']}::{index}",
                "text": text,
                "source": document["source"],
                "title": document["title"],
                "chunk_index": index,
            })

    return all_chunks
