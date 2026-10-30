from documents import load_documents
def chunk_text(text, chunk_size, overlap):
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")

    if overlap < 0:
        raise ValueError("overlap cannot be negative")

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    text = text.strip()

    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks

def chunk_documents(documents, chunk_size, overlap):
    chunked_documents = []

    for document in documents:
        chunks = chunk_text(
            document["text"],
            chunk_size,
            overlap,
        )

        for chunk in chunks:
            chunked_documents.append({
                "source": document["source"],
                "title": document["title"],
                "text": chunk,
            })

    return chunked_documents
if __name__ == "__main__":
    documents = load_documents("data/poetry_docs")

    chunks = chunk_documents(
        documents,
        chunk_size=300,
        overlap=60,
    )

    print("Number of documents:", len(documents))
    print("Number of chunks:", len(chunks))

    for index, chunk in enumerate(chunks[:3], start=1):
        print(f"\nChunk {index}")
        print("Source:", chunk["source"])
        print("Title:", chunk["title"])
        print("Length:", len(chunk["text"]))
        print("Text:", chunk["text"][:150])