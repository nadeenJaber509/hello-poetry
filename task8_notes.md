# Task 8: Documents, Chunking, and Vector Database

## 1. Project Overview

The goal of this task is to build a document search system using Poetry documentation.

The system loads Markdown documents, cleans their content, splits them into smaller chunks, and stores their embeddings in ChromaDB.

It supports two search methods:

- **Semantic Search:** Finds content based on meaning.
- **Hybrid Search:** Combines semantic search and keyword search.

The main goal is to understand how chunk size, overlap, and search methods affect retrieval accuracy.

---

## 2. Document Exploration

I downloaded the Poetry documentation from the Poetry GitHub repository and extracted the Markdown files into:

`data/poetry_docs`

The folder contains:

- 16 Markdown files.
- 1 LICENSE file.
- 224,878 characters of documentation in total.

### Document Size Analysis

| Document | Characters | Lines |
|---|---:|---:|
| cli.md | 40,855 | 1,320 |
| pyproject.md | 25,048 | 931 |
| community.md | 654 | 30 |

The `cli.md` file is the largest document, while `community.md` is the smallest.

### Observations

- The documents have different sizes.
- Some documents contain long explanations and many sections.
- Markdown files contain headings, code examples, and technical terms.
- Searching large documents directly can make it difficult to find the exact information.

This is why document cleaning and chunking are important.

---

## 3. Document Loading and Cleaning

I created `documents.py` to load and clean the Markdown files.

The `load_documents()` function:

1. Reads all Markdown files from the documentation folder.
2. Extracts the title from YAML frontmatter.
3. Removes the frontmatter from the document content.
4. Removes unnecessary blank lines.
5. Returns the documents with their source, title, and cleaned text.

Each document has the following structure:

- `source`: Original file name.
- `title`: Document title.
- `text`: Cleaned document content.

### Why is this important?

Cleaning removes unnecessary information and makes the content easier to process.

Keeping the source and title is also important because the search system needs to identify where each result came from.

### Result

The loader successfully loaded 16 Markdown documents.

---

## 4. Chunking

I created `chunking.py` to split documents into smaller pieces called chunks.

The implementation uses:

- `chunk_size`: Maximum number of characters in a chunk.
- `overlap`: Number of characters repeated between consecutive chunks.

The chunking function also preserves the original document metadata, including its source and title.

### What is Overlap?

Overlap means that the end of one chunk is repeated at the beginning of the next chunk.

For example, if the chunk size is 300 and the overlap is 60:

- The first chunk contains approximately 300 characters.
- The next chunk starts 240 characters after the previous starting position.
- Therefore, 60 characters are shared between the two chunks.

### Why use Overlap?

A useful sentence or explanation might be split between two chunks.

Overlap helps preserve some of the surrounding context and reduces the chance of losing useful information at chunk boundaries.

### Initial Result

Using a chunk size of 300 and an overlap of 60 produced 936 chunks from the 16 documents.

### Important Observation

The current implementation splits text by character count, not by sentences or paragraphs.

Therefore, a chunk may sometimes start or end in the middle of a sentence or word.

---

## 5. Vector Database

I used ChromaDB as the vector database.

The embedding model is:

`all-MiniLM-L6-v2`

This model converts text into numerical vectors called embeddings.

These vectors help the system compare the meaning of a user's question with the meaning of stored document chunks.

### Database Configuration

- Database: ChromaDB.
- Storage: Persistent local storage.
- Embedding model: `all-MiniLM-L6-v2`.
- Distance metric: Cosine distance.
- Batch size: 100 chunks.

### How Does It Work?

1. The document chunks are converted into embeddings.
2. Each chunk is stored in ChromaDB.
3. The original text and metadata are stored with the embedding.
4. When a user searches, the query is converted into an embedding.
5. ChromaDB compares the query embedding with the stored embeddings.
6. The system returns the most similar chunks.

### Similarity Score

The database uses cosine distance.

I calculate the similarity score using:

`similarity = 1 - cosine_distance`

A higher similarity score means that the vectors are closer according to cosine distance.

---

## 6. Questions Dataset

I created:

`data/questions.csv`

The dataset contains 30 questions.

It includes the following columns:

| Column | Description |
|---|---|
| question | The question asked by the user |
| expected_source | The document expected to contain the answer |
| key_phrase | An important phrase expected in the result |
| difficulty | Question difficulty |
| answerable | Whether the question has an answer in the documentation |

The dataset includes:

- 28 answerable questions.
- 2 unanswerable questions.
- 3 hard questions.

The two unanswerable questions are excluded from the retrieval accuracy calculations because they do not have an expected source to retrieve.

The dataset helps evaluate the search system using questions with known expected answers.

---

## 7. Chunking Experiments

I tested six different chunking configurations.

| Configuration | Chunk Size | Overlap |
|---|---:|---:|
| poetry_300_0 | 300 | 0 |
| poetry_300_60 | 300 | 60 |
| poetry_800_0 | 800 | 0 |
| poetry_800_150 | 800 | 150 |
| poetry_1500_0 | 1500 | 0 |
| poetry_1500_300 | 1500 | 300 |

For each configuration, I measured:

- Number of chunks.
- Average chunk length.
- Hit@1.
- Hit@3.
- Hit@5.

### What are Hit@1, Hit@3, and Hit@5?

- **Hit@1:** The expected result appears in the first search result.
- **Hit@3:** The expected result appears within the first three results.
- **Hit@5:** The expected result appears within the first five results.

A result counts as a hit when both conditions are satisfied:

1. Its source matches the expected source.
2. All normalized key phrase words appear in the returned chunk.

The key phrase words do not have to appear next to each other or in the original order.

---

## 8. Semantic Search Results

Semantic Search retrieves documents based on the similarity between embeddings.

The results from the six configurations were:

| Configuration | Chunks | Avg. Length | Hit@1 | Hit@3 | Hit@5 |
|---|---:|---:|---:|---:|---:|
| 300 / 0 | 752 | 296.23 | 25.00% | 57.14% | 60.71% |
| 300 / 60 | 936 | 296.90 | 39.29% | 64.29% | 67.86% |
| 800 / 0 | 286 | 779.37 | 35.71% | 60.71% | 71.43% |
| 800 / 150 | 350 | 779.79 | 32.14% | 71.43% | 82.14% |
| 1500 / 0 | 156 | 1429.36 | 46.43% | 67.86% | 71.43% |
| 1500 / 300 | 192 | 1434.14 | 39.29% | 82.14% | 82.14% |

### Observations

- Small chunks produced more chunks.
- Larger chunks produced fewer chunks but contained more context.
- Adding overlap improved some results.
- The 1500 / 300 configuration achieved the best Semantic Hit@3 and tied for the best Semantic Hit@5.
- The 1500 / 0 configuration achieved the best Semantic Hit@1.

This shows that chunk size and overlap can affect search accuracy differently depending on how many results are considered.

---

## 9. Hybrid Search

Semantic Search is useful for finding content based on meaning, but it may not always identify exact technical terms or commands.

For example, a question about `poetry add` may return content about `poetry install` because both commands are related to package installation.

To improve retrieval, I implemented Hybrid Search.

### Keyword Search

The keyword search function:

1. Normalizes the query.
2. Removes common stop words.
3. Compares the remaining query words with words in stored chunks.
4. Counts matching words.
5. Adds a bonus when the complete normalized query appears in the chunk.
6. Sorts the results by keyword score.

### Hybrid Search

Hybrid Search combines:

- Semantic Search results.
- Keyword Search results.

I used Reciprocal Rank Fusion (RRF) to combine the rankings.

The RRF formula is:

`RRF score = sum(1 / (60 + rank))`

A result appearing near the top of both rankings receives a higher combined score.

This allows the system to benefit from both semantic meaning and exact keyword matches.

---

## 10. Hybrid Search Results

I evaluated Hybrid Search using the same six chunking configurations.

| Configuration | Hit@1 | Hit@3 | Hit@5 |
|---|---:|---:|---:|
| 300 / 0 | 42.86% | 60.71% | 71.43% |
| 300 / 60 | 39.29% | 75.00% | 75.00% |
| 800 / 0 | 42.86% | 78.57% | 85.71% |
| 800 / 150 | 46.43% | 85.71% | 89.29% |
| 1500 / 0 | 39.29% | 71.43% | 78.57% |
| 1500 / 300 | 50.00% | 78.57% | 92.86% |

### Best Results

- Best Hit@1: 50.00% with 1500 / 300.
- Best Hit@3: 85.71% with 800 / 150.
- Best Hit@5: 92.86% with 1500 / 300.

The best Hit@5 result means that the expected source and key phrase were found within the first five results for 26 out of 28 answerable questions.

### Comparison with Semantic Search

Hybrid Search improved Hit@5 in all six configurations.

For example:

- With 800 / 150, Semantic Hit@5 was 82.14%, while Hybrid Hit@5 was 89.29%.
- With 1500 / 300, Semantic Hit@5 was 82.14%, while Hybrid Hit@5 was 92.86%.

However, Hybrid Search did not improve Hit@1 in every configuration.

This means combining search methods improves overall retrieval in these experiments, but it does not guarantee that the correct result will always be ranked first.

---

## 11. Failure Analysis

I examined the failed search results to understand why some questions were not answered correctly.

### Example 1: `poetry add`

Question:

"Which command adds a package and installs it with its dependencies?"

Expected source:

`basic-usage.md`

Expected key phrase:

`poetry add`

The search returned several chunks from `cli.md` discussing `poetry install`.

**Analysis:**

Both commands are related to installing packages, so Semantic Search may consider their content similar.

However, they are different commands with different purposes.

This shows why exact technical keywords are important.

### Example 2: `poetry new`

Question:

"Which command creates a new Poetry project?"

Expected source:

`basic-usage.md`

Expected key phrase:

`poetry new`

The expected chunk exists in the documentation, but it did not always appear near the top of the search results.

**Analysis:**

Finding semantically related content does not guarantee retrieving the exact chunk that contains the required command.

### Example 3: `requires-python`

Question:

"Where can the project's Python version requirement be declared in the project metadata?"

Expected source:

`pyproject.md`

Expected key phrase:

`requires-python`

In one of the Hybrid Search failure examples, `pyproject.md` appeared in the second result, but the returned chunk did not satisfy the key phrase condition.

**Analysis:**

Retrieving the correct document is not always enough. The returned chunk must also contain the information required to answer the question.

### Example 4: Exclusive Extras and Environment Markers

Question:

"How can an exclusive extra be used with environment markers when defining dependencies?"

Expected source:

`dependency-specification.md`

Expected key phrase:

`exclusive extras; environment markers`

Some results came from the expected document, but the required information could be spread across different chunks.

**Analysis:**

Chunk boundaries can separate related information. Overlap may help preserve context, but it does not solve every chunking problem.

### Example 5: Package Mode

Question:

"What does package-mode = false change about a Poetry project?"

Expected source:

`basic-usage.md`

Expected key phrase:

`non-package mode`

Some results discussed configuration, repositories, and plugins instead of the expected explanation.

**Analysis:**

The search may retrieve content related to configuration without retrieving the exact explanation needed by the question.

### Main Lessons from Failure Analysis

- Semantic similarity does not always mean the result contains the exact answer.
- Technical commands need accurate keyword matching.
- Retrieving the correct document does not guarantee retrieving the correct chunk.
- Chunk boundaries can separate related information.
- Hybrid Search improves retrieval but does not eliminate all failures.

---

## 12. Lessons Learned

During this task, I learned:

1. **Document Cleaning:** Cleaning Markdown documents makes them easier to process and search.

2. **Chunking:** Splitting documents into chunks allows the system to search smaller pieces instead of processing entire documents.

3. **Chunk Size:** Small chunks provide focused content, while large chunks preserve more context.

4. **Overlap:** Overlap reduces the risk of losing information at chunk boundaries, but it also increases the number of chunks.

5. **Embeddings:** Embeddings represent text as numerical vectors and allow semantic similarity search.

6. **Vector Databases:** ChromaDB stores embeddings and retrieves similar chunks efficiently.

7. **Semantic Search:** Semantic Search understands related meanings, but it can confuse similar technical concepts.

8. **Keyword Search:** Keyword Search helps identify exact words and technical commands.

9. **Hybrid Search:** Combining semantic and keyword search can improve retrieval accuracy.

10. **Evaluation:** Hit@1, Hit@3, and Hit@5 help measure how often the expected result appears near the top of the search results.

11. **Failure Analysis:** Examining incorrect results helps identify weaknesses that accuracy numbers alone cannot explain.

---

## 13. Current Project Status

### Completed

- Downloaded and explored Poetry documentation.
- Created `documents.py`.
- Created `chunking.py`.
- Installed and configured ChromaDB.
- Used the `all-MiniLM-L6-v2` embedding model.
- Created the questions dataset.
- Implemented Semantic Search.
- Implemented Keyword Search.
- Implemented Hybrid Search using RRF.
- Ran six chunking configurations.
- Compared Semantic and Hybrid Search results.
- Saved experiment results and failure reports.

### Remaining

- Analyze the failure reports further if needed.
- Create the Streamlit Docs Search page.
- Test the search interface manually.

The task is not fully completed until the Streamlit search page is implemented.