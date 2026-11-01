## Task 8 — Documentation Source

- Repository: https://github.com/python-poetry/poetry
- Version: 2.1.4
- Source directory: `data/poetry_docs/`
- License: `data/poetry_docs/LICENSE`
- Source commit: a8f0889a54a545ec4f7ceed7bf41f8c2a7677bbb        refs/tags/2.1.4

## Step 2 — Documentation Exploration

I created `explore_docs.py` to explore the Poetry 2.1.4 documentation before cleaning and chunking it.

The script reads all Markdown files, counts their characters and lines, and sorts them from longest to shortest.

### Dataset Overview

- **Total Markdown files:** 16
- **Total characters:** 224,878
- **Longest file:** `cli.md` — 40,855 characters, 1,320 lines
- **Second longest file:** `pyproject.md` — 25,048 characters, 931 lines
- **Shortest file:** `community.md` — 654 characters, 30 lines

### File Statistics

| File | Characters | Lines |
|---|---:|---:|
| cli.md | 40,855 | 1,320 |
| pyproject.md | 25,048 | 931 |
| dependency-specification.md | 23,224 | 784 |
| repositories.md | 20,559 | 592 |
| configuration.md | 18,740 | 611 |
| faq.md | 16,606 | 436 |
| _index.md | 13,289 | 412 |
| basic-usage.md | 13,247 | 300 |
| contributing.md | 12,970 | 272 |
| building-extension-modules.md | 9,874 | 377 |
| plugins.md | 8,335 | 293 |
| pre-commit-hooks.md | 5,957 | 181 |
| managing-dependencies.md | 5,924 | 186 |
| managing-environments.md | 5,503 | 209 |
| libraries.md | 4,093 | 135 |
| community.md | 654 | 30 |

### Initial Observations

The documentation files have different lengths, so a single embedding for an entire document would not be suitable for precise retrieval.

The first lines of `cli.md` contain a YAML front matter block:

```yaml
---
title: "Commands"
draft: false
type: docs
layout: single

menu:
  docs:
    weight: 30
---
```

This block contains document metadata rather than the main documentation content. During preprocessing, I should extract the title and remove the front matter from the searchable text while keeping the original source filename.

### Potential Retrieval Challenges

1. **Long documents:** Files such as `cli.md` contain many sections, so they need to be divided into smaller chunks to retrieve specific information.
2. **Front matter:** Metadata at the beginning of documents should be handled separately to avoid adding unnecessary text to embeddings.
3. **Chunk boundaries:** Character-based chunking may split related explanations across different chunks, which could reduce retrieval quality.

### Next Step

I will inspect the first 40 lines and selected code blocks from `cli.md` and `pyproject.md`, then implement `load_documents()` to extract document metadata and clean the text before chunking.

### Step 2 — Document Inspection and Retrieval Challenges

I inspected the two longest documentation files, `cli.md` and `pyproject.md`.

I noticed three things that could make semantic search harder:

1. **YAML front matter:** The files start with metadata such as title, layout, and menu settings. This information is not part of the main documentation content, so it should be extracted or removed before creating embeddings.
2. **Code blocks:** The documentation includes commands such as `poetry add requests pendulum`. If chunking separates a command from its explanation, the retrieved text may not contain enough context.
3. **Headings and special tags:** The files contain Markdown headings, links, and tags such as `{{% note %}}`. These structures can affect how text is split and what information appears in each chunk.

The files also vary significantly in length, which shows why we need to explore the documents before choosing a chunking strategy.

## Step 3 — Load and Clean Documents

I created `documents.py` with a `load_documents()` function to read and clean the Markdown documentation.

The function extracts the document title from YAML front matter, removes the header block, and cleans repeated blank lines while preserving the main content.

I tested it on the Poetry documentation and successfully loaded all 16 files.

For `cli.md`, the extracted title was "Commands", and for `pyproject.md`, it was "The pyproject.toml file". Both documents started with their actual content instead of the YAML header.

This step helped me understand why document cleaning is important before chunking and generating embeddings.

### Step 4 — Initial Chunking Inspection

I tested fixed-size character chunking on `cli.md` using a chunk size of 300 characters and an overlap of 0.

**Results:**
- Total chunks for `cli.md`: 136
- Total chunks across all 16 documents: 752
- Maximum chunk length: 300 characters
- All original text preserved: True

I printed five consecutive chunks (indices 0–4) and observed two problematic splits:

1. The word `information` was split into `inf` and `ormation` across chunks 0 and 1.
2. The command-line option `--no-ansi` was split into `--no-an` and `si` across chunks 1 and 2.

These examples show that fixed-size character chunking can break words and commands at arbitrary boundaries. This may make individual chunks less meaningful and reduce retrieval quality.

Adding overlap may help preserve context across chunk boundaries, although it increases the total amount of stored text.

### Step 4 — Chunking Validation

I implemented `chunk_text()` and `chunk_documents()` using fixed-size character chunking with configurable overlap.

All six manual validation checks passed:
- Overlap between consecutive chunks is correct.
- Empty text returns an empty list.
- The final part of the text is preserved.
- Invalid overlap raises a `ValueError`.
- Chunk IDs are unique across the documentation.
- Each chunk contains the required metadata.

With a chunk size of 300 characters, the 16 documents produced 752 chunks without overlap and 934 chunks with 60 characters of overlap.

The increase demonstrates the additional storage and embedding work introduced by overlap. Whether it improves retrieval accuracy will be evaluated in later experiments.

I also observed that character-based chunking can split words and command-line options, such as `information` and `--no-ansi`, across chunk boundaries.

## Step 5 — Persistent Vector Storage

I implemented `store.py` using ChromaDB for persistent vector storage and similarity search.

The system uses the pretrained `all-MiniLM-L6-v2` model to generate embeddings for the document chunks. Each stored record includes its chunk ID, text, source file, title, and chunk index.

### Validation Results

- **Collection:** `poetry_300_60`
- **Stored chunks:** 934
- **Persisted chunks after reopening ChromaDB:** 934
- **ChromaDB unit tests:** 2 passed
- **Full project test suite:** 51 passed
- **Ruff lint and formatting checks:** Passed for the Step 5 files

### Retrieval Test

**Query:** How do I add a dependency using Poetry?

| Rank | Source | Similarity |
|---|---|---|
| 1 | dependency-specification.md | 0.7294 |
| 2 | cli.md | 0.7278 |
| 3 | managing-dependencies.md | 0.7205 |

The retrieval pipeline worked, and results were ranked by similarity. However, the top three chunks did not directly show the expected `poetry add` command.

This demonstrates that semantic similarity does not necessarily mean the retrieved chunk contains the answer. Fixed-character chunking may also separate commands from their explanations.

I will keep this configuration as a baseline and evaluate other chunk sizes and overlaps in the upcoming experiments.

### Step 6 — Initial Dataset Validation

I created `data/questions.csv` with 10 evaluation questions.

The validation script confirmed that eight answerable questions reference existing documentation files and contain key phrases found in those files. Two additional questions were marked as having no documented answer.

An initial CSV header formatting issue caused a `KeyError`, which was resolved by correcting the file structure.

The next step is to manually verify that the key phrases represent correct answers, ensure that at least three questions are sufficiently challenging, and combine the questions with contributions from the other trainees to create the final 30-question evaluation dataset.

### Step 6 — Question Difficulty

I revised three evaluation questions to make them more challenging:

- **Q05:** Project root configuration and relative path resolution.
- **Q06:** Dependency metadata required for locking versus building.
- **Q08:** Project dependencies versus Poetry's own runtime dependencies.

These questions require understanding the documentation rather than simply matching command names.

The evaluation dataset currently contains 10 questions, including 8 answerable questions and 2 no-answer questions. Source and key-phrase validation has passed for the initial dataset; the revised questions still require verification.

## Step 7 — Chunking Experiments

I evaluated six fixed-character chunking configurations using the same Poetry documentation, embedding model (`all-MiniLM-L6-v2`), and eight answerable evaluation questions.

### Experimental Results

| Chunk Size | Overlap | Chunks | Avg. Length | Hit@1 | Hit@3 | Hit@5 |
|---|---|---:|---:|---:|---:|---:|
| 300 | 0 | 752 | 296.58 | 12.5% | 37.5% | 37.5% |
| 300 | 60 | 934 | 297.76 | 37.5% | 37.5% | 50.0% |
| 800 | 0 | 286 | 779.83 | 37.5% | 37.5% | 37.5% |
| 800 | 150 | 348 | 783.99 | 25.0% | 37.5% | 50.0% |
| 1500 | 0 | 156 | 1429.68 | 25.0% | 25.0% | 50.0% |
| 1500 | 300 | 190 | 1448.58 | 25.0% | 37.5% | 37.5% |

### Observations

Three configurations achieved the highest preliminary Hit@5 of 50%: `300/60`, `800/150`, and `1500/0`.

The `300/60` configuration also achieved a Hit@1 of 37.5%, tying with `800/0` for the highest Hit@1.

Adding overlap improved Hit@5 for chunk sizes 300 and 800, but reduced Hit@5 for chunk size 1500.

These results suggest that overlap does not guarantee better retrieval. Smaller chunks can provide more focused results but require more stored vectors, while larger chunks reduce the number of vectors but may include unrelated information.

### Limitations

Only eight answerable questions were available for this preliminary evaluation. Each question represents 12.5 percentage points, so the results are sensitive to individual questions.

The experiment must be repeated after combining and verifying the full 30-question dataset.

### Next Step

Analyze retrieval failures to identify whether they are caused by chunk boundaries, noisy chunks, semantic mismatch, or ambiguous ground truth.

### Step 8 — No-Answer Question Analysis

I tested two questions outside the Poetry documentation using the `poetry_300_60` collection.

| Question | Top-1 Similarity | Top-2 Similarity | Top-3 Similarity |
|---|---:|---:|---:|
| What is the current weather in Ramallah? | 0.1809 | 0.1718 | 0.1659 |
| What is today's Bitcoin price? | 0.1765 | 0.1228 | 0.1200 |

ChromaDB returned results for both questions even though they were outside the documentation's scope.

The highest similarity scores were relatively low compared with the earlier dependency-search example, which returned scores around 0.72.

This suggests that a minimum similarity threshold may help filter irrelevant results. A preliminary threshold of 0.30 would reject both unanswered questions in this test.

However, the threshold is not fully validated. More answerable and unanswered questions are needed to measure false rejections and false acceptances.

**Proposed improvement:** Calibrate a similarity threshold using a larger, verified evaluation dataset before using it as a reliable no-answer detection mechanism.

### Step 9 — Streamlit UI Testing

I tested the Docs Search interface locally using the `poetry_800_150` collection.

**No-answer test:**
- Query: What is the weather in Ramallah?
- Top-k: 5
- Minimum similarity: 0.30
- Stored chunks: 348
- Result: No good match found.

The interface successfully filtered irrelevant retrieval results using the selected similarity threshold.

The positive retrieval test, empty-query validation, and cloud deployment still require verification.