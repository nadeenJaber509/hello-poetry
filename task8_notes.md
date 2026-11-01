# Task 8 — Documents, Chunking, and a Vector Database

## Project overview

In this task, I built the retrieval part of a documentation search system using Poetry's Markdown documentation. The pipeline loads and cleans the documents, splits them into overlapping character-based chunks, generates embeddings with `all-MiniLM-L6-v2`, and stores them in a persistent ChromaDB collection. I then compared six chunking settings, reviewed retrieval failures, and added a Docs Search page to the existing Streamlit application.

The system retrieves relevant passages; it does **not** use an LLM to generate answers.

## 1. Documentation source

- **Repository:** https://github.com/python-poetry/poetry
- **Version:** `2.1.4`
- **Local documents:** `data/poetry_docs/`
- **License copy:** `data/poetry_docs/LICENSE`
- **Git tag reference recorded for `2.1.4`:** `a8f0889a54a545ec4f7ceed7bf41f8c2a7677bbb`

The tag reference above was recorded during setup. It should not be described as the peeled source commit unless that has been verified separately.

## 2. Exploring the documents

I wrote `scripts/task8/explore_docs.py` to count Markdown files, characters, and lines before processing the content.

**Dataset summary:** 16 Markdown files, containing **224,878 characters** in total. The longest file is `cli.md` (40,855 characters and 1,320 lines), followed by `pyproject.md` (25,048 characters and 931 lines). The shortest is `community.md` (654 characters and 30 lines).

| File | Characters | Lines |
|---|---:|---:|
| `cli.md` | 40,855 | 1,320 |
| `pyproject.md` | 25,048 | 931 |
| `dependency-specification.md` | 23,224 | 784 |
| `repositories.md` | 20,559 | 592 |
| `configuration.md` | 18,740 | 611 |
| `faq.md` | 16,606 | 436 |
| `_index.md` | 13,289 | 412 |
| `basic-usage.md` | 13,247 | 300 |
| `contributing.md` | 12,970 | 272 |
| `building-extension-modules.md` | 9,874 | 377 |
| `plugins.md` | 8,335 | 293 |
| `pre-commit-hooks.md` | 5,957 | 181 |
| `managing-dependencies.md` | 5,924 | 186 |
| `managing-environments.md` | 5,503 | 209 |
| `libraries.md` | 4,093 | 135 |
| `community.md` | 654 | 30 |

I inspected the first 40 lines and code examples in `cli.md` and `pyproject.md`. Both contain YAML front matter, Markdown headings, and code examples; the documentation also uses links and special tags such as `{{% note %}}`.

Three issues stood out:

1. **Front matter:** Fields such as `title`, `draft`, and `menu` describe the documentation page rather than its main content.
2. **Code examples:** Commands can lose context if a chunk boundary separates them from their explanations.
3. **Document structure:** Large files, headings, links, and special tags can complicate fixed-length splitting and retrieval.

## 3. Loading and cleaning

I implemented `load_documents()` in `documents.py`. It reads each `.md` file and returns `source`, `title`, and cleaned `text`. It extracts the title from the YAML header, removes that header from the searchable text, and reduces repeated blank lines while keeping the documentation content.

I checked two examples after loading all **16 documents**:

| Source | Extracted title | Header removed |
|---|---|---|
| `cli.md` | `Commands` | Yes |
| `pyproject.md` | `The pyproject.toml file` | Yes |

The original Markdown files were left unchanged; cleaning happens when the files are loaded.

## 4. Character-based chunking

I implemented `chunk_text()` and `chunk_documents()` in `chunking.py`. Chunks have a maximum character length, optional overlap, and a unique ID such as `cli.md::0`. Each record retains its source, title, and chunk index.

I first split `cli.md` with **chunk size 300** and **overlap 0**, then printed five consecutive chunks. The file produced **136 chunks**, and joining them reproduced the cleaned text exactly. Across all 16 files, this setting generated **752 chunks**.

Two boundary problems were visible:

- The word `information` was split into `inf` and `ormation`.
- The option `--no-ansi` was split into `--no-an` and `si`.

These are limitations of splitting strictly by character count. Overlap may preserve some context, but it cannot guarantee a complete sentence or command in every chunk.

I also checked overlap behavior, empty text, the final remainder, invalid overlap values, unique IDs, and required metadata. All six manual checks passed. At chunk size 300, adding 60 characters of overlap increased the total from **752 to 934 chunks**.

## 5. Persistent vector storage

I built `store.py` around ChromaDB's persistent client. The application creates normalized embeddings with `all-MiniLM-L6-v2`, saves each chunk together with its text and metadata, and searches the collection using cosine similarity. Rebuilding a named collection replaces the old collection rather than duplicating its records.

For `poetry_300_60`, **934 chunks** were stored. A separate Python process reopened the collection and confirmed that all **934 records** remained available. The ChromaDB unit tests passed using a deterministic fake embedding model.

I tested the query **“How do I add a dependency using Poetry?”** with the following results:

| Rank | Source | Similarity |
|---:|---|---:|
| 1 | `dependency-specification.md` | 0.7294 |
| 2 | `cli.md` | 0.7278 |
| 3 | `managing-dependencies.md` | 0.7205 |

The retrieved passages were related to dependencies, but these top three did not directly include the expected `poetry add` instruction. This was a useful reminder that a high similarity score does not guarantee that a passage answers the question.

## 6. Evaluation questions

I prepared `data/questions.csv` with **10 questions**: **8 answerable** and **2 intentionally outside the documentation's scope**. The fields are `id`, `question`, `expected_source`, and `key_phrase`.

The validation script confirmed that the eight answerable questions have existing source files containing their expected phrases. The two no-answer entries have empty reference fields. I revised Q05, Q06, and Q08 to make them more challenging by asking about project-root behavior, dependency information used for locking, and Poetry's own runtime dependencies.

This validation checks file and phrase presence; it does **not** prove that every reference is the best answer. Manual ground-truth review remains important, especially for ambiguous questions.

The training guide calls for **30 shared questions** in the final evaluation. The experiments below are preliminary because the additional questions from the other trainees have not yet been incorporated.

## 7. Comparing chunking configurations

Using the same documentation, embedding model, and eight answerable questions, I compared six settings. A hit counted only when a top-k result contained **both** the expected source filename and the expected key phrase (case-insensitive).

| Chunk size | Overlap | Chunks | Average length | Hit@1 | Hit@3 | Hit@5 |
|---:|---:|---:|---:|---:|---:|---:|
| 300 | 0 | 752 | 296.58 | 12.5% | 37.5% | 37.5% |
| 300 | 60 | 934 | 297.76 | 37.5% | 37.5% | 50.0% |
| 800 | 0 | 286 | 779.83 | 37.5% | 37.5% | 37.5% |
| 800 | 150 | 348 | 783.99 | 25.0% | 37.5% | 50.0% |
| 1500 | 0 | 156 | 1429.68 | 25.0% | 25.0% | 50.0% |
| 1500 | 300 | 190 | 1448.58 | 25.0% | 37.5% | 37.5% |

Three settings tied for the highest Hit@5 (**50%**): `300/60`, `800/150`, and `1500/0`. The best Hit@1 (**37.5%**) was shared by `300/60` and `800/0`.

Overlap improved Hit@5 at sizes 300 and 800, but reduced it at size 1500. Larger chunks also required fewer stored vectors. Neither larger chunks nor more overlap was consistently better.

**Limitation:** With only eight answerable questions, one question changes the measured hit rate by **12.5 percentage points**. These differences are too small to support a confident final ranking. Summary and failure files are saved in `results/task8/`.

## 8. Reviewing failures and unanswered questions

For the `300/60` setting, **four of eight** answerable questions failed at Hit@5. I inspected the saved top-ranked results and assigned preliminary failure categories:

| Question | Likely cause | What I observed |
|---|---|---|
| Q01 — Adding a dependency | Semantic mismatch | Results discussed dependencies and `self add` but did not retrieve the expected `poetry add` passage. |
| Q02 — Poetry version | Ground-truth ambiguity | A result containing `poetry --version` could answer the wording of the question, even though the reference expected `poetry about`. |
| Q05 — Project root | Semantic mismatch | Results focused on project paths and environments instead of `--project=PROJECT`. |
| Q08 — Poetry runtime packages | Semantic mismatch | Results discussed plugins and package management rather than the expected `self add` passage. |

These are **likely explanations**, not verified root causes. In particular, the saved failure previews alone cannot prove whether a chunk boundary contributed to a miss.

I then tested two questions with no answer in the Poetry documents:

| Question | Top-1 similarity | Top-2 similarity | Top-3 similarity |
|---|---:|---:|---:|
| What is the current weather in Ramallah? | 0.1809 | 0.1718 | 0.1659 |
| What is today's Bitcoin price? | 0.1765 | 0.1228 | 0.1200 |

ChromaDB still returned nearest neighbors, even for unrelated questions. A **tentative similarity threshold of 0.30** would reject both examples. However, this is not yet a validated no-answer detector; it needs testing against more answerable and unanswered questions to understand false acceptances and false rejections.

**Proposed improvement (not implemented):** Review and clarify ground-truth answers, then calibrate a similarity threshold on a larger verified question set. I would also consider heading-aware chunking as a separate future experiment, rather than changing the baseline mid-evaluation.

## 9. Streamlit Docs Search

I added `pages/5_Docs_Search.py` to the existing app without changing the Task 7 search page. The interface supports choosing one of the six chunk configurations, setting Top-k from 1 to 10, adjusting a minimum similarity threshold, and viewing each retrieved passage with its title, source, chunk index, and similarity score.

The page reuses the existing embedding model and storage functions. It opens a saved ChromaDB collection when available and builds one from the documentation when needed.

**Local checks performed:**

- The page appeared in the application's navigation.
- The `poetry_800_150` collection reported **348 stored chunks**.
- The query **“What is the weather in Ramallah?”**, with Top-k 5 and threshold 0.30, displayed **“No good match found.”**
- A subsequent positive-search check was reported as working, but the exact retrieved passages and scores were not recorded in these notes.

An empty-query UI test and Streamlit Cloud deployment are **not yet documented as verified**.

## 10. Tests, limitations, and next steps

The latest reported full test run completed with **51 tests passed and one third-party deprecation warning**. Ruff checks for the Task 8 scripts and Docs Search files passed. The experiment scripts were moved into `scripts/task8/`; running them with `poetry run python -m scripts.task8.<module>` preserved their imports and behavior.

Before final submission, I still need to:

1. Collect and manually verify the remaining evaluation questions, then rerun the six configurations on the shared dataset.
2. Record the empty-query test and deploy/test Docs Search on Streamlit Cloud, including automatic collection creation.
3. Review the README and repository contents, run the final test suite, and push the verified changes.

**Main takeaway:** Building the vector store was straightforward compared with evaluating retrieval quality. The most useful lesson was that relevant-looking passages, high similarity scores, and even a passing source/phrase check do not always mean the user received the right answer.