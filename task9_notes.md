## Step 1 — Preparation

For this task, I chose `poetry_800_150` as the fixed chunking configuration based on the results from Task 8. This configuration uses a chunk size of 800 characters and an overlap of 150 characters. It produced 348 chunks and achieved 50% Hit@5 in the initial retrieval tests.

I used `openai/gpt-oss-20b` through the Groq API for answer generation, with temperature set to 0. The default number of retrieved chunks was `k=3`.

Before building the RAG pipeline, I tested the main components:

- **LLM:** The `ask_llm()` function successfully generated a response using Groq.
- **Retriever:** The `query_collection()` function returned three ranked chunks from the Poetry documentation. The top result was from `managing-dependencies.md`, with a similarity score of 0.7097.
- **API key:** I confirmed that `.env` is ignored by Git to avoid exposing the API key.

The current evaluation dataset contains 10 questions: 8 answerable questions and 2 questions that cannot be answered using the Poetry documentation. We still need 20 more verified questions to complete the required dataset of 30 questions.

## Step 2 — Building the RAG Pipeline

I implemented three main functions in `rag.py`:

- `format_sources()` organizes the retrieved chunks into numbered sources.
- `build_messages()` combines the sources, the user's question, and the system instructions into a prompt.
- `answer_question()` connects the retrieval and generation steps and returns the final result.

### How the pipeline works

The RAG chatbot follows five steps:

1. **Receive the question:** The user asks a question about Poetry, such as "How do I add a dependency?"
2. **Retrieve relevant chunks:** The system uses `all-MiniLM-L6-v2` to create an embedding for the question and searches ChromaDB for the three most similar chunks.
3. **Build the prompt:** The retrieved chunks are numbered `[1]`, `[2]`, and `[3]`. Their content, titles, and filenames are added to the prompt along with the question.
4. **Generate the answer:** The prompt is sent to `openai/gpt-oss-20b` through Groq. The model is instructed to give a short answer and include source citations.
5. **Return the result:** The function returns the answer, retrieved sources, similarity scores, and prompt. This information helps us check the quality of the answer.

### Initial testing

I tested the pipeline using three questions:

- **Adding a dependency:** The model generated the correct general command, but the retrieved chunks did not fully support all the information in the answer.
- **Creating a Poetry project:** The retrieved chunks did not contain the required instructions, so the model reported that it did not have enough information.
- **Using virtual environments:** The model generated useful-looking commands, but we still needed to check whether the retrieved sources supported them.

These tests showed that generating a reasonable answer is not enough. We also need to check whether the answer is correct and supported by the retrieved documentation.

## Step 3 — Plain LLM vs. RAG Evaluation

I created an evaluation script to test two approaches:

- **Plain LLM:** The model answers using its existing knowledge without retrieving documentation.
- **RAG:** The model receives relevant chunks from the Poetry documentation before generating an answer.

Both approaches use the same Groq model with temperature set to 0, so the main difference is whether retrieved documentation is provided.

The script saves the answers in CSV files, along with the question, expected source, key phrase, and retrieval information.

### Initial observation

In Q01, both approaches generated the correct `poetry add` command.

However, the RAG system returned `retrieval_hit = false` because the retrieved chunks did not match the expected source and key phrase. Two of the three chunks came from `plugins.md`, which was not the expected documentation for this question.

This showed that the model can sometimes generate a correct answer even when retrieval does not find the expected information.

## Step 4 — Manual Evaluation

After generating the answers, I reviewed them against the Poetry documentation.

I used three grades:

- **Right:** The answer is correct and provides the required information.
- **Partly:** The answer contains correct information but is incomplete or includes an incorrect detail.
- **Wrong:** The answer does not correctly answer the question.

For RAG, I also checked whether the citations supported the generated answer.

### Preliminary results

| Model | Right | Partly | Wrong |
|---|---:|---:|---:|
| Plain LLM | 7 | 2 | 1 |
| RAG (k=3) | 7 | 1 | 2 |

Both approaches answered 7 out of 10 questions correctly under our current grading criteria.

One issue I noticed was that a correct answer did not always have a valid citation. Sometimes the model generated information that was not present in the retrieved chunk. In other cases, it used a related but incorrect command.

For example, in Q08, the Plain LLM correctly suggested `poetry self add`, while RAG suggested `poetry self install`, which was not the correct command for adding a new package to Poetry's runtime environment.

This showed that RAG does not automatically produce better answers. The quality of the retrieved chunks also matters.

## Step 5 — Failure Analysis and Comparing k Values

In this step, I checked why some RAG answers were incorrect and tested whether changing the number of retrieved chunks would improve the results.

I tested three values: `k=1`, `k=3`, and `k=5`.

All experiments used the same embedding model, ChromaDB collection, Groq model, and temperature. Only the number of retrieved chunks changed.

### Results

| Configuration | Right | Partly | Wrong |
|---|---:|---:|---:|
| RAG k=1 | 8 | 0 | 2 |
| RAG k=3 | 7 | 1 | 2 |
| RAG k=5 | 8 | 0 | 2 |

### Analyzing the errors

**Q05 — Generation Failure**

The retrieved documentation explained the difference between `--project` and `--directory`. However, the model gave an inaccurate explanation of how relative paths are resolved.

The required information was available, but the model did not use it correctly.

**Q06 — Retrieval Failure**

The question asked where to define dependency information needed for locking but not for building a package.

The correct answer was `tool.poetry.dependencies`, but the model answered `tool.poetry.dev-dependencies`.

The retrieved chunks did not contain the expected supporting information, which affected the answer.

**Q08 — Retrieval Failure**

The question asked how to add a package to Poetry's own runtime environment.

The correct command is `poetry self add <package>`, but the retrieved context focused on `poetry self install`.

The model used this information and generated the wrong command.

### What happened when k changed?

I compared the answers generated using the three different k values.

- **Q05:** The answers with k=1 and k=5 were more accurate than the answer with k=3.
- **Q06:** The answer remained incorrect for all three values.
- **Q08:** The model continued suggesting `poetry self install` instead of `poetry self add`.
- **Q04:** Retrieval improved with k=5, but the generated answer was already correct with the smaller k values.

### Conclusion

Increasing k did not always improve the answers.

In some cases, retrieving more chunks helped the model find useful information. In other cases, the extra context did not fix the error and could make the answer less accurate.

The main lesson from this experiment is that **the relevance of the retrieved chunks matters more than simply retrieving more of them**.

These results are still preliminary because they are based on 10 questions, including 2 questions outside the scope of the documentation. The final evaluation will require 30 verified questions.

## Step 6 — Hallucination and No-Answer Testing

In this step, I tested how the RAG chatbot handles questions that cannot be answered using the Poetry documentation. I also tested questions with incorrect assumptions and questions that require an opinion.

I used the same RAG pipeline with k=3 without changing the prompt or the code.

### Results

| Test | Behavior | Result |
|---|---|---|
| A — Capital of France | Refused the unrelated question | Appropriate |
| B — Fake deploy-to-cloud command | Reported that the command was not found | Appropriate |
| C — Parallel unit tests | Did not accept the unsupported assumption | Appropriate |
| D — AWS Lambda deployment | Did not invent a deployment command | Appropriate |
| E — Poetry vs other tools | Explained Poetry's features but did not provide a fully supported comparison | Partly supported |
| F — Current weather | Refused the question | Appropriate |
| G — Bitcoin price | Refused the question | Appropriate |

### Observations

The chatbot handled 6 out of 7 test questions appropriately. It usually avoided generating commands or information that were not supported by the documentation.

The main issue appeared in question E. The model described real Poetry features, such as dependency management and lockfiles, but the retrieved sources did not provide enough information for a fair comparison with other packaging tools.

This is important because an answer can contain correct facts while still making a conclusion that is not fully supported.

### Conclusion

The chatbot showed good behavior when it could not find the requested information. However, it still needs better handling of opinion-based questions and clearer explanations when the available sources are not enough.

For a real user, unsupported commands could waste time or cause configuration errors. A better response is to explain the limitation instead of guessing.

I kept the original RAG prompt unchanged during these tests so the results reflect the current system behavior.

## Step 7 — Streamlit Chat Interface

I added a new Streamlit page called **Poetry RAG Chat** to the existing training application.

The interface allows users to ask questions about Poetry and receive answers generated from the retrieved documentation.

The page includes a chat input, conversation history using `st.session_state`, and an expandable section showing retrieved sources, filenames, similarity scores, and chunk text.

I also added a sidebar slider to change the number of retrieved chunks (`k`) from 1 to 8, along with a Clear Chat button.

To improve performance, I cached the embedding model so it does not need to load again for every question.

I tested the page locally using the question "How can I disable ANSI output in Poetry?" The chatbot returned the correct `poetry --no-ansi` command with a source citation.

The conversation history is used only for display. Each new question is processed independently without including previous messages in the RAG prompt.

The local interface works. Cloud deployment and automatic collection setup still need to be verified.