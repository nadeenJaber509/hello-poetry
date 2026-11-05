# AI Learning Lab

**Learn AI concepts through simple explanations, practical examples, and interactive Python applications.**

AI Learning Lab is a growing collection of hands-on exercises covering data analysis, machine learning, large language models, semantic search, and Retrieval-Augmented Generation (RAG).

The goal is to make AI concepts easier to understand by connecting each topic to a working implementation that anyone can explore and experiment with.

This project is developed as part of my **AI training at ProGineer**, and it continues to grow as I learn and implement new concepts.

### Live Application

[Explore AI Learning Lab](https://hello-poetry-n3g8uw4ccm6afnqqothxus.streamlit.app)

*The application is hosted on Streamlit Community Cloud. The latest local updates may not appear until they are pushed and deployed.*

---

## Learning Path

The project follows a step-by-step learning journey, starting with working with datasets and progressing toward AI-powered applications.

| Task | Topic | What You'll Learn |
|---|---|---|
| 1–2 | Data Exploration | Explore datasets, understand data types, statistics, and visualizations |
| 3 | Data Cleaning | Handle missing values, duplicates, outliers, and prepare data for ML |
| 4 | Machine Learning | Train classification models and evaluate their performance |
| 6 | LLM Classification | Use language models for text classification and compare prompts |
| 7 | Semantic Search | Understand embeddings and search text by meaning |
| 8 | Document Search | Explore document chunking, vector databases, and retrieval |
| 9 | RAG Assistant | Combine document retrieval with an LLM to generate answers |

Each module includes a short explanation, an interactive example, and points to observe while experimenting.

Task numbering follows the original training sequence.

## Explore the Modules

### Task 1–2 — Data Exploration

Start by understanding a dataset before using it for analysis or machine learning.

**What you can try:**
- Upload a CSV file or use the built-in sample dataset.
- Inspect rows, columns, and data types.
- View summary statistics and missing values.
- Create interactive visualizations.

**Key concept:** Understanding the data is the first step toward building reliable AI systems.

### Task 3 — Data Cleaning & Preparation

Learn how to identify and handle common data quality issues.

**What you can try:**
- Detect missing values, duplicates, and potential outliers.
- Convert numeric-looking text into numerical data.
- Apply different missing-value strategies.
- Compare data before and after cleaning.
- Download the cleaned dataset.
- Prepare train/test datasets with encoding and scaling.

**Key concept:** Preprocessing should be fitted on training data only to help prevent data leakage.

### Task 4 — Machine Learning

Explore how classification models learn from data and make predictions.

**What you can try:**
- Upload a classification dataset.
- Choose target and feature columns.
- Train Logistic Regression, Decision Tree, or Random Forest.
- Compare train, test, and baseline accuracy.
- Examine the confusion matrix.

**Key concept:** A model should be evaluated on unseen data, not only on the data used for training.

### Task 6 — LLM Classification

Experiment with using language models to classify customer comments by sentiment.

**What you can try:**
- Classify comments as positive, negative, or neutral.
- Compare zero-shot, few-shot, and improved prompts.
- Experiment with different Groq-hosted models.
- Explore saved evaluation results.
- Compare accuracy, invalid predictions, and confusion matrices.

**Key concept:** Prompt design and model choice can affect prediction quality.

### Task 7 — Semantic Search

Understand how to search for text based on meaning rather than exact keywords.

**What you can try:**
- Search a dataset of customer comments.
- Retrieve semantically similar comments.
- Compare similarity scores.
- Change the number of results (Top-k) and similarity threshold.

**Key concept:** Embeddings represent text numerically, allowing us to compare semantic similarity.

### Task 8 — Document Search

Explore how technical documentation can be prepared for semantic retrieval.

This module uses **Poetry 2.1.4 documentation** as its knowledge source.

**What you can try:**
- Compare six chunking configurations.
- Explore different chunk sizes and overlaps.
- Search documentation stored in ChromaDB.
- Adjust Top-k and minimum similarity.
- Inspect retrieved text and source metadata.

**Key concept:** Chunking and retrieval settings affect which information a search system finds.

### Task 9 — RAG Assistant

Explore Retrieval-Augmented Generation by asking questions about Poetry documentation.

The chatbot retrieves relevant passages from ChromaDB and sends them to a Groq-hosted language model to generate answers.

**What you can try:**
- Ask questions about Poetry.
- Read generated answers with source citations.
- Inspect retrieved chunks and similarity scores.
- Change the number of retrieved chunks.
- Compare how different retrieval settings influence the answers.

**Key concept:** A generated answer can appear correct without being fully supported by its retrieved sources.

---

## Technologies Used

| Technology | Purpose |
|---|---|
| Python 3.14 | Main programming language |
| Poetry | Dependency and environment management |
| Streamlit | Interactive web interface |
| Pandas | Data exploration and cleaning |
| scikit-learn | Preprocessing, machine learning, and evaluation |
| Sentence Transformers | Text embeddings |
| ChromaDB | Vector database and semantic retrieval |
| Groq API | LLM inference |
| pytest | Automated testing |
| Ruff | Code quality and linting |
| Git & GitHub | Version control and project collaboration |

## Getting Started

### 1. Clone the repository

```bash
git clone -b nadeen-jaber https://github.com/nadeenJaber509/hello-poetry.git
cd hello-poetry
```

### 2. Install dependencies

Make sure Python 3.14 and Poetry are installed.

```bash
poetry install
```

### 3. Configure the API key

Some modules use the Groq API.

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key
```

The `.env` file is excluded from Git and should never be committed.

Modules that do not use Groq can be explored without an API key.

### 4. Run the application

```bash
poetry run streamlit run app.py
```

Open:

http://localhost:8501

Use the sidebar to navigate through the learning modules.

## Testing

The project includes automated tests for data utilities, preprocessing, model training, search, RAG functionality, and application behavior.

Run all tests:

```bash
poetry run pytest -q
```

Check code quality:

```bash
poetry run ruff check .
```

**Latest reported local result:** 70 tests passed. Code-quality checks should be rerun after changes.

## RAG Experiments and Evaluation

Task 9 includes experiments comparing Plain LLM responses with RAG-generated answers.

The experiments explore:

- Retrieval accuracy.
- Answer correctness.
- Citation support.
- Retrieval and generation failures.
- The effect of different Top-k values.
- Hallucination and no-answer behavior.

The initial tests showed that retrieving more chunks does not always improve answer quality. Finding relevant sources and using them correctly are both important.

Evaluation notes and saved results are available in `task9_notes.md` and `results/task9/`.

**Evaluation status:** The current findings are preliminary and based on 10 questions. The required final evaluation dataset contains 30 verified questions.

## Sample Data

The repository includes sample datasets used in the exercises, including:

- `sample.csv` — Example dataset for exploration.
- `invalid.csv` — Invalid CSV used to test error handling.
- `messy_customers.csv` — Dataset with missing values and duplicates for cleaning experiments.

You can also upload your own CSV files to the relevant modules.

## Project Status

**Ongoing — actively developed during AI training.**

This learning lab will continue to grow with new exercises, experiments, and improvements.

The focus is on understanding AI concepts, implementing them in Python, testing their behavior, and making the results easy to explore.

---

**Developed by Nadeen Jaber**

Computer Engineering Student at Birzeit University  
AI Trainee at ProGineer