# Task 7 — Embeddings and Semantic Search

## Introduction

Before starting Task 7, I reviewed the main concepts of embeddings, vectors, cosine similarity, and semantic search to understand how they work together.

The main idea is that semantic search compares the meaning of sentences rather than only matching their words. To do this, we first convert text into numerical vectors called embeddings, then use cosine similarity to compare them.

I also reviewed how an embedding model processes text:

1. **Tokenization:** The sentence is split into tokens (words or parts of words).
2. **Transformer:** The model processes the tokens and their context to create numerical representations.
3. **Pooling:** The token representations are combined into one vector representing the sentence.
4. **Output:** The model returns the final sentence embedding.

For this task, I used the pretrained model **`all-MiniLM-L6-v2`**, which produces a vector containing **384 numbers for each sentence**.

The model runs locally, so I do not need a Groq API key to generate embeddings.

## Step 1 — Generating Embeddings

I started by creating `embed_demo.py` and using the model to generate embeddings for the 10 sentences provided in the task guide.

The output was:

```text
Embedding shape: (10, 384)
First 5 numbers:
[ 0.1654904  -0.08392975 -0.03161799  0.0416702  -0.03970969]
```

The shape `(10, 384)` means that we have 10 sentences, and each sentence is represented by a vector of 384 numbers.

These numbers do not represent individual words directly. Instead, the complete vector represents the sentence in a numerical space where semantically related sentences are expected to have similar representations.

The model was already trained, so I only used it to generate embeddings without training it again. After the model files are downloaded, they can be reused locally.

## Step 2 — Measuring Cosine Similarity

Before comparing the real sentence embeddings, I created `similarity.py` and implemented the cosine similarity formula using NumPy.

Cosine similarity compares the directions of two vectors. The score ranges from -1 to 1:

- **1:** The vectors point in the same direction.
- **0:** The vectors are perpendicular.
- **-1:** The vectors point in opposite directions.

First, I tested the function with two simple, identical vectors:

```python
cosine_similarity([1, 0], [1, 0])
```

The result was `1.0`, as expected. This helped me verify that the function worked correctly for this basic case before using it with real embeddings.

After that, I connected `similarity.py` with `embed_demo.py` and calculated the similarity between the seven sentence pairs required in the guide.

I recorded my predictions **before running the comparisons**, then compared them with the actual results.

### Predictions vs. Actual Results

| Pair                               | My Prediction | Real Score |
| ---------------------------------- | ------------- | ---------- |
| 1–2: Cat / Kitten                  | High          | 0.6962     |
| 3–4: Stock market / Shares         | Low           | 0.5548     |
| 5–6: Pizza / Favorite food         | High          | 0.8032     |
| 9–10: Slow delivery / Late package | High          | 0.5055     |
| 7–8: Financial bank / River bank   | Low           | 0.2203     |
| 1–3: Cat / Stock market            | Low           | 0.0726     |
| 5–9: Pizza / Delivery              | Low           | 0.0543     |

### Observations

The model generally gave higher similarity scores to sentences with related meanings, even when they used different words.

The highest score was **0.8032** for the two pizza sentences, which describe almost the same preference.

One result that surprised me was the stock market pair (3–4). I predicted low similarity, but the actual score was **0.5548**. After reviewing the sentences, I understood that both describe a decline in the stock market using different expressions.

Another interesting comparison was between the two bank sentences (7–8) and the delivery sentences (9–10).

The bank sentences share the word *bank*, but one refers to a financial institution and the other to the side of a river. Their similarity was only **0.2203**.

In contrast, the delivery sentences describe a similar problem without sharing the same important words, and their similarity was **0.5055**.

This showed me that the embedding model can capture context and meaning instead of simply counting matching words.

However, I was expecting a higher score for the delivery pair because both sentences describe a late delivery. This also reminded me that semantic similarity scores are not perfect measures of human understanding.

### Similarity Matrix

I also generated the full **10 × 10 similarity matrix**, comparing every sentence with every other sentence.

I noticed that the diagonal values were all `1.0`, because each sentence was being compared with itself.

The matrix was also symmetric, meaning the similarity between sentence A and sentence B was the same as the similarity between B and A.

The cosine similarity function compares the entire embedding vectors, not individual words. It calculates the dot product of the vectors and divides it by the product of their magnitudes.

## Step 3 — PCA Visualization

After calculating the similarity scores, I wanted to visualize the relationships between the sentence embeddings.

Since each embedding contains 384 dimensions, it cannot be directly displayed on a standard 2D plot.

I used **Principal Component Analysis (PCA)** from scikit-learn to reduce the embeddings from 384 dimensions to 2.

Then I used matplotlib to create a scatter plot and saved it as `plot_sentences.png`.

### What I Observed

The plot showed several clear groups:

- **Sentences 1 and 2 (Cat / Kitten):** These appeared very close to each other.
- **Sentences 3 and 4 (Stock market / Shares):** These also formed a close pair.
- **Sentences 5 and 6 (Pizza):** These appeared very close, which agreed with their high cosine similarity score.
- **Sentences 9 and 10 (Delivery):** These appeared in the same general area but were not as close as the other related pairs.
- **Sentences 7 and 8 (Bank):** These appeared in different parts of the plot, reflecting their different meanings.

One thing I noticed was that sentence 8 (river bank) appeared relatively close to sentence 9 (slow delivery), even though they are not strongly related in meaning.

This helped me understand an important limitation of PCA.

### Why PCA Is Not Enough to Measure Similarity

PCA reduces the dimensions by preserving as much variance as possible in the selected components. However, reducing 384 dimensions to only 2 causes some information to be lost.

As a result, the distances between points in the 2D plot do not always represent the original relationships between their embeddings accurately.

Therefore, I should use the original 384-dimensional embeddings for cosine similarity calculations and use the PCA plot only as a visual tool to understand the data.

## What I Learned So Far

Through the first three steps, I learned that an embedding model converts text into numerical vectors that can be compared mathematically.

I also understood the difference between the responsibilities of the embedding model and cosine similarity. The model creates the numerical representation of the text, while cosine similarity measures how closely two vectors are aligned.

Most importantly, I learned that similar meanings can produce similar embeddings even when sentences use different words, and that visualizations are helpful but should not replace the original similarity calculations.

These concepts will be the foundation for building the semantic search system in the next step.

## Step 4 — Building a Semantic Search Engine

I created `search.py` to search through the 45 comments from Task 6 using semantic similarity.

First, I loaded the comments from `data/test_comments.csv`. Then I used the pretrained `all-MiniLM-L6-v2` model to convert each comment into a 384-dimensional embedding.

I saved the generated embeddings in `embeddings.npy` so I would not need to compute them again for every search.

When a user enters a query, the system converts it into an embedding using the same model, calculates its cosine similarity with all saved comment embeddings, sorts the results by score, and returns the top five matches.

### Search Results

| Query | Real Top 3 IDs | Highest Score |
|---|---|---|
| slow delivery | 1, 25, 40 | 0.7789 |
| broken product | 27, 31, 8 | 0.5746 |
| helpful customer service | 24, 14, 11 | 0.5836 |
| money back problem | 18, 13, 5 | 0.3889 |
| good value | 4, 19, 10 | 0.4438 |
| quantum physics | 33, 25, 3 | 0.1494 |

### Observations

The search successfully found a refund-related comment when I searched for "money back problem", even though the comment used the word "refund" instead of the exact words in my query.

However, I noticed that some high-scoring results had the opposite meaning from the query. For example, "slow delivery" returned a comment about fast delivery, and "helpful customer service" returned a comment describing rude and unhelpful service.

This showed me that an embedding model can identify related topics without always distinguishing opposite meanings or sentiment correctly.

The most important test was "quantum physics". Although none of the comments discussed this topic, the search still returned five results. The highest similarity score was only 0.1494, and the returned comments were unrelated.

This happens because the current search function always returns the top-k results, even if their similarity scores are very low.

A possible solution is to set a minimum similarity threshold and return "No good match found" when no result reaches it. The threshold should be tested on different queries rather than chosen arbitrarily.

I also learned that the highest-ranked result is not necessarily correct, and semantic similarity is not a measure of factual accuracy.

## Step 5 — Comparing Semantic Search and Keyword Search

I implemented `keyword_search()` to compare traditional word matching with semantic search.

The keyword search function converts the query and comments to lowercase, splits them into words, and counts how many query words are present in each comment. Unlike semantic search, it does not use embeddings or a machine learning model.

I created `compare.py` to run both methods on the same comments and queries.

### Case 1 — Semantic Search Performs Better

For the query **"money back problem"**, semantic search returned comment ID 18:

"They charged me twice and refused to refund."

The similarity score was 0.3889.

Keyword search returned comments that contained the word "problem" but were not specifically related to getting money back.

I also tested **"late arrival"**. Semantic search returned comment ID 40, which described an express shipment arriving late, with a score of 0.4825. Keyword search found no exact matching words and returned results with scores of zero.

These examples showed me that semantic search can retrieve relevant sentences even when the query uses different wording.

### Case 2 — Keyword Search Is Sufficient

For the query **"invoice"**, both methods returned comment ID 13:

"I received the invoice by email."

Keyword search found the exact word with a score of 1, while semantic search returned a similarity score of 0.7400.

In this case, keyword search was sufficient because the query contained an exact term already present in the comment.

### Limitations I Observed

Semantic search sometimes returned comments that were topically related but expressed the opposite meaning. For example, "slow delivery" returned a comment about fast delivery.

Keyword search also had limitations. It could not recognize synonyms, and it returned comments with zero matching words when no better results were available.

Neither method guarantees that the highest-ranked result is correct.

### When Would I Use Each Method?

I would use semantic search when users describe problems in natural language and may use different words from those in the stored data.

I would use keyword search for exact terms, identifiers, and specific product names where precise matches are important.

For a real product, I would consider combining both methods to benefit from exact matching and semantic understanding.

## Step 6 — Testing Without Loading the Model

I created `tests/test_search.py` to test the search functionality without downloading or running the real embedding model.

Instead of using `all-MiniLM-L6-v2`, I created a `FakeModel` that returns predefined vectors. This allowed me to test the search logic with known and predictable results.

I tested cosine similarity using identical, perpendicular, and opposite vectors, expecting scores of 1, 0, and -1.

I also tested whether semantic search sorts results correctly, handles a `k` value larger than the number of comments, and handles empty queries without crashing. For keyword search, I verified that matching is case-insensitive.

All seven new tests passed. The complete project test suite also passed with **36 tests**.

Ruff initially reported three import-order formatting issues in `compare.py`, `embed_demo.py`, and `search.py`.

Using a fake model helped me understand the difference between testing our application logic and evaluating the quality of a pretrained embedding model.

## Step 7 — Streamlit Semantic Search Page

I added a Search page to my Streamlit application with a query input, a Top-k slider, and a minimum similarity threshold.

I used caching to avoid repeatedly loading the embedding model and recomputing the comment embeddings.

I tested the page locally using different queries. For "broken product", the system returned comment ID 27 with a similarity of 0.5746. For "invoice", it returned ID 13 with a similarity of 0.7400.

I selected an initial minimum similarity threshold of 0.30 after reviewing the search results. This threshold filtered out all results for "quantum physics", whose highest original score was only 0.1494, and displayed "No good match found."

I also noticed that the threshold does not guarantee relevance. Some results still describe a related topic with a different meaning or sentiment.

The next step is to verify the deployed application on Streamlit Cloud.