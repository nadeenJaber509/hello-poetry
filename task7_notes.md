## Step 1: Embeddings

An embedding changes a sentence into a list of numbers that represents its meaning. A computer can compare these numbers to find sentences with similar meanings.

The output shape was (10, 384). This means there were 10 sentences, and each sentence had 384 numbers.
## Step 2: Predict and Measure Similarity

| Pair | My prediction | Real score |
|---|---|---:|
| 1 and 2 | High | 0.6962 |
| 3 and 4 | High | 0.5548 |
| 5 and 6 | High | 0.8032 |
| 9 and 10 | High | 0.5055 |
| 7 and 8 | Low | 0.2203 |
| 1 and 3 | Low | 0.0726 |
| 5 and 9 | Low | 0.0543 |

The scores show that sentences with similar meanings can have high similarity even when their words are different. The pizza sentences had the highest score, 0.8032. I learned that the stock market sentences also have similar meanings because both describe a drop in stock prices, even though they use different words. Their similarity score was 0.5548.

The two bank sentences scored 0.2203, which was lower than the delivery and package sentences at 0.5055, even though the bank sentences share a word.

## Step 3 : PCA 

The PCA plot shows that sentences 1 and 2, 3 and 4, and 5 and 6 are close to each other. This matches their similar meanings. I also noticed that sentences 8 and 9 are close in the plot, while sentences 7 and 10 are far apart. PCA reduces the embeddings to two dimensions, so the distances in the plot may not perfectly represent the original similarities.

### Step 4: Semantic Search

I tested six search queries using semantic search.

- "slow delivery": The results were about delivery and shipping, but the top result described fast delivery.
- "broken product": The top result was relevant because the product stopped working after two days.
- "helpful customer service": The top result described rude and unhelpful service, which did not match the query.
- "money back problem": The top result was relevant because the customer was refused a refund.
- "good value": The results were related to price and quality.
- "quantum physics": The results had very low similarity scores and were not relevant.

I learned that semantic search can find related topics, but it does not always understand the exact meaning of a query. A minimum similarity threshold can help filter out unrelated results.
I added a minimum similarity threshold of 0.30. The search returned only comments with scores of 0.30 or higher. The "quantum physics" query returned no results because all scores were below the threshold.

However, some results were still not fully relevant. For example, the top result for "slow delivery" described fast delivery. I learned that a similarity threshold filters low-scoring results, but it does not guarantee that every result matches the query meaning.
### Step 5: Compare Semantic Search and Keyword Search

I compared keyword search and semantic search using eight queries.

Keyword search finds comments containing the query words. For example, for "invoice", both methods found the comment "I received the invoice by email." However, keyword search returned no results for "late arrival" because these exact words were not found in the comments.

Semantic search can find related meanings even when the exact words are different. For "late arrival", it found comments about delivery and shipping. For "money back problem", it found the comment "They charged me twice and refused to refund."

However, semantic search does not always return the most relevant results. For example, the top result for "slow delivery" described fast delivery, and the results for "quantum physics" were unrelated but had low similarity scores.

A minimum similarity threshold helps remove low-scoring results. In my search page, I used a threshold of 0.30. The "quantum physics" query returned no results because its similarity scores were below this threshold.

I learned that keyword search is useful for exact words, while semantic search is useful for finding related meanings. Both methods have limitations, so the best method depends on the search task.
## Step 6: Testing and Code Quality

I wrote five unit tests for the cosine similarity function.

- Identical vectors returned a similarity score of 1.0.
- Orthogonal vectors returned a similarity score of 0.0.
- A zero vector returned 0.0 without causing an error.
- Similarity was the same when the order of the vectors was reversed.
- Opposite vectors returned a similarity score of -1.0.

I also tested semantic search using a fake model. These tests checked that search returns the expected result, handles a value of `k` larger than the number of comments, and handles an empty query.

I added a test for keyword search to check that it ignores capitalization.

I ran `poetry run pytest`. All 26 tests passed.

Finally, I ran `poetry run ruff check .`. All checks passed.

I learned that unit tests help verify different cases, while Ruff helps identify code quality issues such as incorrectly ordered imports.