## 10. Questions You May Be Asked

### 1. Why did chunk size 300 give a different hit rate than 800 in your results?

Chunk size affects how much information is stored in each chunk.

- With 300 characters, chunks are smaller and more focused, but some information may be separated.
- With 800 characters, chunks contain more context, so the search may find related information more easily.

In my experiments, the 800 / 150 configuration achieved a higher Hit@3 and Hit@5 than 300 / 0.

This shows that keeping enough context can improve retrieval.

### 2. What happens when the answer is split between two chunks? How does overlap help, and what does it cost?

When an answer is split between two chunks, each chunk may contain only part of the information. The search may retrieve one part without the complete answer.

Overlap repeats some text from the previous chunk in the next chunk. This helps keep related information together.

However, overlap also increases the number of chunks. This means more storage, more embeddings, and potentially more processing time.

### 3. Show me a question that failed. Which group of failure is it and what would you change?

One example from my evaluation was:

"Which command adds a package and installs it with its dependencies?"

The expected answer was `poetry add` in `basic-usage.md`, but the top results included unrelated sections from `cli.md`.

This is mainly a retrieval and ranking problem. The correct information was not ranked highly enough.

I would improve keyword matching for exact commands and test different chunk configurations and ranking methods.

### 4. Your hit rate at 5 is 90 percent. How do you know it will stay high with questions from other people?

I cannot guarantee that it will stay the same.

My evaluation used 28 answerable questions from my dataset. Other users may ask different questions or use different words.

To check general performance, I need to test the system with more questions from different users and different writing styles.

The current result is useful, but it does not guarantee the same performance on every dataset.

### 5. What metadata would you store if every answer must show its source?

I would store:

- `source`: the original file name.
- `title`: the document title.
- `chunk_index`: the position of the chunk.
- `text`: the retrieved content.

I could also store a page number or section heading if available.

This metadata helps the application show users where the retrieved information came from.

### 6. What similarity score would you use as a minimum, and how did you choose it?

I tested a minimum similarity threshold of 0.60.

In my tests, it worked for some questions, but it did not reject all irrelevant results. One unanswerable question returned irrelevant results with a similarity score as high as 0.792845.

Therefore, I would not consider 0.60 a final or reliable threshold.

I need to test different thresholds using more answerable and unanswerable questions before choosing a suitable value.

### 7. Why do we test retrieval before we add the LLM?

We test retrieval first to make sure the system can find the correct information.

If retrieval returns irrelevant documents, the LLM may use incorrect information and generate a wrong answer.

Testing retrieval separately helps us identify whether the problem comes from searching or from generating the answer.

It also makes debugging easier.