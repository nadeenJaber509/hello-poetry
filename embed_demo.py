
import numpy as np
from sentence_transformers import SentenceTransformer

from similarity import cosine_similarity

sentences = [
    "The cat sat on the sofa.",
    "A kitten is resting on the couch.",
    "The stock market fell sharply today.",
    "Shares dropped after the bad news.",
    "I love eating fresh pizza.",
    "Pizza is my favorite food.",
    "The bank is closed on Fridays.",
    "We walked along the river bank.",
    "The delivery was very slow.",
    "My package arrived late.",
]

model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(sentences)

print("Embeddings shape:", embeddings.shape)
print("First 5 numbers:", embeddings[0][:5])

pairs = [(1, 2), (3, 4), (5, 6), (9, 10),
         (7, 8), (1, 3), (5, 9)]

print("\nPairwise cosine similarities:")

for first, second in pairs:
    score = cosine_similarity(
        embeddings[first - 1],
        embeddings[second - 1],
    )
    print(f"Sentences {first} and {second}: {score:.4f}")

similarity_matrix = np.zeros((len(sentences), len(sentences)))

for i in range(len(sentences)):
    for j in range(len(sentences)):
        similarity_matrix[i, j] = cosine_similarity(
            embeddings[i], embeddings[j]
        )

print("\nFull 10x10 similarity matrix:")
print(np.round(similarity_matrix, 3))
