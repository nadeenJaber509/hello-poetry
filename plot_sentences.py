
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA

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

pca = PCA(n_components=2)
points = pca.fit_transform(embeddings)

plt.figure(figsize=(10, 7))
plt.scatter(points[:, 0], points[:, 1])

for i, (x, y) in enumerate(points):
    plt.annotate(str(i + 1), (x, y), xytext=(5, 5),
                 textcoords="offset points")

plt.title("Sentence Embeddings Visualized with PCA")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.grid(True)
plt.tight_layout()
plt.savefig("plot_sentences.png", dpi=150)
plt.show()
