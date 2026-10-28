
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA

from similarity import cosine_similarity

SENTENCES = [
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


def main():
    # Load a pre-trained embedding model.
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Convert all sentences into numerical vectors.
    embeddings = model.encode(SENTENCES)

    print("Embedding shape:", embeddings.shape)
    print("First 5 numbers:", embeddings[0][:5])
    
    pairs = [(1, 2), (3, 4), (5, 6), (9, 10), (7, 8), (1, 3), (5, 9)]

    for first, second in pairs:
        score = cosine_similarity(
            embeddings[first - 1],
            embeddings[second - 1],
        )
        print(f"Sentences {first} and {second}: {score:.4f}")

    print("\nFull similarity matrix:")
    for a in embeddings:
        print([
            round(cosine_similarity(a, b), 4)
            for b in embeddings
        ])
        # Reduce 384-dimensional embeddings to 2D.
    pca = PCA(n_components=2)
    points = pca.fit_transform(embeddings)

    # Plot each sentence as a point.
    plt.figure(figsize=(9, 6))
    plt.scatter(points[:, 0], points[:, 1])

    for index, (x, y) in enumerate(points):
        plt.annotate(
            str(index + 1),
            (x, y),
            xytext=(5, 5),
            textcoords="offset points",
        )

    plt.title("Sentence Embeddings — PCA 2D")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("plot_sentences.png", dpi=150)
    plt.close()

    print("Saved plot_sentences.png")


if __name__ == "__main__":
    main()
