
from sentence_transformers import SentenceTransformer

from similarity import cosine_similarity

ENGLISH = [
    "The product stopped working after two days.",
    "They charged me twice and refused to refund.",
    "The staff were friendly and solved my problem quickly.",
    "Great quality for the price, I will order again.",
    "I received the invoice by email.",
]

ARABIC = [
    "المنتج توقف عن العمل بعد يومين.",
    "خصموا مني المبلغ مرتين ورفضوا يرجعوا المصاري.",
    "الموظفون كانوا لطفاء وحلوا مشكلتي بسرعة.",
    "الجودة ممتازة بالنسبة للسعر، ورح أطلب مرة ثانية.",
    "وصلتني الفاتورة على الإيميل.",
]

MODELS = [
    "all-MiniLM-L6-v2",
    "paraphrase-multilingual-MiniLM-L12-v2",
]


def compare_model(model_name):
    print(f"\nModel: {model_name}")

    model = SentenceTransformer(model_name)

    english_embeddings = model.encode(ENGLISH)
    arabic_embeddings = model.encode(ARABIC)

    for index, arabic_embedding in enumerate(arabic_embeddings):
        scores = [
            cosine_similarity(arabic_embedding, english_embedding)
            for english_embedding in english_embeddings
        ]

        best_match = max(range(len(scores)), key=lambda i: scores[i])

        print(f"\nArabic query: {ARABIC[index]}")
        print(f"Expected English: {ENGLISH[index]}")
        print(f"Retrieved English: {ENGLISH[best_match]}")
        print(f"Similarity: {scores[best_match]:.4f}")
        print(f"Correct match: {best_match == index}")


if __name__ == "__main__":
    for model_name in MODELS:
        compare_model(model_name)
