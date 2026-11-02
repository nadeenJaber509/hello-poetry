
from rag import answer_question
from store import build_vector_store

QUESTIONS = [
    "What is the capital of France?",
    "How do I use poetry deploy-to-cloud?",
    "Why does Poetry run unit tests in parallel by default?",
    "How can I deploy an AWS Lambda function using only a Poetry command?",
    "Which is better: Poetry or another Python packaging tool, and why?",
]


def main():
    collection, embedding_model = build_vector_store()

    for index, question in enumerate(QUESTIONS, start=1):
        print(f"\n{'=' * 60}")
        print(f"Question {index}: {question}")

        result = answer_question(
            question=question,
            collection=collection,
            model=embedding_model,
            k=5,
        )

        print("\nAnswer:")
        print(result["answer"])

        print("\nRetrieved sources:")
        if not result["sources"]:
            print("No sources retrieved.")
        else:
            for source in result["sources"]:
                print(f"- {source['title']} ({source['source']})")


if __name__ == "__main__":
    main()
