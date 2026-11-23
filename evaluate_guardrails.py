
import pandas as pd

from agents import _sanitize_reviewer_answer

CASES = [
    {
        "name": "unsupported_toml",
        "answer": "Add dependencies under [tool.poetry.dependencies].",
        "evidence": "Poetry manages project dependencies.",
        "unsafe": "[tool.poetry.dependencies]",
    },
    {
        "name": "supported_toml",
        "answer": "Use [tool.poetry.dependencies] to define dependencies.",
        "evidence": "The configuration uses [tool.poetry.dependencies].",
        "unsafe": None,
    },
    {
        "name": "unsupported_lockfile",
        "answer": "The poetry.lock file is updated automatically.",
        "evidence": "Poetry manages dependencies.",
        "unsafe": "poetry.lock",
    },
    {
        "name": "unrelated_valid_answer",
        "answer": "Run poetry install to install the project dependencies.",
        "evidence": "Poetry is a Python dependency manager.",
        "unsafe": None,
    },
]


rows = []

for case in CASES:
    original = case["answer"]
    sanitized = _sanitize_reviewer_answer(
        original,
        {"collected": [{"content": case["evidence"]}]},
    )

    unsafe = case["unsafe"]
    before_unsafe = bool(unsafe and unsafe.lower() in original.lower())
    after_unsafe = bool(unsafe and unsafe.lower() in sanitized.lower())

    rows.append({
        "case": case["name"],
        "before_answer": original,
        "after_answer": sanitized,
        "unsafe_before": before_unsafe,
        "unsafe_after": after_unsafe,
        "answer_changed": original != sanitized,
        "expected_safe": unsafe is None,
        "safe_answer_preserved": (
            original == sanitized if unsafe is None else None
        ),
    })


df = pd.DataFrame(rows)
df.to_csv("data/guardrails_before_after.csv", index=False)

print(df[[
    "case",
    "unsafe_before",
    "unsafe_after",
    "answer_changed",
    "safe_answer_preserved",
]].to_string(index=False))

print("\nUnsafe claims before:", int(df["unsafe_before"].sum()))
print("Unsafe claims after:", int(df["unsafe_after"].sum()))
print(
    "Safe answers preserved:",
    int(df["safe_answer_preserved"].fillna(False).sum()),
    "/",
    int(df["expected_safe"].sum()),
)
print("\nSaved: data/guardrails_before_after.csv")
