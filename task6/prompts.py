
# Prompt A: Zero-Shot
PROMPT_A = """
Classify the sentiment of the comment as positive, negative, or neutral.

Return only a valid JSON object in this exact format:
{"label": "positive"}

The label must be one of: positive, negative, neutral.
Do not include explanations or any text outside the JSON object.
"""


# Prompt B: Few-Shot
PROMPT_B = """
Classify the sentiment of the comment as positive, negative, or neutral.

Examples:
Comment: The checkout process was smooth and easy.
Output: {"label": "positive"}

Comment: I have been waiting for my refund for weeks.
Output: {"label": "negative"}

Comment: The box arrived on Tuesday.
Output: {"label": "neutral"}

Comment: The screen looks beautiful, but the device breaks easily.
Output: {"label": "neutral"}

Return only a valid JSON object with the key "label".
Use only positive, negative, or neutral as the label.
Do not include explanations or text outside the JSON object.
"""


# Prompt C: Improved Prompt
PROMPT_C = """
You are a careful sentiment classification assistant.

Classify the overall sentiment of the entire comment.

Rules:
- positive: mainly expresses satisfaction, praise, or a good experience.
- negative: mainly expresses dissatisfaction, criticism, or a bad experience.
- neutral: factual, unclear, or balanced between positive and negative
  without one sentiment clearly dominating.
- Consider the meaning and context of the whole comment.
- Do not classify a comment based only on one positive or negative word.
- Treat sarcasm according to its intended meaning when it is clear.
- Do not assume that a factual statement expresses an opinion.

Return only a valid JSON object in this exact format:
{"label": "positive"}

The label must be one of: positive, negative, neutral.
Do not include explanations or any text outside the JSON object.
"""
