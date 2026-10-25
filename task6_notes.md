# Task 6 — LLM Classifier with Evaluation

## Step 2: Dataset

**Total comments:** 45 (30 provided comments, 10 additional English comments, and 5 Arabic comments).

**Label distribution:**
- Positive: 14
- Negative: 15
- Neutral: 16

I added different types of comments to make the dataset more challenging, including sarcasm, mixed opinions, short comments, and Arabic comments.

I expected comments 31 and 35 to be difficult because they sound positive at first, but they're actually sarcastic. Comments 32 and 36 contain both positive and negative opinions, so the model needs to decide whether the overall sentiment is neutral. I also wanted to see how the model would handle sarcasm in Arabic, especially in comment 44.

I think these examples are harder because the model needs to understand the meaning of the whole comment instead of just looking for positive or negative words.

## Step 4: Prompt Design

I tested three different prompts:

- **Prompt A:** Zero-shot prompting with basic classification rules.
- **Prompt B:** Few-shot prompting with four examples.
- **Prompt C:** My improved version, with more detailed rules.

For Prompt C, I wanted to focus on the types of comments that might confuse the model, especially sarcasm and mixed feelings. I also added instructions for Arabic comments because my dataset includes both English and Arabic.

### Prompt C

```text
You are a customer sentiment classifier.

Classify each customer comment using its intended meaning and context,
not just individual positive or negative words.

Return ONLY valid JSON in this exact format:
{"label": "positive"}

Choose exactly one of: positive, negative, neutral.

Rules:
- positive: The customer is satisfied, happy, or praising the experience.
- negative: The customer expresses dissatisfaction, frustration, or criticism.
- neutral: The comment states facts without emotion, or positive and negative
  aspects balance each other.
- Pay attention to sarcasm. A sentence using positive words ironically to
  complain should be classified as negative.
- For mixed comments, evaluate whether one sentiment clearly dominates.
  If neither dominates, choose neutral.
- Handle English and Arabic comments using the same sentiment rules.
- Do not add explanations, markdown, or any text outside the JSON.

Examples:
Comment: Really helpful staff, they solved everything quickly.
Answer: {"label": "positive"}
Comment: Wonderful, the app froze again.
Answer: {"label": "negative"}
Comment: The color is nice, but the material feels cheap.
Answer: {"label": "neutral"}
Comment: وصل الطلب اليوم.
Answer: {"label": "neutral"}
```

I thought Prompt C might perform better because it gives the model clearer rules for understanding difficult comments instead of relying only on a few examples.

## Step 5: Prompt Evaluation

I tested all three prompts using `openai/gpt-oss-20b` with temperature set to 0 and the same 45 comments.

The baseline accuracy was **35.56%**, which means that always predicting the most common label (neutral) would correctly classify 16 out of 45 comments.

### Results

| Prompt | Accuracy | Invalid Outputs | Time |
|---|---|---|---|
| A (Zero-Shot) | 95.56% | 1 | 77.97 s |
| B (Few-Shot) | 95.56% | 0 | 81.19 s |
| C (Improved) | 100.00% | 0 | 80.34 s |

### What I noticed

**Prompt A:** It got 43 out of 45 comments correct. Comment 16 returned an invalid output, and comment 19 ("Okay for the price.") was classified as positive instead of neutral.

**Prompt B:** It also got 43 out of 45 comments correct, but it didn't produce any invalid outputs. It still classified comment 19 as positive, and it classified comment 35 ("Thanks for making me wait three hours. Great service!") as neutral instead of negative.

**Prompt C:** This was the best-performing prompt in my tests. It correctly classified all 45 comments and returned valid labels for all of them.

What surprised me was that adding examples in Prompt B didn't improve accuracy compared with Prompt A. It helped avoid invalid outputs, but the model still made two classification mistakes.

Prompt C performed better, especially on the neutral and sarcastic comments that caused problems in the other versions.

I learned that adding more examples doesn't always make a prompt better. Sometimes, giving clearer instructions about how to handle difficult cases makes more of a difference.

## Step 6: Model Comparison

After comparing the three prompts, I chose Prompt C to test with a second model.

I compared `openai/gpt-oss-20b` and `openai/gpt-oss-120b` using the same prompt, dataset, and temperature (0).

### Results

| Prompt | Model | Accuracy | Invalid Outputs | Time |
|---|---|---|---|---|
| A | GPT-OSS 20B | 95.56% | 1 | 77.97 s |
| B | GPT-OSS 20B | 95.56% | 0 | 81.19 s |
| C | GPT-OSS 20B | 100.00% | 0 | 80.34 s |
| C | GPT-OSS 120B | 91.11% | 0 | 81.92 s |

The results were interesting because I expected the larger model to perform at least as well as the smaller one.

With Prompt C, the 20B model classified all 45 comments correctly, while the 120B model got 41 out of 45 correct.

The four mistakes made by the 120B model were:

| Comment ID | Actual Label | Predicted Label |
|---|---|---|
| 7 | neutral | negative |
| 25 | neutral | negative |
| 36 | neutral | positive |
| 45 | neutral | negative |

All four comments contained a mix of positive and negative opinions. The larger model seemed to focus more on one side of the comment, while the smaller model followed the neutral label rules in this dataset.

### Which one would I choose?

Based on my results, I would choose **Prompt C with GPT-OSS 20B** for this type of classification task.

It gave me the highest accuracy, had no invalid outputs, and took slightly less time than the 120B model in this experiment.

However, I wouldn't assume that the 20B model is always better. My dataset is small, and some mixed-sentiment comments could reasonably be interpreted differently.

Also, I designed Prompt C knowing that my dataset included sarcasm and mixed opinions. To be more confident in the results, I would test it on a new set of comments that I haven't used while designing the prompt.

## Final Thoughts

This task helped me understand the difference between testing an LLM manually and actually evaluating it using code.

In Task 5, I compared outputs by looking at them. In Task 6, I used a labeled dataset, accuracy, a baseline, and a confusion matrix to see exactly where the model made mistakes.

The most interesting part for me was seeing that the longer prompt wasn't automatically better just because it had more text, and that the larger model didn't get the best score.

I also learned why saving predictions is useful, since I can review mistakes and compare different prompts without making the same API calls again.