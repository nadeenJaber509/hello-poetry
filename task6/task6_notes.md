# Task 6: LLM Classifier with Evaluation

## 1. Test Dataset

- Total comments: 40
- Positive: 13
- Negative: 14
- Neutral: 13
- Majority-class baseline accuracy: 35%

The hardest comments are the sarcastic and mixed-sentiment comments, especially comments where positive and negative opinions are balanced.

## 2. Prompt C

Prompt C adds clearer rules for positive, negative, and neutral sentiment. It tells the model to consider the whole comment, recognize clear sarcasm, and avoid relying on a single positive or negative word.

I expected it to improve classification by making the label rules clearer.

## 3. Prompt Evaluation

Model: `openai/gpt-oss-120b`  
Temperature: 0

| Prompt | Accuracy | Invalid answers | Runtime |
|---|---:|---:|---:|
| A - Zero-Shot | 95% | 0 | 64.92 sec |
| B - Few-Shot | 100% | 0 | 127.27 sec |
| C - Improved | 90% | 0 | 127.98 sec |

- Prompt A incorrectly classified comments 37 and 40 as negative instead of neutral.
- Prompt B classified all 40 comments correctly.
- Prompt C made mistakes on comments 7, 19, 37, and 40. Mixed sentiment was difficult to classify consistently.

## 4. Model Comparison

| Prompt | Model | Accuracy | Invalid answers | Runtime |
|---|---|---:|---:|---:|
| B - Few-Shot | `openai/gpt-oss-120b` | 100% | 0 | 127.27 sec |
| B - Few-Shot | `openai/gpt-oss-20b` | 97.5% | 0 | 62.81 sec |

These runtimes are for 40 comments, which is the size of my dataset.

## 5. Conclusion

I would choose Prompt B with `openai/gpt-oss-120b` when accuracy is the main priority because it achieved 100% accuracy on this test set. The smaller `openai/gpt-oss-20b` model was faster and achieved 97.5% accuracy, so it may be useful when speed matters more. I would test both on more unseen comments before using either model in a real product.

## 6. What I Learned

- Few-shot examples can improve sentiment classification.
- Accuracy and a confusion matrix help measure model performance.
- Temperature 0 helps make outputs more consistent.
- A valid JSON response makes predictions easier to process in Python.
- A high score on one test set does not guarantee the same performance on new comments.

## 7. Questions and Challenges

- Mixed-sentiment comments can be difficult to label consistently.
- I need to understand how to evaluate the classifier on new, unseen comments.
- How can I reduce API costs and processing time when classifying thousands of comments?
