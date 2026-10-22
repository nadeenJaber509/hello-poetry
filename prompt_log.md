## Experiment 1: Temperature

- **Model and settings:** I used openai/gpt-oss-120b with temperature 0 and 1.2. I tried each setting three times.
- **What I saw:** At temperature 0, I got three café names: Al-Qahwa al-Qadima, Rihan Café, and Sahra Café. At temperature 1.2, I got different names: Bustan Bean Café, Al-Quds Café, and Nablus Nook Café.
- **What I would change in the prompt:** I would ask for shorter descriptions and more creative names.

## Experiment 2: System Prompt

- **Model and settings:** I used openai/gpt-oss-120b with two different system prompts.
- **What I saw:** Prompt A gave a formal answer and explained the problem first. Prompt B used simple words and examples, like apples and a ruler. Both explained why we should split data before scaling.
- **What I would change in the prompt:** I would ask for short sentences and one simple example.

## Experiment 3: Few-Shot Prompting

- **Model and settings:** I used openai/gpt-oss-120b with prompts with and without examples. I tested temperature 0 and 1.2.
- **What I saw:** Without examples, the model got 9 out of 10 at temperature 0 and 10 out of 10 at temperature 1.2. With examples, it got 10 out of 10 at both settings. However, it also gave explanations when I asked for labels only.
- **What I would change in the prompt:** I would clearly ask for only 10 labels, with no extra text.

## Experiment 4: Prompt Chaining

- **Model and settings:** I used openai/gpt-oss-120b at temperature 0 with two prompts.
- **What I saw:** The first prompt found 96 missing cells and 15 duplicate rows. The second prompt suggested filling missing values and removing duplicate rows.
- **What I would change in the prompt:** I would ask for simple Python code to solve each problem.

## Experiment 5: Role Prompting

- **Model and settings:** I used openai/gpt-oss-120b at temperature 0. I tried two different roles.
- **What I saw:** The teacher role explained DataFrames using a simple example about weekly meals. The data scientist role used technical words and showed Python code.
- **What I would change in the prompt:** I would clearly say who the answer is for and how simple it should be.

## Experiment 6: Constrained Prompting

- **Model and settings:** I used openai/gpt-oss-120b at temperature 0 with two prompts.
- **What I saw:** The first prompt asked for three bullet points with no more than 12 words each. The second asked for exactly 8 words per point, starting with a verb. The model followed both prompts correctly.
- **What I would change in the prompt:** I would ask the model to check the word count before answering.

## Experiment 7: Model Comparison

- **Model and settings:** I compared openai/gpt-oss-120b and openai/gpt-oss-20b at temperature 0.
- **What I saw:** The first model used a weekly-meals example to explain a DataFrame. The second used a grocery list. Both gave simple explanations without Python code.
- **What I would change in the prompt:** I would test both models with more questions to compare them better.
