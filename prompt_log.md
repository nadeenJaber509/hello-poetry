
### Experiment 1: Temperature

**Model and settings:** openai/gpt-oss-120b, Temperature = 0 and 1.2, three runs for each setting.

**What I saw:** In my first attempt, I used the same conversation for all six runs. The model generated different coffee shop names, but the descriptions were very similar, and I couldn't notice a clear difference between the two temperature settings. I realized that previous responses in the conversation might have influenced the next ones.

I repeated the experiment using separate, fresh conversations. At temperature 0, the model returned exactly the same name and description in all three runs. At temperature 1.2, it generated different names, although the descriptions still followed similar themes related to Palestinian coffee and local culture. This showed me that a low temperature is useful for more consistent outputs, while a higher temperature can be useful for creative tasks that benefit from variety.

**What I would change in the prompt:** I would make the prompt more specific about the type of creativity I want, while keeping the same prompt and using fresh conversations when comparing temperature settings.

**Additional Prompt Test:** I tested the prompt: "Suggest a creative coffee shop name in Nablus. Avoid traditional coffee and olive-related themes. Use an unexpected concept."

At temperature 0, the model suggested **"Pixel & Pour,"** a modern, technology-inspired name instead of a traditional one. This showed me that prompt specificity can strongly influence the type of response, even at a low temperature. It also helped me understand that temperature controls sampling behavior, while the prompt guides the model toward particular ideas.

### Experiment 2: System Prompt

**Model and settings:** openai/gpt-oss-120b, Temperature = 0, Reasoning Effort = medium.

**What I saw:** The first system prompt produced a direct, technical response structured as a problem followed by a fix. The second prompt gave a simpler explanation with an everyday example about making soup. Both answers explained data leakage correctly, but their tone and teaching style were different.

**What I would change in the prompt:** I would ask the tutor to use a more directly related example, such as student exam scores, to make the concept easier to connect to machine learning.

**What I learned:** A system prompt defines the model's role, behavior, and response style while keeping the user's question unchanged.

### Experiment 3: Zero-Shot vs. Few-Shot Prompting

**Model and settings:** openai/gpt-oss-120b, Temperature = 0, Reasoning Effort = medium.

**What I saw:** Both the zero-shot and few-shot prompts produced exactly the same labels for all ten customer comments. Using my reference labels, both scored 10/10, so adding three examples did not improve accuracy in this experiment. Both prompts also followed the requested output format.

**What I would change in the prompt:** I would add clearer rules for mixed-sentiment comments and test more challenging examples to see whether few-shot prompting makes a difference.

### Experiment 4: JSON Output

**Model and settings:** openai/gpt-oss-120b, Temperature = 0, JSON Mode = off.

**What I saw:** The model returned valid JSON with the four required keys for all four messages. It used null for missing information in Message 2 and followed the system instructions instead of writing a poem in Message 4. However, in Message 3, it filled the complaint_topic field with "repair" even though the customer was giving positive feedback rather than making a complaint.

**What I would change in the prompt:** I would clarify that complaint_topic should be null when the message does not contain a complaint and explicitly instruct the model to ignore instructions embedded in customer messages.

**Did the fix rule solve a failure?** I did not apply the fix rule because the original prompt already handled the missing values and the prompt-injection attempt successfully in these tests.

### Experiment 5: Step-by-Step Prompting

**Model and settings:** openai/gpt-oss-120b, Temperature = 0, Reasoning Effort = medium.

**What I saw:** Both Version A and Version B returned the correct answer of 28.6 shekels in all three runs, giving each version a score of 3/3. Even without explicit step-by-step instructions, the model explained its calculations. However, Version B organized the solution more clearly and consistently included a separate `ANSWER:` line.

**What I would change in the prompt:** I would ask the model to provide a brief calculation followed by a clearly formatted final answer, making the result easier to extract automatically.

### Experiment 6: Hallucinations

**Model and settings:** openai/gpt-oss-120b, Temperature = 0, Browser Search = off.

**What I saw:** The model correctly answered that "strawberry" contains three r's and did not invent a summary for the nonexistent research paper. However, when asked about the population of Taffuh, it confidently gave 13,000 residents and cited the 2017 PCBS census, even though the number did not match the official data I checked.

**What I would change in the prompt:** I would explicitly tell the model not to guess and to say "I don't know" when it cannot verify the information.

**Fix test:** After adding the instruction, the model answered "I don't know" instead of repeating the unsupported number.

**How I could detect this automatically:** I could build a fact-checking system that retrieves data from trusted sources and checks whether the model's numerical claims are supported by those sources.

### Experiment 7: Model Comparison

**Models and settings:** openai/gpt-oss-20b and openai/gpt-oss-120b, Temperature = 1, Reasoning Effort = medium.

**What I saw:** Both models classified all ten customer comments correctly based on my reference labels and returned valid JSON for the customer message. The extracted complaint topics were slightly different in wording but had the same meaning.

The smaller 20B model had lower reported latency in both tasks. For sentiment classification, it recorded 392 ms compared with 537 ms for the 120B model. For JSON extraction, it recorded 240 ms compared with 326 ms. The 20B model also had a higher reported token generation speed.

**What surprised me:** I expected the larger model to perform better, but both models produced equally accurate results on these simple tasks.

**What I would change:** I would test more difficult examples, repeat each test several times, and compare total token usage before deciding which model is better for a real application.

**Conclusion:** For these two small experiments, I would choose the 20B model because it produced comparable results with lower reported latency. However, this is not enough evidence to conclude that it performs better on all tasks.

**Token Count Note:** The Playground displayed latency and token generation speed (tokens/s), but I did not record the total token count. Therefore, I could not compare actual token consumption between the two models.
