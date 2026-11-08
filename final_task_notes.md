# Final Task: The Poetry Support Team

## Introduction

I started this task by understanding the main concepts and the big picture of building a small team of AI agents that use tools, return structured JSON, and check their own answers.

The goal is to understand how an AI agent system works using plain Python before exploring frameworks such as LangGraph and LangSmith.

## The Big Picture

The system consists of four AI agents, followed by validation checks in Python:

1. **Router:** Classifies the user's question (how-to, concept, troubleshooting, or out-of-scope).
2. **Researcher:** Uses tools to search and read the Poetry documentation to collect relevant evidence.
3. **Writer:** Generates an answer in JSON format using only the evidence provided by the Researcher.
4. **Reviewer:** Checks whether the answer is supported by the evidence. It either approves the answer or sends it back to the Writer with feedback.
5. **Code Checks:** Verify the sources and commands before returning the final answer to the user.

The agents have different responsibilities, and a Python coordinator will control the workflow between them.

---

## Part 1: Tool Calling

### Goal

The goal of Tool Calling is to let the language model decide which function it needs to use instead of having a fixed sequence of functions for every question.

An important concept I learned is that **the LLM does not execute Python functions itself**. It only requests a tool call, and our Python code executes the requested function and returns the result to the model.

### Step 1.1: Creating the Tools

I created `src/hello_poetry/tools.py` with three Python functions:

1. **`search_docs(query, k=3)`**
   Searches the Poetry documentation using ChromaDB and returns relevant chunks with their source, title, similarity score, and text.

2. **`read_doc(source)`**
   Reads the content of a specific file from `data/poetry_docs`, limited to 6,000 characters. It returns an error message if the requested file does not exist.

3. **`list_docs()`**
   Lists the available documentation files in `data/poetry_docs`.

### Tool Schemas and Tool Registry

In addition to the functions, I created:

- **`TOOL_SCHEMAS`:** Describes each tool to the LLM, including its name, purpose, parameters, and when to use it.
- **`TOOL_FUNCTIONS`:** A dictionary that maps each tool name to its Python function so our application can execute the requested tool.

The model uses the tool descriptions to decide which function to request, while Python controls the actual execution.

### Testing the Tools

I tested the three functions individually before connecting them to an LLM.

- `list_docs()` successfully returned the available Poetry documentation filenames.
- `read_doc()` successfully returned the content of a documentation file.
- `search_docs()` successfully returned relevant document chunks with their metadata and similarity scores.

I also tested error handling by calling:

```python
read_doc("not_existing.md")
```

The function returned:

```text
Error: Document 'not_existing.md' was not found.
```

The program did not crash.

I ran Ruff to check the code quality:

```bash
poetry run ruff check src/hello_poetry/tools.py
```

**Result:** `All checks passed!`

### What I Learned

- Tool Calling allows an LLM to choose which available tool to request based on the user's question.
- The LLM does not directly run the function. Python receives the request, validates it, and executes the function.
- Tool descriptions are important because they explain to the model when each function should be used.
- `TOOL_SCHEMAS` defines the interface visible to the model, while `TOOL_FUNCTIONS` connects tool names to actual Python functions.
- Error handling is important because an invalid tool request should not crash the entire agent system.


---

## Step 1.2: Agent Loop

## Step 1.2: Agent Loop

### Understanding the Agent Loop

The concept of an Agent Loop is that the user asks a question, and the Groq LLM decides whether it needs to use a tool.

For example, the LLM may request `search_docs`. Python executes the function and sends the retrieved results back to the LLM.

The LLM then reads the results and decides whether it has enough information to answer the question or needs to call another tool. This process continues until the model generates a final answer or reaches the maximum number of steps.

### Implementation

For this part, I added a function called `chat_with_tools()` to `llm.py`. This function calls the Groq API and allows the model to request local tools using their schemas.

I also created `agent.py` with the help of ChatGPT. It contains the main function:

```python
run_agent(
    question,
    system_prompt,
    tool_schemas,
    tool_functions,
    model,
    max_steps=5,
)
```

This function manages the Agent Loop. It sends the user's question to the LLM, receives tool requests, validates the tool names and arguments, executes the requested Python functions, and sends the results back to the model.

The loop stops when the LLM provides a final answer or when `max_steps` is reached.

The function also records the tools used, their arguments, execution time, total tokens, and whether the agent stopped because of the step limit. The execution trace is saved as a JSON file.

### Testing the Agent

**Test Question:**

How do I add a dependency using Poetry?

The Agent Loop executed successfully.

**Results:**
- `search_docs` was called 3 times.
- `read_doc` was called 1 time to read `cli.md`.
- Total tokens: 14,674.
- Stopped by limit: `False`.
- The agent generated a final answer explaining how to add dependencies using the `poetry add` command.

This test showed that the LLM can request multiple tools and search more than once before generating an answer.

However, I noticed that the agent used many tokens for a relatively simple question. This shows why monitoring token usage and limiting model calls are important.

### Error Handling Tests

I tested two error cases using `execute_tool()` directly.

**Test 1: Unknown Tool**

```python
execute_tool("unknown_tool", "{}", TOOL_FUNCTIONS)
```

Result:

```text
Error: Unknown tool 'unknown_tool'.
```

**Test 2: Invalid JSON Arguments**

```python
execute_tool("search_docs", "{invalid}", TOOL_FUNCTIONS)
```

Result:

```text
Error: Invalid tool arguments: Expecting property name enclosed in double quotes: line 1 column 2 (char 1)
```

Both tests passed without crashing the program.

These tests showed that Python checks tool names and arguments before executing the functions. If a tool request is invalid, the function returns an error message instead of executing an unknown function or stopping unexpectedly.

### Testing max_steps = 1

I tested the same question again, but changed `max_steps` from 5 to 1.

**Results:**
- Model calls: 1.
- Total tokens: 324.
- Tool executed: `search_docs`.
- Final answer: `None`.
- Stopped by limit: `True`.

The result was expected because Python sent the question to the LLM, and the LLM requested `search_docs`. Python executed the search, but the loop reached its maximum number of steps before the LLM could read the retrieved results and generate a final answer.

This helped me understand that `max_steps` limits the number of LLM calls, not necessarily the number of tool executions.

### What I Learned

- An Agent Loop allows an LLM to request tools multiple times before producing a final answer.
- The LLM decides which tool to request, but Python is responsible for executing it.
- The model may repeat a search with different queries to collect more information.
- Tool errors should be handled without crashing the application.
- A step limit prevents the agent from continuing indefinitely and helps control API usage and costs.
- Saving execution traces helps us understand and debug the agent's decisions.

### Testing Five Questions

I tested the agent with five different questions, including one outside the Poetry documentation.

| Question | Tools Used | Tokens | Outcome |
|---|---|---:|---|
| How do I add a dependency using Poetry? | `search_docs` ×3, `read_doc` ×1 | 14,674 | Generated an answer |
| How do I create a new Poetry project? | `search_docs` ×5 | 16,146 | Reached the step limit without an answer |
| What is the purpose of poetry.lock? | `search_docs` ×2, `read_doc` ×1 | 9,780 | Generated an answer |
| How can I check which dependencies are outdated? | `search_docs` ×1 | 1,809 | Returned an empty answer |
| Who won the FIFA World Cup in 2022? | None | 454 | Refused to answer |

### Observations

The agent successfully used tools to retrieve information from the Poetry documentation. For the dependency question, it searched multiple times and then read a specific file before answering.

However, the agent did not always behave efficiently. When asked how to create a new Poetry project, it repeated similar searches until the maximum step limit was reached.

Another issue occurred when the model returned an empty final answer without reaching the step limit. This showed that checking whether tool calls are absent is not enough; the final response should also be checked.

For the out-of-scope World Cup question, the agent refused to answer without searching the documentation.

### What I Learned

The LLM can decide which tools to use, but its decisions are not always efficient. A working tool-calling loop does not guarantee a useful final answer.

The step limit is important because it prevents unnecessary repeated model calls. Execution traces help identify where the agent stopped, which tools it selected, and what went wrong.

This experiment also showed why we need stronger validation and guardrails in the next parts.

