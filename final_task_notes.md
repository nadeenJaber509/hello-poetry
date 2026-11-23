# Poetry Support Team — Final Project Report

**Last updated:** 21 November 2026
**Project:** AI Training — Final Task: The Poetry Support Team
## Abstract

This project presents an AI-powered multi-agent support system designed to answer questions about the Poetry dependency management tool using its official documentation.

The system consists of four specialized agents: Router, Researcher, Writer, and Reviewer. It combines Retrieval-Augmented Generation (RAG), documentation search tools, structured output validation, and guardrails to generate evidence-based responses.

The project also explores different retrieval configurations and compares Plain LLM, RAG, Single-Agent, and Multi-Agent approaches. Evaluation includes automated metrics, manual assessment, prompt-injection test cases, and Reviewer ON/OFF experiments.

The results highlight the benefits of documentation-based retrieval while also identifying limitations in multi-agent coordination, answer accuracy, evaluation metrics, and API resource constraints.
## 1. Project Overview and Implementation

The project implements a multi-agent support system designed to answer questions using the official Poetry documentation.

The workflow consists of four agents:

- **Router:** Classifies the user's request and determines its intent.
- **Researcher:** Searches and retrieves relevant information from the official documentation.
- **Writer:** Produces an evidence-based structured answer.
- **Reviewer:** Checks the draft for factual accuracy, relevance, completeness, and unsupported claims.

The coordinator supports:

- Reviewer ON/OFF configurations.
- A maximum of 12 model calls.
- Up to 2 review rounds.
- Controlled review routing and termination.

A LangGraph implementation was also developed using a state graph, conditional review routing, and a bounded review loop.

### Application and technical components

The Streamlit application includes a Poetry Support Team page displaying:

- Final answer and commands.
- Sources and confidence.
- Agent route and research results.
- Writer's draft.
- Reviewer decision and feedback.
- Review count and model-call count.
- Budget status.

Other implemented components include:

- Pydantic structured output using the `SupportAnswer` schema.
- Tool calling and documentation search.
- Documentation tools, including search, reading, and listing documents.
- Guardrail sanitization.
- Model-call budget controls.
- Reviewer feedback and revision handling.

 Tool Calling and Agent Loop

The project implemented a tool-calling system that allows the language model to request documentation operations through predefined function schemas, while Python remains responsible for executing the tools.

Three documentation tools were implemented:

search_docs: Searches the Poetry documentation and returns relevant results with source information and similarity scores.

read_doc: Reads the contents of a specified documentation file, with support for controlled reading ranges and character limits.

list_docs: Lists the available documentation files.

The implementation includes:

Tool schemas compatible with the Groq chat completions API.

A tool-name-to-function mapping.

JSON argument parsing and validation.

Handling of unknown tools and invalid arguments without crashing the agent.

A bounded execution loop with a configurable maximum number of steps.

Duplicate or highly similar search-query detection.

Execution details, including tool names, arguments, results, and timing.

A practical documentation-reading test successfully retrieved content from basic-usage.md and generated an explanation of how poetry run works.

An earlier argument mismatch in read_doc was identified and corrected by aligning the tool schema with the function's supported arguments.

## 2. Code Quality and Automated Testing

The latest recorded local checks were:

| Check | Result | Status |
|---|---:|---|
| Full pytest suite | 63 passed | Passed |
| Ruff — entire project | All checks passed | Passed |
| Ruff — store.py and tests/test_store.py | All checks passed | Passed |
| Poetry project validation | All set! | Passed |

Five additional tests were added in `tests/test_store.py`. They cover:

1. Exact CLI-option prioritization.
2. Stop-word-only queries.
3. Empty collections.
4. Duplicate-result merging.
5. Invalid similarity thresholds.

These results represent the latest completed local checks. No additional full test run was required after the subsequent LangSmith evaluator-only correction.

### Latest Automated Testing and Code Quality Results

The latest project validation confirmed that:

- 63 automated tests passed using Pytest (this replaces the earlier figure of 50 passed tests).
- Ruff completed successfully across the project with no reported issues.
- Additional tests were added for the document retrieval and storage functionality.

These results confirm that the latest tested code passed the automated test suite and static code-quality checks.

Structured Output and Validation

The SupportAnswer Pydantic model was implemented to provide a consistent machine-readable response format.

The model includes:

answerable

answer

commands

sources

confidence

Generated responses are validated using Pydantic before being passed through the remaining workflow. This provides a structured contract between the agents and the application.
### Additional Coordinator Correction-Path Test

An additional unit test was added in `tests/test_reviewer_rejection.py` to verify the coordinator's rejection, revision, and approval sequence.

The test passed successfully:

- Correction-path test: 1 passed.
- Full automated test suite: 64 passed.
- Ruff: All checks passed.

The correction-path test uses a simulated Reviewer rejection to verify the workflow orchestration. It validates the coordinator's behavior, not the language model's real-world ability to identify incorrect answers.

## 3. Retrieval and Chunking Experiments

Twelve configurations were evaluated using Semantic and Hybrid search with different chunk sizes and overlap settings.

| Search Type | Chunk / Overlap | Hit@1 | Hit@3 | Hit@5 |
|---|---:|---:|---:|---:|
| Semantic | 300 / 0 | 25.00% | 57.14% | 60.71% |
| Hybrid | 300 / 0 | 42.86% | 60.71% | 71.43% |
| Semantic | 300 / 60 | 39.29% | 64.29% | 67.86% |
| Hybrid | 300 / 60 | 39.29% | 75.00% | 75.00% |
| Semantic | 800 / 0 | 35.71% | 60.71% | 71.43% |
| Hybrid | 800 / 0 | 42.86% | 78.57% | 85.71% |
| Semantic | 800 / 150 | 32.14% | 71.43% | 82.14% |
| Hybrid | 800 / 150 | 46.43% | 85.71% | 89.29% |
| Semantic | 1500 / 0 | 46.43% | 67.86% | 71.43% |
| Hybrid | 1500 / 0 | 39.29% | 71.43% | 78.57% |
| Semantic | 1500 / 300 | 39.29% | 82.14% | 82.14% |
| Hybrid | 1500 / 300 | **50.00%** | 78.57% | **92.86%** |

### Findings

- Hybrid search with 1500-character chunks and 300-character overlap achieved the best Hit@1 (50%) and Hit@5 (92.86%).
- Semantic search with the same chunk configuration achieved the best Hit@3 (82.14%).
- The project uses the `poetry_1500_300` collection.
- Hybrid search provided the strongest overall retrieval results at Hit@1 and Hit@5 among the tested configurations.
The guardrail implementation includes deterministic checks and sanitization to reduce unsupported or unsafe output.

The validation process covers:

Question-scope checks.

Evidence-based answer validation.

Documentation-source validation.

Poetry-command validation.

Structured response validation.

Sanitization of unsupported configuration and lock-file claims.

The guardrail behavior was also tested using fixed before-and-after examples, with the results recorded in data/guardrails_before_after.csv.

## 4. Automated System Comparison

An initial comparison was conducted on 30 normal cases.

| System | Completed | Phrase Matches | Phrase Accuracy | Coverage | Errors |
|---|---:|---:|---:|---:|---:|
| Plain LLM | 30/30 | 15/30 | 50.00% | 100% | 0 |
| RAG | 30/30 | 19/30 | 63.33% | 100% | 0 |
| Single Agent | 23/30 | 18/23 | 78.26% | 76.67% | 6 |
| Multi-Agent | 30/30 | 18/30 | 60.00% | 100% | 0 |

There were 23 cases completed by all four systems.
Successful LangSmith Tracing

Before the later 10-case evaluation attempt, LangSmith tracing was successfully configured and verified.

The LangSmith project was configured as:

poetry-support-team

Tracing was enabled, and the API key configuration was verified.

A successful LangGraph execution appeared in LangSmith as a parent run with nested agent traces for:

Router

Researcher

Writer

Reviewer

The final structured output was visible in the trace. One recorded execution completed in approximately 42.80 seconds.

This confirms that the project successfully integrated LangSmith tracing for an individual LangGraph execution. It is separate from the later 10-case evaluation experiment described in Section 9.
### Metric limitations

- Phrase accuracy is based on simple substring matching against expected phrases. It is not a semantic or human-judged accuracy score.
- The Single Agent percentage uses its 23 completed cases as the denominator; coverage is reported separately.
- Incomplete executions are not equivalent to incorrect generated answers.
- The four-system comparison uses 30 normal cases and must not be combined with the separate 40-case evaluation dataset.

## 5. Multi-Agent Evaluation

The evaluation dataset contains 40 cases:

- 30 normal cases.
- 7 special cases.
- 3 prompt-injection cases.

The successful Multi-Agent results covered all 40 case IDs. The recorded `data/multi_agent_results.csv` snapshot contained 49 rows, including 9 incomplete attempts. Those incomplete attempts were kept separate from successful answers.

For the first 10 cases in the earlier recorded evaluation:

- Phrase matches: 8/10.
- Source matches: 9/10.
- Average model calls: 6.

Missing expected phrases or sources must be treated as N/A rather than ordinary evaluation failures.

### Updated Multi-Agent Evaluation Results

The latest `data/multi_agent_results.csv` file contained 49 execution rows:

- 40 successful results covering all 40 unique test cases.
- 9 incomplete attempts associated with Groq token-per-day quota limitations.

The 9 incomplete attempts were not counted as incorrect answers. All 40 unique case IDs eventually had successful results.

For the 30 normal cases, the evaluation showed:

- Phrase matches: 18/30 (60%).
- Source matches: 22/30 (73.3%).
- Both phrase and source matched: 13/30.
- Average model calls: 5.73.

Phrase and source matching are automated evaluation indicators. They do not replace manual semantic assessment of answer correctness.

### Context Optimization and Groq Token-Limit Issue

During testing, the system encountered a Groq 413 error because the requested token throughput exceeded the available limit.

The error indicated that the request required 8,785 tokens while the configured limit was 8,000 tokens per minute.

To reduce unnecessary context, the following changes were implemented:

1. Limited the `read_doc` output to 2,000 characters when reading `cli.md` and `plugins.md`.
2. Modified the Writer Agent to receive only the Researcher's collected documentation evidence instead of the entire research object.
3. Applied the same context reduction to the Reviewer Agent.

These changes reduced the amount of information passed between agents, improving how information is transferred between the agents rather than only adjusting the answer.

A subsequent targeted AWS Lambda test completed successfully without the previous 413 error.

 Verified End-to-End LangGraph Execution

A successful end-to-end execution was recorded for the question:

What Python version does Poetry require?

The system returned:

Poetry requires Python 3.9 or newer.

The response included supporting documentation sources and high confidence.

Execution details:

Model calls used: 6

Configured maximum: 12

Chroma retrieval collection: 192 chunks

This demonstrated that the Router, Researcher, Writer, and Reviewer could execute together and return a sourced structured answer within the configured model-call budget.

## 6. Manual Evaluation

Manual evaluation was used to assess correctness, completeness, relevance, and whether answers were supported by the available documentation.

The grading categories were:

- **Right:** Factually correct and adequately answers the question.
- **Partly Correct:** Contains correct information but has missing details, incomplete explanations, or potentially misleading statements.
- **Wrong:** Incorrect, refuses an answerable question, or fails to address the requested information.
- **Not Evaluable:** No final answer was generated because of an execution error.

### 6.1 Normal Cases

Fifteen selected normal cases were reviewed.

| Case ID | Single Agent | Multi-Agent |
|---|---|---|
| 2 | Right | Right |
| 3 | Right | Right |
| 5 | Right | Right |
| 6 | Right | Partly Correct |
| 9 | Wrong | Wrong |
| 11 | Right | Right |
| 12 | Right | Right |
| 14 | Right | Right |
| 15 | Right | Right |
| 16 | Right | Right |
| 18 | Right | Right |
| 22 | Right | Wrong |
| 23 | Partly Correct | Right |
| 27 | Not Evaluable | Wrong |
| 29 | Not Evaluable | Right |
The earlier practical verification covered:

Successful documentation searches and document reading.

Handling of unknown tool names.

Handling of invalid tool arguments.

Enforcement of the tool execution limit.

Prevention of repeated or highly similar search requests.

Successful Streamlit execution of the multi-agent workflow.

Display of the final structured answer and agent execution details.

These checks complement the automated test suite and the later manual evaluation results.
#### Results

| Evaluation Category | Single Agent | Multi-Agent |
|---|---:|---:|
| Right | 11 | 11 |
| Partly Correct | 1 | 1 |
| Wrong | 1 | 3 |
| Not Evaluable | 2 | 0 |
| Total Cases | 15 | 15 |

Single Agent produced 12 evaluable answers, while Multi-Agent produced final answers for all 15 cases.

Among evaluable answers:

- Single Agent achieved 11 fully correct answers out of 12.
- Multi-Agent achieved 11 fully correct answers out of 15.

These results show that producing a final answer more consistently does not automatically mean higher answer correctness.

#### Notable observations

- **Case 6:** Multi-Agent provided the correct `poetry run` command but included an outdated or potentially misleading optional reference to `poetry shell`.
- **Case 9:** Both systems failed to provide the expected `poetry env list` command, even though the research stage identified it. This indicates a problem transferring retrieved evidence into the final answer.
- **Case 22:** Multi-Agent incorrectly classified a question about extension-module build tools as outside its scope.
- **Case 23:** Single Agent gave a generally correct explanation of the `poetry-install` hook but overstated its guarantee about the environment matching the lock file before a commit.
- **Cases 27 and 29:** Single Agent encountered execution errors and did not generate final answers. Multi-Agent answered both, although its answer to case 27 was an incorrect refusal.

### 6.2 Special Cases

The Multi-Agent system was manually reviewed on cases 31–37.

| Case ID | Result | Observation |
|---|---|---|
| 31 | Right | Correctly refused an out-of-scope question. |
| 32 | Partly Correct | Avoided inventing an undocumented cloud deployment command, but research evidence was limited. |
| 33 | Right | Explained that test parallelism is controlled by the test runner, not Poetry itself. |
| 34 | Right | Did not claim that Poetry provides a built-in AWS Lambda deployment command. |
| 35 | Right | Correctly refused an out-of-scope comparison request. |
| 36 | Right | Correctly explained dependency resolution and lock-file creation when no lock file exists. |
| 37 | Right | Did not claim that Poetry automatically fixes security vulnerabilities. |

**Summary:** 6 Right, 1 Partly Correct, 0 Wrong.

### 6.3 Prompt-Injection Cases

Cases 38–40 tested attempts to:

- Reveal system instructions.
- Steal passwords or secrets.
- Follow malicious instructions embedded in documentation.

| Case ID | Result |
|---|---|
| 38 | Right |
| 39 | Right |
| 40 | Right |

All three tested prompt-injection cases were handled successfully.

This is a limited manual result based on three examples. It does not establish comprehensive resistance to prompt injection or replace a broader security assessment.

### 6.4 Single Agent Special-Case Limitation

The Single Agent results for cases 31–40 had execution errors and contained no final answers.

These cases were therefore classified as Not Evaluable rather than incorrect generated answers.

## 7. Reviewer ON/OFF Comparison and Correction-Path Testing

### 7.1 Experiment Scope

A controlled comparison was conducted using the first 10 cases from `data/cases.csv` to evaluate the Multi-Agent workflow with the Reviewer enabled and disabled.

Both configurations used the same model (`openai/gpt-oss-20b`) and the same 10 test cases.

### 7.2 Quantitative Results

| Metric | Reviewer OFF | Reviewer ON |
|---|---:|---:|
| Successful cases | 10/10 | 10/10 |
| Phrase matches | 9/10 (90%) | 9/10 (90%) |
| Source matches | 10/10 (100%) | 8/10 (80%) |
| Total model calls | 52 | 53 |
| Reviewer approvals | N/A | 10 |
| Reviewer rejections | N/A | 0 |

The results show that enabling the Reviewer did not improve phrase matching in this sample. Source matching was lower with the Reviewer enabled, decreasing from 100% to 80%.

The Reviewer-enabled configuration used one additional model call (53 compared with 52).

### 7.3 Reviewer Behavior and Answer Revision

In the 10-case comparison, the Reviewer approved all 10 generated drafts. No rejection decisions or corrective feedback were recorded.

The combined draft length was 3,618 characters, compared with 2,040 characters in the final answers, representing an approximate 43.6% reduction.

Although all final answers were shorter, shorter answers do not necessarily mean more accurate or complete answers. In case 1, for example, the final answer omitted an evidence quotation that appeared in the draft, although the source remained listed.

These observations indicate that the Reviewer executed successfully, but the experiment did not demonstrate that it independently detected and corrected errors in the generated answers.

### 7.4 Correction-Path Unit Test

A separate unit test was implemented in `tests/test_reviewer_rejection.py` to verify the coordinator's rejection-and-revision workflow.

The test simulates the following sequence:

1. The Reviewer rejects the initial draft.
2. The Reviewer provides corrective feedback.
3. The Writer receives the feedback and generates a revised answer.
4. The Reviewer evaluates the revised answer and approves it.

**Test result:**
- Test: `tests/test_reviewer_rejection.py`
- Result: 1 passed.
- The test verified that the coordinator correctly handled the rejection, Writer revision, and subsequent approval sequence.

This test confirms that the correction path is implemented and works under the simulated test conditions. It does not establish that the language model Reviewer will reliably detect real factual errors in production responses.

### 7.5 Interpretation and Limitations

The Reviewer ON/OFF comparison was limited to 10 cases. The automated phrase and source matching metrics also have limitations:

- Case 1 used "Python 3.9 or newer" instead of the expected phrase "Python 3.9+". The answer was semantically correct despite the phrase mismatch.
- In cases 3 and 5, the expected source was `basic-usage.md`, while the system returned `cli.md`. These may be valid alternative sources and require careful interpretation.

The experiment therefore does not establish that enabling the Reviewer improves answer accuracy. It shows that the Reviewer executed and approved the tested drafts, while the separate unit test verified that the rejection-and-revision mechanism works when a rejection is simulated.

**Conclusion:** The Reviewer is successfully integrated into the workflow, and its correction path is covered by a unit test. However, its actual ability to improve answer quality remains unproven and requires evaluation on drafts containing known factual or completeness errors.
## 8. Guardrail Evaluation

The deterministic sanitizer evaluation in `data/guardrails_before_after.csv` used four fixed examples.

Results:

- Two unsupported claims involving a TOML path and lock-file updates were removed.
- Two supported or unrelated answers were preserved.
- Unsafe claims in these four examples decreased from 2 to 0.

This demonstrates the sanitizer's behavior on the selected examples only. It does not prove general prompt-injection resistance or comprehensive security.

## 9. LangSmith Evaluation

A 10-case LangSmith experiment was attempted under the experiment name:

`langgraph-support-team-86d8c866`

The experiment recorded 10 runs, but one run failed after the Groq token-per-day quota was reached. The execution also encountered technical issues, including tool-choice errors, duplicate or similar search-query errors, and empty model responses.

The initial LangSmith dashboard displayed:

| Metric | Recorded Value |
|---|---:|
| Phrase Match | 0.60 |
| Source Match | 1.00 |
| Answerability Match | 0.40 |

### Evaluator correction

A subsequent code inspection identified an evaluator reference-mapping problem.

The dataset creation script stores the expected values in `example.outputs` under these keys:

- `key_phrase`
- `expected_source`
- `answerable`

The earlier evaluator attempted to read some of these values using incorrect keys or from metadata. This made the earlier metric results unreliable.

The evaluator was subsequently corrected to read all three expected values from the correct output fields. Missing phrase or source references are now omitted from the corresponding metric rather than automatically counted as matches.

**Important:** The existing LangSmith experiment was not rerun after this correction. The old dashboard metrics therefore remain historical, preliminary values and must not be presented as validated performance results.

The LangSmith project and 40-case dataset were created. A complete, corrected LangSmith performance evaluation remains unavailable because of the quota limitation and the decision not to consume additional quota for another run.

### Latest LangSmith Validation

The initial 10-case LangSmith experiment was interrupted by Groq token-per-day quota limitations and was not considered a valid evaluation run.

After optimizing the context passed between agents, a targeted retry was performed using the AWS Lambda question:

> "How can I deploy an AWS Lambda function using only a Poetry command?"

Observed results:

- Execution status: Success.
- Model calls: 7.
- Answerable: False.
- Sources: `cli.md` and `plugins.md`.
- Commands: Empty list.

The system correctly indicated that the available Poetry documentation did not provide a built-in AWS Lambda deployment command.

This successful single-case retry confirms that the tested workflow completed without the previous token-limit error. It does not replace the incomplete 10-case LangSmith experiment.

## 10. Important Interpretation Notes

1. The initial four-system comparison is based on 30 normal cases and simple phrase matching.
2. The later Multi-Agent dataset contains 40 cases, including special and prompt-injection cases. These evaluation sets must not be combined as if they were the same experiment.
3. Missing expected phrases or sources should be marked N/A, not False.
4. Incomplete runs and quota failures must be reported separately from incorrect answers.
5. The prompt-injection manual result is limited to three test cases.
7. Phrase matching is a lexical metric, not a complete measure of semantic correctness.
8. Source matching does not establish that every factual claim in an answer is correct.
9. The original LangSmith metrics are not validated because of the evaluator reference-mapping issue.
10. Manual evaluation provides additional evidence about correctness, completeness, and misleading details.

## 11. Final Submission Checklist

The following implementation and evaluation tasks have already been completed and should not be repeated:

- Four-agent support workflow and coordinator.
- LangGraph workflow.
- Streamlit Support Team page.
- Structured output, documentation tools, and guardrail sanitization.
- Local automated checks: 63 tests passed, Ruff passed, and Poetry validation passed.
- Retrieval and chunking experiments.
- Initial four-system comparison.
- Manual evaluation of 15 normal cases.
- Manual evaluation of special cases 31–37.
- Manual review of prompt-injection cases 38–40.
- Multi-Agent evaluation of all 40 cases (49 CSV rows, including 9 quota-related incomplete attempts).
- Groq 413 context optimization and targeted AWS Lambda retest.
- Targeted single-case LangSmith validation (AWS Lambda question).
- LangSmith evaluator code correction.

Remaining submission actions:

- Confirm whether a deployed Streamlit URL is required and verify it if applicable.
- Complete the final review of the repository changes and included files.
- Ensure `.env`, `.venv`, Chroma database files, and planted prompt-injection test content are not included in the repository.
- Complete the required Git commit/PR process after final approval.
- Prepare the required three-line group update.

No additional full test suite or LangSmith experiment is claimed as completed after the changes described above.

## 12. Conclusion

The project implements a four-agent Poetry Support Team consisting of a Router, Researcher, Writer, and Reviewer. It includes a LangGraph workflow, structured Pydantic output, documentation tools, guardrail sanitization, budget controls, and a Streamlit interface.

The latest recorded local automated checks passed: 63 tests passed, Ruff reported no issues, and Poetry project validation succeeded.

The project was evaluated using retrieval experiments, a four-system comparison on 30 normal cases, manual evaluation of 15 normal cases and 10 special and prompt-injection cases, and a Multi-Agent evaluation covering all 40 cases.

The retrieval experiments identified Hybrid search with 1500-character chunks and 300-character overlap as the strongest configuration for Hit@1 and Hit@5 among the tested settings.

Manual evaluation showed that both Single Agent and Multi-Agent produced 11 fully correct answers in the selected normal cases, while their coverage and error patterns differed. All three manually tested prompt-injection cases were handled successfully, but this small sample is not a comprehensive security assessment.

On the 30 normal cases, the Multi-Agent evaluation showed 18/30 phrase matches (60%), 22/30 source matches (73.3%), and an average of 5.73 model calls. A Reviewer ON/OFF comparison was not conducted, so no conclusion is drawn about the Reviewer Agent's independent contribution. The Groq 413 token-limit error was addressed by reducing the context passed between agents, and a targeted AWS Lambda retest succeeded.

The initial 10-case LangSmith experiment was attempted, but its original metrics were affected by execution issues, quota exhaustion, and an evaluator reference-mapping problem. The evaluator code was corrected, but the experiment was not rerun. Its original metrics are therefore not treated as validated performance results.

**Overall, the core implementation, recorded local checks, and planned manual evaluations are complete. Final submission readiness depends on the remaining repository and submission steps, and deployment verification if required by the task.**

---

# Latest Progress Update – 18 November 2026

## 1. Researcher Agent Fix

Updated the Researcher execution flow to resolve the conflicting system instructions that previously caused the model to return `Tool choice is none`.

- Separated the Researcher instructions from the final-answer instructions.
- Preserved the evidence and safety requirements.
- Successfully tested the Researcher with a Poetry documentation question.
- Confirmed that the Researcher could retrieve documentation and return a final answer.

## 2. Reviewer JSON Response Fix

Investigated a LangSmith execution failure:

`ValueError: Reviewer did not return valid JSON.`

The Reviewer was using Groq's JSON response format:

`response_format={"type": "json_object"}`

Removed this parameter from the Reviewer agent call while keeping the explicit JSON requirements in the Reviewer prompt and the existing response parser.

### Validation

The following checks passed:

- `poetry run ruff check agents.py`
- `poetry run python -m compileall -q agents.py`
- Successfully imported `reviewer_agent`.

A direct Reviewer test was then executed using:

`How can I list the available Poetry commands?`

**Observed result:**

- `review_decision`: `approve`
- Valid structured response returned.
- `answerable`: `true`
- `commands`: `["poetry --help"]`
- `sources`: `["cli.md"]`
- `confidence`: `high`

This confirms that the Reviewer successfully returned a valid structured response in the direct test.

## 3. Full Workflow Test

Tested the complete workflow using `run_support_team` with the same Poetry commands question.

The ChromaDB initialization completed successfully:

- Collection: `poetry_1500_300`
- Existing chunks: 192
- Expected chunks: 192
- Final collection count: 192

The Router also successfully classified the question under `commands`.

However, the workflow could not complete because Groq reached its daily token limit.

**Groq API error:**

- Daily TPD limit: 200,000 tokens.
- Tokens already used: 199,645.
- Requested tokens: 3,121.
- Error code: `429 – rate_limit_exceeded`.

The workflow stopped after 9 model calls and returned `ANSWER: None`.

This was an API quota limitation, not confirmation of a new Reviewer error.

## 4. LangSmith Evaluation – Previous Results

The latest 10-case LangSmith experiment completed with the following results:

| Metric | Result |
|---|---:|
| Execution success | 8/10 (80%) |
| Valid output among successful executions | 8/8 (100%) |
| Answerability match | 7/8 (87.5%) |
| Phrase match | 3/5 (60%) |
| Source match | 3/3 (100%) |
| Median latency | 94.45 seconds |

Two executions failed:
- One Reviewer JSON response failure.
- One Groq rate-limit failure (429).

The direct Reviewer test after removing JSON response mode passed, but the full workflow and LangSmith experiment have not yet been rerun to verify the fix.

## 5. Current Status and Remaining Work

**Completed in this update:**
- Researcher instruction conflict fix and successful test.
- Reviewer JSON response adjustment.
- Ruff and compilation checks for `agents.py`.
- Direct Reviewer test successfully returned `approve` with valid JSON.
- ChromaDB collection integrity confirmed.

**Pending:**
1. Wait for Groq's daily token quota to become available.
2. Rerun the full `run_support_team` test and confirm that it returns a valid final answer.
3. Rerun the LangSmith experiment to verify whether the Reviewer JSON failure has been resolved.
4. Review and document the final evaluation results.
5. Complete the remaining project documentation and submission checks.

**Important:** The Reviewer fix has passed its direct test, but the full workflow is not yet confirmed successful after the change.

## Reviewer JSON Generation Fix & Integration Testing

### 1. Problem Description

During the integration testing of the Poetry Support Team, the Reviewer agent failed to generate a valid JSON response.

The initial error was:

```text
400 - json_validate_failed
Failed to validate JSON.
```

After switching the Reviewer from `json_object` to strict `json_schema`, another error appeared:

```text
max completion tokens reached before generating a valid document
```

This indicated that the model reached its completion-token limit before generating a complete JSON document.

### 2. Root Cause

The Reviewer needed to generate a structured response containing multiple fields:

- `review_decision`
- `review_feedback`
- `answerable`
- `answer`
- `commands`
- `sources`
- `confidence`

The available completion-token budget was insufficient for some structured responses, particularly when the model needed additional reasoning.

### 3. Implemented Fixes

**A. Switched to Structured Outputs**

Updated the Reviewer response format in `agents.py` from:

```python
response_format={"type": "json_object"}
```

to strict JSON Schema:

```python
response_format={
    "type": "json_schema",
    "json_schema": {
        "name": "reviewer_response",
        "strict": True,
        "schema": review_schema,
    },
}
```

**B. Updated LLM Request Parameters**

Updated `task6/llm.py` to include:

```python
request_args = {
    "model": model,
    "messages": messages,
    "temperature": temperature,
    "max_completion_tokens": 4096,
    "reasoning_effort": "low",
}
```

These settings provide a larger completion-token budget and reduce the reasoning effort used by the model.

**C. Improved Error Handling**

Updated `agent.py` so that `BadRequestError` is re-raised when a structured response is required, instead of silently returning an empty response or fallback answer.

This makes structured-output failures easier to identify and debug.

### 4. Test and Code Quality Results

After updating the test expectation in `tests/test_agents.py` to verify the new JSON Schema response format:

- Pytest: **63 passed**
- Ruff: **All checks passed**

### 5. Integration Testing Results

The application was tested through its Streamlit interface using three different questions.

| Test | Question | Expected Behavior | Result |
|---|---|---|---|
| 1 | What is the difference between `poetry add` and `poetry install`? | Explain the difference with supporting sources. | Passed |
| 2 | How can I remove a dependency from a Poetry project? | Provide the correct Poetry command and relevant options. | Passed |
| 3 | How can I deploy a Django application to AWS Lambda? | Reject the question because it is outside the Poetry Support Team scope. | Passed |

### 6. Detailed Integration Test Observations

**Test 1: Dependency Installation**

The system returned:

- `answerable: true`
- A comparison between `poetry add` and `poetry install`.
- Sources: `cli.md`, `basic-usage.md`.
- Confidence: `high`.

**Test 2: Dependency Removal**

The system returned:

- `answerable: true`
- The command: `poetry remove <package>`.
- Relevant options, including `--group`, `--dev`, `--dry-run`, and `--lock`.
- Sources: `cli.md`, `managing-dependencies.md`.
- Confidence: `high`.

**Test 3: Out-of-Scope Question**

The system correctly returned:

- `answerable: false`
- An explanation that the question was outside the Poetry Support Team scope.
- `commands: []`
- `sources: []`

This confirmed that the system rejected an unrelated question instead of generating unsupported instructions.

### 7. Current Status

The following checks have been completed successfully:

- [x] Reviewer structured-output configuration updated.
- [x] Completion-token and reasoning settings updated.
- [x] Structured-output error handling improved.
- [x] All 63 automated tests passed.
- [x] Ruff checks passed.
- [x] Integration test: dependency comparison.
- [x] Integration test: dependency removal.
- [x] Integration test: out-of-scope question.

**Conclusion:** The Reviewer JSON generation issue was resolved in the three integration scenarios tested. The application successfully generated structured answers, retrieved relevant sources, provided Poetry commands, and rejected an out-of-scope question.

The next step is the final evaluation and comparison of Plain LLM, RAG, Single Agent, and Multi-Agent Support Team approaches.

# Reviewer ON vs OFF – Experimental Findings

## 1. Experiment Scope

The experiment compared the Poetry Support Team with the Reviewer enabled and disabled.

- Dataset: First 10 cases from `data/cases.csv`.
- Model: `openai/gpt-oss-20b`.
- Reviewer ON results: `multi_agent_reviewer_on_detailed_results.csv`.
- Reviewer OFF results: `multi_agent_reviewer_off_results.csv`.

**Note:** These results cover 10 cases, not the complete 40-case dataset.

## 2. Quantitative Results

| Metric | Reviewer OFF | Reviewer ON |
|---|---:|---:|
| Successful cases | 10/10 | 10/10 |
| Phrase Match | 9/10 | 9/10 |
| Source Match | 10/10 | 8/10 |
| Total Model Calls | 52 | 53 |
| Reviewer Approvals | N/A | 10 |
| Reviewer Rejections | N/A | 0 |

## 3. Reviewer Behavior

With Reviewer enabled:

- All 10 cases received an `approve` decision.
- Each case required one review cycle.
- No rejection decisions were recorded.
- No reviewer feedback was recorded.

The experiment confirms that the Reviewer executed successfully, but it does not demonstrate that it detected and corrected errors in the generated answers.

## 4. Draft vs. Final Answer

The combined draft length was 3,618 characters, while the final answer length was 2,040 characters.

This represents an approximate 43.6% reduction in answer length.

The final answers were shorter in all 10 cases. However, shorter answers do not necessarily indicate better answers.

For example, in case 1, the final answer omitted the evidence and quotation included in the draft, although the source remained recorded in the `sources` field.

Further inspection is needed to determine whether this reduction affects answer completeness and evidence visibility.

## 5. Evaluation Limitations

Two types of evaluator mismatches were observed:

- **Case 1 – Phrase Match:** The answer used "Python 3.9 or newer" instead of the expected phrase "Python 3.9+". The answer was semantically correct, but the exact-match evaluator marked it as a failure.
- **Cases 3 and 5 – Source Match:** The evaluator expected `basic-usage.md`, while the results recorded `cli.md`. The source-selection criterion should be reviewed to determine whether valid alternative documentation sources are being incorrectly marked as failures.

These mismatches indicate limitations in the current evaluation criteria and should not automatically be treated as answer-quality failures.

## 6. Model Call Comparison

Reviewer ON used 53 model calls, compared with 52 for Reviewer OFF.

The observed difference was one additional call. This is a small difference in this sample, but token usage and actual API costs should also be considered before drawing conclusions about efficiency.

## 7. Final Conclusion

The experiment demonstrated that the Reviewer can execute and approve generated answers successfully.

However, no measurable improvement in answer accuracy was demonstrated in these 10 cases. The Reviewer approved every draft without providing corrective feedback.

The main observed effect was shorter final answers, but this reduction may also remove useful evidence and explanations.

Therefore, the Reviewer should not yet be considered a proven quality-improvement component.

Further testing should include cases with intentionally incomplete or incorrect drafts, so that the system's ability to detect problems, reject answers, and trigger revisions can be evaluated.

**Overall conclusion:** Reviewer execution was successful, but its effectiveness in improving answer quality remains unproven.
