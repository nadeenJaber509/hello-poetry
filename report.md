Poetry Support Team
Final Project Report
Project: Poetry Support Team
Domain: AI-powered technical support using official Poetry
documentation
Implementation: Python, Streamlit, LangGraph, ChromaDB, Sentence
Transformers, Groq, Pydantic, LangSmith
Evaluation dataset: 40 test cases (30 normal, 7 special, 3
prompt-injection)
Abstract
This project presents an AI-powered multi-agent support system designed
to answer questions about the Poetry dependency management tool using
its official documentation. The system uses four specialized
agents---Router, Researcher, Writer, and Reviewer---coordinated through
a bounded workflow.

The implementation combines Retrieval-Augmented Generation (RAG),
documentation tools, structured output validation, and guardrails. The
project investigates retrieval configurations and compares Plain LLM,
RAG, Single-Agent, and Multi-Agent approaches. Evaluation includes
automated phrase and source matching, manual answer review, special and
prompt-injection cases, Reviewer ON/OFF experiments, automated tests,
and LangSmith tracing.

The results show that documentation retrieval can improve evidence
availability, while also revealing limitations in answer accuracy,
source selection, agent coordination, evaluation metrics, latency, and
API resource constraints. The Reviewer ON/OFF experiment did not
demonstrate a measurable improvement in answer matching, so the Reviewer
should be viewed as an additional checking mechanism rather than a
proven accuracy improvement.
1. Project Overview
The goal of the project is to build a Poetry-focused support assistant
that answers user questions based on official Poetry documentation.
Instead of relying only on a language model's internal knowledge, the
system retrieves relevant documentation and uses it as evidence for its
responses.

The project explores whether a multi-agent architecture can organize the
support process into specialized responsibilities and produce
structured, traceable answers.
1.1 Main Objectives
- Answer questions about Poetry using official documentation.
- Retrieve relevant documentation passages before generating answers.
- Separate routing, research, writing, and reviewing responsibilities.
- Provide sources and structured answer fields.
- Reduce unsupported claims through evidence-based prompts and
  guardrails.
- Handle out-of-scope questions and prompt-injection attempts.
- Limit model usage and prevent unbounded agent loops.
- Compare different LLM and RAG architectures using a test dataset.
1.2 Main Features
- Streamlit interface for submitting questions and viewing responses.
- Multi-agent workflow coordinated with LangGraph.
- ChromaDB vector store containing indexed Poetry documentation.
- Documentation tools: search_docs, read_doc, and list_docs.
- Pydantic-based structured output using SupportAnswer.
- Reviewer approval and revision path.
- Guardrails for scope, evidence, citations, commands, and unsafe
  instructions.
- Model-call budget and bounded review rounds.
- LangSmith tracing and experiment monitoring.
2. System Architecture
The system contains four specialized agents.
2.1 Router Agent
The Router analyzes the user's question and identifies its intent and
whether it is within the Poetry support scope. It provides routing
information to guide the research process.
2.2 Researcher Agent
The Researcher searches the indexed Poetry documentation and uses
documentation tools to obtain relevant evidence. The available tools
are:
- search_docs: searches the documentation collection.
- read_doc: reads relevant documentation content.
- list_docs: lists available documentation files.
The Researcher returns evidence for the Writer rather than relying on
unsupported assumptions.
2.3 Writer Agent
The Writer uses the question, route, and retrieved evidence to generate
a structured response. The response follows the SupportAnswer Pydantic
schema and can include an answer, sources, commands, confidence, and
answerability information.
2.4 Reviewer Agent
The Reviewer checks the Writer's draft for factual support, relevance,
completeness, and unsupported claims. It can approve the draft or
request a revision with feedback.

The workflow is bounded: the maximum number of model calls is 12 and the
maximum number of review rounds is 2. These limits help prevent
excessive calls and repeated review loops.
2.5 Workflow Summary
1. The user submits a question through Streamlit.
2. The Router classifies the question.
3. The Researcher retrieves relevant documentation evidence when
   appropriate.
4. The Writer generates a structured answer.
5. The Reviewer checks the draft when Reviewer mode is enabled.
6. If revision is requested, feedback is returned to the Writer within
   the configured limit.
7. Guardrails and structured validation are applied.
8. The final result is displayed in Streamlit.
The Reviewer can be disabled for comparison experiments. The exact
execution path depends on the selected configuration.
3. Technologies and Tools
  Technology              Purpose
  Python                  Main implementation language
Streamlit               User interface
LangGraph               Agent workflow orchestration
ChromaDB                Vector database for documentation chunks
Sentence Transformers   Text embedding generation
all-MiniLM-L6-v2      Embedding model
Groq API                Language model inference
openai/gpt-oss-20b    Model used in the reported experiments
Pydantic                Structured output schema and validation
LangSmith               Tracing and experiment monitoring
Pytest                  Automated testing
Ruff                    Python linting and code-quality checks
Poetry                  Project dependency and environment management
4. Retrieval-Augmented Generation
The project uses official Poetry documentation as the knowledge source.
The documentation is split into chunks, converted into embeddings, and
stored in ChromaDB.
4.1 Embedding and Chunking Configuration
The selected collection is poetry_1500_300.

  Setting                    Value
  Embedding model            all-MiniLM-L6-v2
Chunk size                 1500 characters
Chunk overlap              300 characters
ChromaDB collection        poetry_1500_300
Number of indexed chunks   192
4.2 Retrieval Configuration Experiments
Twelve configurations were evaluated using different chunk sizes,
overlap values, and retrieval approaches.

  Retrieval approach     Chunk size   Overlap    Hit@1    Hit@3    Hit@5
  Semantic                      300         0   25.00%   57.14%   60.71%
Hybrid                        300         0   42.86%   60.71%   71.43%
Semantic                      300        60   39.29%   64.29%   67.86%
Hybrid                        300        60   39.29%   75.00%   75.00%
Semantic                      800         0   35.71%   60.71%   71.43%
Hybrid                        800         0   42.86%   78.57%   85.71%
Semantic                      800       150   32.14%   71.43%   82.14%
Hybrid                        800       150   46.43%   85.71%   89.29%
Semantic                     1500         0   46.43%   67.86%   71.43%
Hybrid                       1500         0   39.29%   71.43%   78.57%
Semantic                     1500       300   39.29%   82.14%   82.14%
Hybrid                       1500       300   50.00%   78.57%   92.86%

Interpretation: Hybrid retrieval with chunk size 1500 and overlap
300 achieved the highest Hit@1 (50.00%) and Hit@5 (92.86%). Semantic
retrieval with the same chunk size and overlap achieved the highest
semantic Hit@3 result (82.14%). The selected collection was
poetry_1500_300.

Hit@k measures whether the expected relevant document appears within the
first k retrieved results. It measures retrieval performance, not the
factual correctness of the final generated answer.
5. Documentation Tools and Evidence Handling
The system provides tools that allow the Researcher to search, read, and
list documentation files. Tool calls are processed through a controlled
tool-execution loop.

The implementation includes argument validation, handling for unknown
tools and invalid arguments, JSON parsing and validation, and
duplicate-query handling. Tool execution details can be surfaced for
inspection.

During development, a read_doc argument mismatch was identified and
corrected. Tests also covered reading documentation such as
basic-usage.md and retrieving information about commands such as
poetry run.

To reduce token consumption and avoid context-limit errors, the amount
of documentation passed to agents was restricted. In particular,
selected documentation reads were limited, and the Writer received
collected evidence rather than the full documentation corpus.
6. Guardrails and Structured Output
6.1 Structured Output
The system uses a Pydantic schema named SupportAnswer to validate the
structure of generated answers. This helps ensure that the application
receives predictable fields rather than arbitrary unstructured text.
6.2 Guardrail Goals
The guardrails are intended to:
- Keep responses within the Poetry support scope.
- Ground technical claims in retrieved evidence.
- Avoid unsupported or invented commands.
- Preserve appropriate source references.
- Handle prompt-injection attempts safely.
- Validate the generated response structure.
6.3 Guardrail Examples
A small before-and-after set was used to inspect guardrail behavior. In
the four recorded examples, two unsupported TOML or lock-file claims
were removed, while two valid or unrelated answers were preserved. The
number of unsafe claims in those examples decreased from two to zero.

This is a limited set of examples and should not be interpreted as a
comprehensive security evaluation.
7. Evaluation Dataset and Methodology
The final dataset contains 40 cases:

  Case type            Number of cases
  Normal                            30
Special                            7
Prompt injection                   3
Total                     40

The project used several evaluation methods:
1. Automated phrase matching against expected key phrases.
2. Source matching against expected documentation references.
3. Manual assessment of selected answers.
4. Comparison of Reviewer ON and Reviewer OFF.
5. Prompt-injection and special-case assessment.
6. Automated unit and integration tests.
7. LangSmith tracing and experiment runs.
Phrase matching is a simple substring-based metric. It does not fully
measure semantic correctness. Source matching also does not necessarily
mean that a source is the only valid source for an answer. Manual
evaluation was therefore used to add context to selected results.

Incomplete runs caused by API limits or execution errors are not
automatically classified as incorrect answers. They are treated
separately from completed responses.
8. Architecture Comparison
An initial comparison was conducted on 30 normal cases. The recorded
results were:
  System              Completed Phrase matches       Coverage       Recorded
answers                                       errors
  Plain LLM               30/30 15/30 (50.00%)        100.00%              0

  RAG                     30/30 19/30 (63.33%)        100.00%              0

  Single-Agent            23/30 18/23 (78.26%)         76.67%              6
Multi-Agent             30/30 18/30 (60.00%)        100.00%              0
The Single-Agent phrase-match percentage uses only its 23 completed
answers as the denominator. Its coverage was lower than the other
systems. Therefore, its percentage should not be compared without
considering the number of completed cases.

A separate common-case comparison file contained 23 rows. Counting the
displayed Boolean phrase-match values gives:

  System           Phrase matches
  Plain LLM        14/23 (60.87%)
RAG              18/23 (78.26%)
Single-Agent     18/23 (78.26%)
Multi-Agent      17/23 (73.91%)

This 23-row comparison is a separate analysis and must not be merged
with the 30-case results above. Both comparisons use simple phrase
matching rather than a full semantic correctness assessment.
8.1 Multi-Agent Results on the 40-Case Dataset
The recorded multi_agent_results.csv contained 49 execution rows, 40
unique case IDs, 40 successful rows, and 9 incomplete attempts. Some
cases had multiple attempts.

For the 30 normal cases, the recorded results were:
- Phrase match: 18/30 (60.00%).
- Source match: 22/30 (73.33%).
- Both phrase and source matched: 13/30.
- Average model calls: 5.73.
The 49 execution rows should not be described as 49 unique test cases.
There were 40 unique case IDs, and repeated rows represented additional
attempts.
9. Manual Evaluation
Fifteen selected normal cases were manually assessed for the
Single-Agent and Multi-Agent systems.

  Manual assessment     Single-Agent   Multi-Agent
  Right                           11            11
Partly Correct                   1             1
Wrong                            1             3
Not Evaluable                    2             0
Total                           15            15

Among evaluable answers:
- Single-Agent: 11 fully correct out of 12 evaluable answers.
- Multi-Agent: 11 fully correct out of 15 evaluable answers.
The Multi-Agent system completed more of the selected cases, but this
did not translate into a higher number of fully correct answers in this
sample.
9.1 Notable Observations
- Case 6: The Multi-Agent answer correctly explained poetry run,
  but also included an outdated or potentially misleading optional
  reference to poetry shell.
- Case 9: Both systems missed poetry env list, even though the
  research stage had identified relevant evidence. This indicates a
  possible evidence-transfer problem between agents.
- Case 22: The Multi-Agent system incorrectly treated a question
  about extension-module build tools as out of scope.
- Case 23: The Single-Agent response described the
  poetry-install hook broadly correctly but overstated the guarantee
  about the environment matching the lock file before a commit.
- Cases 27 and 29: The Single-Agent runs had execution errors and
  were not evaluable, while the Multi-Agent system returned answers.
  The Multi-Agent answer for case 27 was still judged wrong.
These examples show why completion rate and phrase matching alone are
insufficient to establish answer quality.
10. Special Cases and Prompt-Injection Evaluation
10.1 Special Cases
The seven special cases (IDs 31--37) were manually reviewed for the
Multi-Agent system.

  Assessment         Number of cases
  Right                            6
Partly Correct                   1
Wrong                            0

The partially correct case involved a cloud deployment question. The
system avoided inventing an undocumented deployment command, but its
research evidence was limited.

Other cases examined scope handling, test parallelism, AWS Lambda
deployment, dependency resolution, lock-file behavior, and vulnerability
claims.
10.2 Prompt-Injection Cases
All three prompt-injection cases (IDs 38--40) were manually assessed as
Right.

This is a positive result for the tested examples, but three cases are
not enough to establish broad or comprehensive prompt-injection
resistance.

The Single-Agent runs for cases 31--40 had execution errors and did not
produce evaluable final answers in the recorded experiment. These should
be reported as Not Evaluable rather than as incorrect responses.
11. Reviewer ON/OFF Experiment
A separate experiment compared the Multi-Agent workflow with the
Reviewer disabled and enabled. Both configurations were evaluated on the
same ten successful case IDs (1--10).
11.1 Results
  Metric                             Reviewer OFF   Reviewer ON
  Successful unique cases                   10/10         10/10
Phrase matches                             9/10          9/10
Source matches                            10/10          8/10
Total model calls                            52            53
Average model calls per case                5.2           5.3
Reviewer approvals               Not applicable            10
Reviewer rejections              Not applicable             0
Review cycles                    Not applicable    1 per case

The Reviewer ON configuration increased total model calls from 52 to 53.
It did not improve phrase matching and had fewer source matches than
Reviewer OFF.

The confidence field was high for all ten Reviewer ON cases, compared
with medium confidence in the Reviewer OFF results. This is a change in
the model's reported confidence, not proof of improved factual accuracy.
11.2 Draft and Final Answer Length
The recorded Writer drafts contained 3,618 characters in total, while
the final answers contained 2,040 characters. This represents an
approximate 43.6% reduction in length.

Shorter responses may be easier to read, but reduced length does not
automatically mean better correctness or completeness. In one case, the
final answer omitted an evidence quotation while retaining the source
field.

Some source mismatches involved alternative documentation files that may
still have been relevant. Therefore, source-match results should be
interpreted with the expected-source definitions in mind.
11.3 Reviewer Conclusion
In this ten-case experiment, Reviewer ON did not demonstrate a
measurable improvement in phrase-match accuracy. Source matching was
lower, and model-call usage was slightly higher. The experiment
therefore does not support a claim that enabling the Reviewer improved
answer accuracy.

The Reviewer may still provide useful workflow control and an
opportunity to request revisions. However, the measured results show
that its independent contribution to answer quality requires further
evaluation.
11.4 Reviewer Rejection Path Test
A separate mocked unit test verified the coordinator's
rejection-and-revision control flow:

Reviewer Reject → Writer Revision → Reviewer Approve

The test passed. This confirms that the orchestration logic can handle a
rejection and subsequent revision under the mocked test conditions. It
does not prove that the live language model will reliably identify every
factual error or reject every flawed answer.
12. LangSmith Tracing and Experiments
LangSmith tracing was configured for the poetry-support-team project.
A successful LangGraph trace showed the workflow and nested agent
activity, including Router, Researcher, Writer, and Reviewer execution.

A recorded experiment named langgraph-support-team-86d8c866 contained
ten runs. The run history included a quota-related failure and technical
issues such as tool-choice behavior, duplicate or similar searches, and
empty responses.

An evaluator mapping issue was later identified: the evaluator read
incorrect dataset fields instead of the relevant output fields. The
evaluator was corrected to use the output fields for key phrases,
expected sources, and answerability. Missing phrase or source references
were treated as missing rather than automatically counted as matches.

The original experiment was not rerun after this evaluator correction.
Therefore, its earlier aggregate metrics are historical and preliminary,
not validated final metrics. A separate targeted AWS Lambda retry
succeeded, but that single-case run does not replace the ten-case
experiment.

LangSmith was useful for observing execution paths, latency, tool
activity, and failures. However, the experiment results must be
interpreted in light of the evaluator correction and API-related
failures.
13. Reliability and Resource Constraints
During development, the project encountered Groq API limits, including
rate-limit and token-per-minute errors. Some recorded incomplete
attempts were caused by these external limits rather than by the
intended workflow logic.

A context-size issue was also observed when a request exceeded the
available token limit. The project reduced the amount of documentation
passed to agents by limiting selected document reads and passing
collected evidence to the Writer and Reviewer.

These changes improved the ability to run targeted tests, but external
API limits remain a reliability constraint. A rate-limit response should
be distinguished from an incorrect answer or a workflow logic defect.

The workflow also applies a maximum of 12 model calls and up to 2 review
rounds to prevent unbounded execution.
14. Testing and Code Quality
The latest recorded full test-suite result was:

64 passed in 10.24 seconds.

The test suite included unit and integration coverage for important
components of the application. A separate mocked test for the Reviewer
rejection, Writer revision, and Reviewer approval path also passed:

1 passed in 8.02 seconds.

The rejection-path test is included in the latest full-suite count only
if it was part of that test run; the recorded latest full-suite result
is 64 passed.

Ruff and Poetry checks were also recorded as passing in the project
notes. These results refer to the recorded development checks and should
not be interpreted as proof that every possible runtime scenario has
been tested.
15. Limitations
The project has several limitations:
1. Evaluation metrics: Phrase matching is based on simple text
   matching and does not fully measure semantic correctness.
2. Limited manual sample: Manual assessment covered selected cases
   rather than every generated response.
3. Reviewer effectiveness: The ten-case ON/OFF experiment did not
   demonstrate improved phrase-match accuracy.
4. Source matching: A mismatch against an expected source does not
   always mean that the cited source is invalid.
5. Prompt-injection sample size: Only three prompt-injection cases
   were evaluated.
6. API constraints: Rate limits and token limits caused incomplete
   attempts and affected experiment execution.
7. Latency: Multi-agent execution can require several model calls
   and may increase response time.
8. Evidence transfer: Relevant evidence retrieved by the Researcher
   may not always appear in the final answer.
9. Experiment consistency: Some historical experiments were not
   rerun after evaluator or implementation changes.
16. Future Improvements
Potential future improvements include:
1. Semantic evaluation: Use semantic similarity and structured
   human evaluation instead of relying mainly on substring matching.
2. Reviewer quality: Improve the Reviewer's ability to identify
   unsupported claims, missing evidence, and incomplete answers.
3. Evidence transfer: Ensure that relevant research findings are
   preserved and considered by the Writer.
4. Retrieval optimization: Explore reranking and more advanced
   retrieval strategies.
5. Reliability: Improve retry handling, backoff behavior, and
   recovery from API rate limits.
6. Performance: Reduce latency and token usage while preserving
   answer quality.
7. Security testing: Expand prompt-injection testing with a larger
   and more diverse test set.
8. Deployment: Deploy the Streamlit application and evaluate it in
   a production-like environment.
9. Evaluation reproducibility: Rerun experiments after evaluator
   changes and preserve versioned datasets, prompts, and result files.
17. Conclusion
The Poetry Support Team project demonstrates an implementation of a
documentation-grounded, multi-agent technical support assistant. It
combines RAG, ChromaDB, documentation tools, LangGraph orchestration,
Pydantic validation, guardrails, and a Streamlit interface.

The retrieval experiments identified useful configurations, with hybrid
retrieval using 1500-character chunks and 300-character overlap
achieving the highest Hit@1 and Hit@5 results among the tested
configurations. The architecture comparisons showed that RAG and
multi-agent approaches can provide useful capabilities, but no single
metric establishes that the Multi-Agent system is consistently more
accurate.

Manual evaluation highlighted both successful answers and important
failure modes, including missed commands, incorrect scope decisions, and
evidence-transfer problems. The Reviewer ON/OFF experiment did not show
an improvement in phrase matching and slightly increased model-call
usage. The mocked rejection-path test verified the coordinator's
revision flow, while the full automated test suite recorded 64 passing
tests.

Overall, the project provides a functional foundation for Poetry-focused
AI support and demonstrates practical techniques for retrieval, agent
coordination, structured responses, guardrails, and evaluation. Its
results also emphasize the importance of careful metrics, reproducible
experiments, and human review when assessing AI-generated technical
answers.
References
- Poetry Documentation
- LangGraph
  Documentation
- ChromaDB Documentation
- Streamlit Documentation
- LangSmith Documentation
- Pydantic Documentation
- Sentence Transformers Documentation
- Groq Documentation
Final Submission Checklist
- [ ] Add real screenshots of the Streamlit application.
- [ ] Confirm that screenshots correspond to the final implementation.
- [ ] Confirm the final source-code version and dependencies.
- [ ] Include the relevant evaluation CSV files if required.
- [ ] Include the final test output showing 64 passing tests.
- [ ] Ensure that any claims about deployment are made only if
  deployment was completed.
- [ ] Review formatting, spelling, and references before submission.