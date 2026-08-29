# Data Sources

TODO: documented in later milestone.

## Model Selection Decision

**Chat model:** `qwen/qwen3.6-27b` via OpenRouter
**Embedding model:** `openai/text-embedding-3-small` via OpenRouter
**Decision date:** 2026-08-26

### Rationale

Two candidate models were compared using scripts/compare_models.py against 5 Vietnamese
test questions covering: natural language quality, hallucination resistance (no ground-truth
data provided), tool-calling correctness, and honesty when no knowledge base is available.

| Test                                             | DeepSeek V3.2                                                                | Qwen3.6 27B                                                                                                      |
| ------------------------------------------------ | ---------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| Vietnamese fluency                               | Pass                                                                         | Pass                                                                                                             |
| Tool calling (get_dataset_summary)               | Pass                                                                         | Pass                                                                                                             |
| Tool calling (calculate)                         | Pass                                                                         | Pass                                                                                                             |
| Refuses to fabricate numeric answer without data | **FAIL** — invented "3321.8 kWh" with no supporting data or tool call | Pass — asked for the underlying data instead                                                                    |
| Honest about missing knowledge base              | **FAIL** — answered confidently with specific percentages, no caveat  | Partial pass — gave estimated figures but flagged them as unverified and recommended checking the source report |

DeepSeek V3.2 was rejected due to a critical hallucination failure on the core numeric-accuracy
requirement (rubric B2/C4 of the project brief: "never fabricate numbers, never fabricate sources").
Qwen3.6 27B was selected as the sole chat model for V1. No fallback model is configured; if the
Qwen endpoint is unavailable, OpenRouter automatically retries across its other hosting providers
for the same model.

Raw comparison log preserved at: scripts/compare_models.py (run manually, not part of CI).

## Structured Dataset

**Name:** Power Consumption of Tetouan City
**Provider:** UCI Machine Learning Repository
**Dataset ID:** 849
**URL:** https://archive.ics.uci.edu/dataset/849/power+consumption+of+tetouan+city
**DOI:** 10.24432/C5B034
**License:** CC BY 4.0
**Citation:** Salam, A. & El Hibaoui, A. (2018). Power Consumption of Tetouan City [Dataset]. UCI Machine Learning Repository.

### Original characteristics

- 52,417 observations (Jan 1 – Dec 30, 2017), 10-minute intervals
- No missing values reported by UCI
- Collected by SCADA system of Amendis (Tetouan, Morocco public utility)

### Important caveat — units not specified

UCI's published metadata does not specify a physical unit (kW, kWh, etc.) for the
three Zone Power Consumption target variables. This project therefore does NOT
infer, display, or assume a unit for these columns anywhere in tool outputs,
prompts, or the UI. Values are reported as raw dataset units only.

### Data dictionary (processed column names)

| Original column          | Processed column      | Type     | Unit                                                   | Notes                 |
| ------------------------ | --------------------- | -------- | ------------------------------------------------------ | --------------------- |
| DateTime                 | datetime              | datetime | -                                                      | 10-minute interval    |
| Temperature              | temperature           | float    | unspecified (likely °C)                               |                       |
| Humidity                 | humidity              | float    | unspecified (likely %)                                 |                       |
| Wind Speed               | wind_speed            | float    | unspecified                                            |                       |
| general diffuse flows    | general_diffuse_flows | float    | unspecified                                            | solar radiation proxy |
| diffuse flows            | diffuse_flows         | float    | unspecified                                            | solar radiation proxy |
| Zone 1 Power Consumption | zone_1_consumption    | float    | **not specified by UCI — do not assume kWh/kW** |                       |
| Zone 2 Power Consumption | zone_2_consumption    | float    | **not specified by UCI — do not assume kWh/kW** |                       |
| Zone 3 Power Consumption | zone_3_consumption    | float    | **not specified by UCI — do not assume kWh/kW** |                       |

### Derived columns (added by this project, not part of original dataset)

`year`, `month`, `day`, `hour`, `weekday`, `date` — extracted from `datetime` by
`scripts/prepare_data.py`. These are project-generated convenience columns for
grouping/aggregation, not original UCI fields.

### File locations

- `data/raw/tetouan_raw.csv` — untouched download, gitignored (regenerate with `python scripts/download_data.py`)
- `data/processed/tetouan_clean.csv` — cleaned + renamed + derived features, gitignored (regenerate with `python scripts/prepare_data.py`)
- `data/sample/tetouan_sample.csv` — first 1,000 rows of processed data, **committed to git** so the repo is runnable immediately after clone without re-downloading

### Document format note

Both EVN annual reports obtained are the condensed public-facing format
(19 pages each), not the full statutory audited financial statement filing.
This is EVN's standard public annual report format for general distribution.

Extraction coverage: 19/19 pages (2024-2025), 17/19 pages (2022-2023) —
the 2 skipped pages in the 2022-2023 report contain no extractable text layer
(likely cover/section-divider images).

Total corpus after chunking: 48 chunks, 21,780 tokens across both documents.

## CRITICAL BUG FOUND AND FIXED — date-range filtering silently excluded last day's data

**Discovered:** M12, during final eval investigation. **Present since:** M2 (undetected for the
entire project duration until now).

**Root cause:** `aggregate_metric`, `compare_periods`, and `calculate_correlation` all filtered
date ranges using `datetime <= pd.to_datetime(end_date)`, where `pd.to_datetime()` defaults to
00:00:00. This silently kept only the single midnight row of the end date and discarded the
remaining 143 ten-minute readings of that day — a systematic undercount affecting every
date-range query in the project (e.g., "tháng 7" queries used 4,321 rows instead of the correct
4,464; single-day queries like Gate 2's "temperature on peak day" used only 1 row instead of a
true 144-row daily average).

**Why it went undetected:** All prior tests checked tool selection and answer plausibility, not
exact numeric correctness against an independently computed ground truth — so a systematically-
biased-but-plausible-looking number passed every check for the project's full duration.

**Fix:** changed the upper bound to an exclusive comparison against the start of the next day
(`datetime < end_date + 1 day`), correctly including the full end date regardless of data
granularity. Added explicit regression tests (`test_aggregate_metric_end_date_includes_full_last_day`,
`test_aggregate_metric_multi_day_range_includes_all_days`) asserting exact expected row counts,
so a recurrence would be caught immediately rather than silently passing plausibility checks.

**Lesson:** plausibility-based test assertions ("answer contains a number", "correct tool was
called") do not catch systematic numeric bias — only assertions against an independently
verified exact expected value can. This is a genuine limitation of the eval design used
throughout most of this project's development.

## Provider glitch mitigation — empty content retry

**Discovered:** M12, during eval failure investigation. Across 106 agent calls (2 full eval
runs of 53 cases), 7 calls (~6.6%) returned a "successful" API response with no tool_calls
and empty/None content — the LLM equivalent of a silent no-op. Targeted reproduction (15
repeated calls on the 3 specific questions that exhibited this) failed to reproduce it
(0/15), and the affected questions differed between the two eval runs, indicating a
transient, content-independent glitch — most likely provider-side (OpenRouter routes to
multiple backend providers per request; occasional malformed/empty responses from one
provider are plausible) rather than a logic bug tied to specific prompts.

**Fix:** added a retry specifically for this case (empty content + no tool_calls despite a
"successful" API response), separate from MAX_AGENT_ITERATIONS (which governs the tool-calling
loop, not provider-response recovery). Up to 2 retries; if still empty after retries, the
agent returns an explicit fallback message rather than surfacing an empty/None answer to the
user. Verified via deterministic mocked tests (test_empty_content_retry_recovers,
test_empty_content_retry_exhausted_returns_fallback) since the real glitch cannot be reliably
reproduced on demand.

## Known Limitations — Retrieval Quality

### Cross-lingual retrieval gap

Query embeddings in Vietnamese show measurably weaker match quality (distance
~1.22-1.28) than English queries against the same English-language PDF content
(distance ~0.94-1.14), tested against `text-embedding-3-small`. This is a known
limitation of using a single embedding pass without query translation or a
bilingual reranking step. Vietnamese queries still retrieve relevant results in
most cases, but rank position and confidence are lower than an English-phrased
equivalent query.

### Chart/infographic content extraction

Both EVN reports contain data visualizations (bar/pie charts) rendered as
vector graphics with text labels. PyMuPDF extracts the text labels in PDF
draw-order, not visual reading-order, which scrambles the association between
category labels, years, and percentage values in these sections (example:
page 9 of the 2022-2023 report, sector-breakdown chart). This project does not
implement chart/table structure recovery (out of scope for V1). Any retrieval
result identified as originating from a chart-derived chunk should be treated
as unreliable for precise attribution; the agent's system prompt (M7) directs
it to acknowledge uncertainty rather than assert a specific number-to-category
mapping when the retrieved text is this kind of scrambled content.

### Verified clean facts used for demo/evaluation

The following facts were manually verified as cleanly extracted (complete
sentences, unambiguous number-to-subject association) and are used as the
primary Knowledge QA test questions:

| Fact                                      | Value                           | Source                            |
| ----------------------------------------- | ------------------------------- | --------------------------------- |
| Rural household electrification rate      | 99.74%                          | EVN Annual Report 2024-2025, p.9  |
| Total commercial electricity output, 2023 | 253.5 billion kWh (+4.46% YoY)  | EVN Annual Report 2022-2023, p.10 |
| Total power production and purchase, 2023 | 271.68 billion kWh (+3.68% YoY) | EVN Annual Report 2022-2023, p.10 |
| Power loss rate                           | 6.05% (improved 0.2pp)          | EVN Annual Report 2022-2023, p.10 |

**Update — conflicting figures discovered between EVN reports (found during M11 evaluation):**
EVN Annual Report 2024-2025, page 7, states 'Total power production and purchase' for 2023 as
280,814 million kWh (280.814 billion kWh) — this conflicts with the 271.68 billion kWh figure
for the same metric and year sourced from EVN Annual Report 2022-2023, page 10 (used above).
Both figures are genuinely present in their respective official PDFs (verified, not a
retrieval or hallucination artifact) — the discrepancy likely reflects a preliminary estimate
in the earlier report being revised in the later one. The system prompt now instructs the
agent to prefer the most recently published report's figures when such conflicts arise, and
to flag the discrepancy transparently rather than silently choosing one value.

### Multi-domain corpus vocabulary collision (added when policy directive KB was introduced)

Adding Chỉ thị 20/CT-TTg (policy directive on electricity saving) as a third
KB source revealed a compounding retrieval weakness: short, generic Vietnamese
queries about "hộ dân có điện" (households having electricity access) share
surface vocabulary ("điện", "hộ gia đình", "tiết kiệm điện") with the policy
directive's content about electricity *conservation*, despite being
semantically unrelated topics. For the test query "Tỷ lệ hộ dân nông thôn có
điện là bao nhiêu?", the correct chunk (EVN Annual Report 2024-2025, page 9)
drops to rank #11 out of 62 chunks, with a notably flat distance distribution
(1.29-1.46) indicating weak discriminative signal for this query phrasing.

This compounds a pre-existing issue: rank #1 for this query is EVN Annual
Report 2022-2023, page 9 — the chart-derived content already flagged as
unreliable in the "Chart/infographic content extraction" limitation above —
which outranks the correct source independently of the Chỉ thị 20 addition.

Mitigation applied: RAG_TOP_K increased from 8 to 10, maintaining a consistent
~16% selectivity ratio as the corpus grew from 48 to 62 chunks. This does not
fully resolve this specific short-query case (marked as a known xfail in
tests/test_retriever.py). Verified production behavior (the agent's own
tool-generated queries, which tend to be longer and more contextual) correctly
retrieves and cites this fact — see tests/test_agent.py::test_gate_3_knowledge_question,
which passes. The retriever-level unit test uses an intentionally terse
synthetic query that is not representative of actual agent-generated queries.

Root cause is a structural limitation of single-pass dense embedding retrieval
across a small, multi-domain corpus without domain-aware filtering or query
expansion — considered out of scope for V1, consistent with the project's
existing decision to defer hybrid search and reranking to V2.

### Score threshold for weak retrieval matches

SCORE_THRESHOLD = 1.3 was determined empirically (not guessed) by measuring
distances of known-correct chunks (VERIFIED_CASES) against a known-nonsense query, on
2026-08-27. This filters out results server-side when ALL retrieved candidates are too
dissimilar to be reliable, per the original requirement (yêu_cầu_đề.md §2.3: "Nếu
retrieve không ra gì liên quan → agent nói 'không tìm thấy', không đoán").

Initial measurement suggested 1.15 (highest known-good distance 1.0529, plus a small
safety margin, well below the nonsense-query floor of ~1.64). This was tightened too
aggressively: it broke two previously-passing tests using a shorter phrasing of a
known-good query ("Tỷ lệ tổn thất điện năng là bao nhiêu?", without "của EVN"), whose
closest match had distance 1.1618. Per the project's explicit policy of never tightening
a threshold to force a regression to pass, the value was loosened to 1.3 instead of
adjusting the test. All 2 non-xfail VERIFIED_CASES, the nonsense-query test, and all 4
production Gate tests (tests/test_agent.py) were re-verified passing at this value.

This does not fully separate "good" and "bad" matches cleanly — the small, multi-domain
corpus produces overlapping distance ranges between weak-but-correct and clearly-irrelevant
content (see "Multi-domain corpus vocabulary collision" above). The threshold is set
conservatively to avoid false negatives (incorrectly rejecting real information) at the
cost of not catching every weak match; the system prompt's instruction to hedge on
ambiguous retrieved content remains the primary defense for borderline cases.

### Update after IEA corpus expansion (62 -> 6,127 chunks)

After adding 14 IEA reports (World Energy Outlook 2019-2025, Energy Efficiency 2019-2025,
6,065 new chunks), the nonsense-query distance floor dropped from 1.6356 to 1.4624 (safety
margin from SCORE_THRESHOLD=1.3 narrowed from 0.58 to 0.41). Both known-correct-case
distances were unchanged (as expected — cosine distance between two fixed vectors is
independent of corpus size), only their RANK shifted due to competition from new content.
Verified: the interference in all 3 known test cases came from Chỉ thị 20 (pre-existing
vocabulary collision, see above), NOT from the new IEA content — no IEA chunk appeared in
the top-15 results for any Vietnamese-language EVN-specific test query, likely because the
same cross-lingual embedding gap that is a known weakness elsewhere incidentally acts as a
buffer here. SCORE_THRESHOLD=1.3 kept unchanged (still valid, margin positive both directions)
but this margin should be re-measured before any further significant corpus expansion.

document_type filtering (already available via search_knowledge_base's existing
document_type parameter, added in M10c) was empirically shown to improve ranking for
EVN-specific queries by excluding Chỉ thị 20 and IEA content: the known xfail case
improved from rank #11 to rank #5 when filtered to document_type='annual_report'.
The system prompt now instructs the agent to apply this filter when a question is
clearly EVN-report-specific.

## Chunking granularity — mitigation applied

Root cause identified during M5 testing: page 10 of the 2022-2023 report packs
multiple distinct financial topics (commercial electricity output, total
production, power loss rate) into the same page-level chunk under the current
page+token chunking strategy, diluting the chunk's embedding signal. This
caused the correct chunk for a commercial-electricity-output query to rank
#6 out of 48 — just outside a top-5 cutoff.

Mitigation applied: default RAG_TOP_K increased from 5 to 8. Justified by
small corpus size (48 chunks total) where retrieving 8 is still a highly
selective ~17% of the corpus, at negligible additional query cost.

This does not fix the root cause. A proper fix (smaller chunk_size or
semantic/topic-aware chunking instead of raw page+token splitting) is
planned V2 work, consistent with the chunk-size experimentation (500/700/900
tokens) already anticipated in the original project plan.

### Sample size caveat on the Gate 4 fix

The systemic prompt fix (calculate tool enforcement) was validated against 11
runs (10 manual + 1 pytest), improving from ~87% to 100% calculate-invocation
rate. Given the original failure rate was low (~13% on a 15-run baseline), 11
consecutive passes is an encouraging signal but not statistically conclusive
proof of a 0% failure rate — a residual low-single-digit-percent failure rate
cannot be ruled out from this sample size. This is noted here rather than
re-tested further at this stage to avoid disproportionate cost/time spent on
a low-severity residual risk; it should inform the evaluation methodology in
M11 (evals/run_evals.py), which should run each test case multiple times where
feasible rather than treating a single pass as conclusive, especially for
gate-critical behaviors like tool-selection compliance.

### Tool selection — peak-date reasoning (fixed) and residual over-verification pattern

An earlier version of the system prompt caused the agent to occasionally infer
a "peak date" by aggregating with group_metric(group_by="date", aggregation="sum")
instead of calling find_peak directly — these answer different questions
(aggregated period totals vs. the single global extreme record) and the
mismatch caused wrong dates, self-correction, and in one observed case,
hitting MAX_AGENT_ITERATIONS. A prompt rule was added instructing the agent
to call find_peak directly for "which moment had the single highest/lowest
value" questions, reserving group_metric for genuinely grouped comparisons.

Measured impact (10 runs of the affected question, before vs. after):

- Wrong-strategy path via group_metric to infer peak date: 100% of failure
  cases → 0/10 runs after the fix.
- hit_max_iterations rate: reduced from occurring reliably in the failure
  path to 1/10 runs (10%).
- Average iterations: reduced from a fixed 5 (on failure) to a mean of 3.8
  (median 4).

Residual case (1/10): the agent reached the correct answer by iteration 3
(find_peak → aggregate_metric) but then performed two unnecessary
"double-check" calls (a redundant aggregate_metric with count, followed by
a group_metric(group_by="date") call — again a large-payload tool), exhausting
the iteration budget before producing a final answer. This is a distinct
over-verification pattern, not the peak-date misidentification issue above,
and was not addressed in this fix to avoid an untested prompt change that
could reduce the agent's appropriate caution in other contexts. Left as a
known limitation for V2 consideration.

## Security Testing Note

Prompt injection tests (tests/test_security.py) inject malicious content
directly into the message history with role="user" to simulate what the
agent would see after a knowledge-base retrieval, rather than exercising the
full path (agent decides to call search_knowledge_base → Chroma returns the
malicious chunk → content arrives as role="tool" in the message history).
This is a reasonable approximation given the system prompt's defense
instruction applies to "content retrieved via search_knowledge_base"
regardless of the message role it arrives under, but it is a narrower test
than a full end-to-end scenario where the agent autonomously retrieves the
malicious document. Both tests passed, with the model explicitly noting it
was disregarding the injected instruction — a stronger result than the
minimum bar of simply not complying. A true end-to-end injection test
(poisoning the actual vector store temporarily and letting the agent
autonomously retrieve it via a real question) is a more rigorous version
of this test, left as potential future hardening given time constraints.


## Known limitation — search-until-exhausted pattern (not fixed, documented for V2)

Final evaluation (53 cases, 94.3% pass) surfaced 2 cases (both_007, parametric_001) sharing
an identical root cause: the agent calls search_knowledge_base repeatedly (5 times, with
varied queries) without ever concluding, exhausting MAX_AGENT_ITERATIONS before producing
an answer — even though both_007's correct answer required only 1 lookup + 1 calculate, and
parametric_001's correct behavior (refuse — the year isn't in the KB) was reached in content
but not before hitting the iteration limit.

This is the third observed instance of this pattern in the project (see also the M8 finding
on group_metric-based over-verification), indicating a systemic gap rather than isolated
noise: the agent has no explicit guidance on when to stop searching and commit to an answer
(or a refusal) versus trying another query phrasing.

A targeted fix (a system prompt rule capping search_knowledge_base retries with varied
phrasing, e.g. "after 2-3 search attempts, answer with the best available evidence or refuse
clearly, rather than continuing to reformulate the query") was considered but deliberately
NOT applied at this stage of the project, given the proximity to the final deliverable
deadline and the risk that an imprecise prompt change could cause the agent to give up too
early on legitimately multi-step questions. Documented here as a concrete, well-understood
V2 improvement rather than an unknown gap.

kb_001 also failed in this run (content_ok) but for an unrelated reason: the agent correctly
cited 99.6% (2023) from the trend table rather than the calibrated ground-truth value 99.74%
(2024) — both are genuine figures from the same table in EVN Annual Report 2024-2025, page 9;
this is an eval-calibration ambiguity (which year the question implicitly asks about), not an
agent or retrieval error.
