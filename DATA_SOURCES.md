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

| Test | DeepSeek V3.2 | Qwen3.6 27B |
|------|---------------|-------------|
| Vietnamese fluency | Pass | Pass |
| Tool calling (get_dataset_summary) | Pass | Pass |
| Tool calling (calculate) | Pass | Pass |
| Refuses to fabricate numeric answer without data | **FAIL** — invented "3321.8 kWh" with no supporting data or tool call | Pass — asked for the underlying data instead |
| Honest about missing knowledge base | **FAIL** — answered confidently with specific percentages, no caveat | Partial pass — gave estimated figures but flagged them as unverified and recommended checking the source report |

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

| Original column | Processed column | Type | Unit | Notes |
|---|---|---|---|---|
| DateTime | datetime | datetime | - | 10-minute interval |
| Temperature | temperature | float | unspecified (likely °C) | |
| Humidity | humidity | float | unspecified (likely %) | |
| Wind Speed | wind_speed | float | unspecified | |
| general diffuse flows | general_diffuse_flows | float | unspecified | solar radiation proxy |
| diffuse flows | diffuse_flows | float | unspecified | solar radiation proxy |
| Zone 1 Power Consumption | zone_1_consumption | float | **not specified by UCI — do not assume kWh/kW** | |
| Zone 2 Power Consumption | zone_2_consumption | float | **not specified by UCI — do not assume kWh/kW** | |
| Zone 3 Power Consumption | zone_3_consumption | float | **not specified by UCI — do not assume kWh/kW** | |

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

| Fact | Value | Source |
|---|---|---|
| Rural household electrification rate | 99.74% | EVN Annual Report 2024-2025, p.9 |
| Total commercial electricity output, 2023 | 253.5 billion kWh (+4.46% YoY) | EVN Annual Report 2022-2023, p.10 |
| Total power production and purchase, 2023 | 271.68 billion kWh (+3.68% YoY) | EVN Annual Report 2022-2023, p.10 |
| Power loss rate | 6.05% (improved 0.2pp) | EVN Annual Report 2022-2023, p.10 |

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