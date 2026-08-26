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