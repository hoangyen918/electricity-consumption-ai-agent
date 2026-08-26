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
