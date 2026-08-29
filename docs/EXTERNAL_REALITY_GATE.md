# MathModel AI — External Reality Gate (Phase 7A)

## Reality Levels

| Level | Meaning |
|-------|---------|
| CONTRACT_VERIFIED | Schema/interface behavior verified with mocks |
| REAL_EXECUTION_VERIFIED | Real external service returned real data |
| REAL_MODEL_QUALITY_SMOKE_VERIFIED | Real LLM produced valid domain output |
| BENCHMARK_VERIFIED | Full benchmark quality evaluation |

These are NOT the same thing. A valid JSON response only proves
REAL_EXECUTION_VERIFIED.

## RealityContext

Phase 7A default:
- reality_required = true
- mock_allowed = false
- fixture_allowed = false
- real_llm_required = true
- real_search_required = true
- semantic_verification_required = true

## No Silent Mock Rule

When reality_required=true, any real provider failure must NOT silently
fall back to MockProvider. Fallback is only allowed with explicit policy,
and the run status must be REALITY_COMPROMISED.

## Components

- `reality/` — RealityContext, RealityTrace, ExternalRealityGate
- `providers/deepseek.py` — DeepSeekProvider (OpenAI-compatible endpoint)
- `literature/search.py` — BaseLiteratureSearchProvider, CrossrefSearchProvider
- `agents/semantic_agents.py` — SemanticRedTeamAgent, ClaimSupportVerifier, SemanticPaperReviewer

## Live Testing

```bash
# Normal suite (no credentials needed):
python -m pytest tests/ -q

# Live tests (requires real credentials/network):
pytest -m live -v
pytest -m external -v
```

Live tests SKIP honestly when credentials are absent and report
REALITY_NOT_VERIFIED — a skip never makes the gate PASS.

## Live Verification Record (this environment)

Real Crossref search executed: 3 queries, 6 real records (is_fixture=false),
1 real DOI lookup succeeded. See phase7a_live_report.json.
Real LLM: NO CREDENTIAL — live tests skipped. Reality: NOT VERIFIED.
