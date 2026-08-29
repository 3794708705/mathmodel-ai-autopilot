# MathModel AI — Reasoning Core

## Overview

Phase 2 implements the reasoning pipeline that transforms a competition problem
into a selected mathematical model with backup.

## Pipeline

```
Problem Text
    ↓
ProblemAgent (UNDERSTAND)
    ↓
ProblemAnalysis (typed schema)
    ↓
ModelExplorer (EXPLORE)
    ↓
ModelCandidate[] (typed schema)
    ↓
EligibilityGate
    ↓
Eligible Candidates
    ↓
ModelJury (SELECT)
    ↓
Selected Model + Backup Model
    ↓
LiteratureAgent
    ↓
LiteratureQueryPlan (SEARCH_PENDING)
```

## Domain Schemas

### EvidenceItem
- `evidence_id`, `type` (FACT/DATA/PROPOSED_ASSUMPTION/ACCEPTED_ASSUMPTION/DERIVATION)
- Invariants: FACT cannot become ASSUMPTION; PROPOSED_ASSUMPTION needs explicit acceptance; DERIVATION must reference sources

### ProblemAnalysis
- `background`, `core_problem`, `objectives`
- `subproblems[]` with `subproblem_id`, `normalized_goal`, `task_types`, `dependencies`
- `evidence[]` with typed evidence items
- `ambiguities[]` with interpretations and impact
- Self-validating: duplicate IDs, invalid references, evidence invariants

### ModelCandidate
- `candidate_id`, `name`, `model_family`
- `components[]` supporting model chains
- `required_data`, `required_assumptions`, `strengths`, `weaknesses`
- No fabricated results

### EligibilityResult
- `eligible` (boolean), `hard_failures[]`, `warnings[]`
- Policy: hard failures (missing data, math implausible, constraint violations)
- Soft warnings (implementation difficulty, validation complexity)

### ModelJuryResult
- `candidate_scores[]` with raw + weighted scores
- Deterministic: LLM provides raw scores, Python computes weighted totals
- `selected_model`, `backup_model`, `ranking`
- Validates: selected != backup, both in ranking

### LiteratureQueryPlan
- `queries[]` with `purpose`, `keywords`, `status`
- All queries: `SEARCH_PENDING` — no fabricated results

## Agents

| Agent | Stage | Input | Output |
|-------|-------|-------|--------|
| ProblemAgent | UNDERSTAND | raw_problem | ProblemAnalysis |
| ModelExplorer | EXPLORE | ProblemAnalysis | ModelCandidate[] |
| EligibilityGate | (pre-Jury) | ModelCandidate[] | EligibilityResult[] |
| ModelJury | SELECT | Eligible candidates | ModelJuryResult |
| LiteratureAgent | (post-Jury) | Analysis + Candidates | LiteratureQueryPlan |

## Routing (TD-1)

RoutingPolicy now considers:
- `complexity`, `task_type`, `retry_count` (Phase 1)
- `reasoning_requirement`, `math_requirement`, `review_requirement` (TD-1)
- `long_context_requirement`, `blast_radius`, `coding_requirement` (TD-1)

Every routing decision produces a `RoutingExplanation` with:
- `selected_tier`, `primary_reasons`, `triggered_rules`
- `escalation_reason`, `task_profile_snapshot`

## State Persistence

Domain objects → Pydantic validation → `model_dump()` → SQLAlchemy JSON → ProblemState.

Key helpers in `domain/state_helpers.py`:
- `store_analysis()` / `load_analysis()`
- `store_candidates()` / `load_candidates()`
- `store_jury_result()` / `load_jury_result()`
- `record_revision()` for stage history

## Competition Fixtures

- **Fixture A**: Resource allocation / optimization (LP, ILP, Goal Programming)
- **Fixture B**: Prediction + optimization chain (ARIMA + Newsvendor, LSTM + Stochastic, Exp Smoothing + Safety Stock)
- **Fixture C**: Network / routing / evaluation (CVRP: IP, Clarke-Wright, Network Flow)