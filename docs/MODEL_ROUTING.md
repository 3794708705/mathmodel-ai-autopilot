# MathModel AI — Model Routing

## Architecture

```
Task → TaskProfiler → RoutingPolicy → ModelRouter → ModelProvider → Selected Model
```

## TaskProfile

Every task is profiled with:
- `task_type` — classification of the task
- `complexity` — estimated complexity tier
- `reasoning_requirement` — depth of reasoning needed
- `math_requirement` — mathematical sophistication needed
- `coding_requirement` — code generation complexity
- `multimodal_requirement` — whether images/attachments are involved
- `long_context_requirement` — context window needs
- `review_requirement` — review/verification intensity
- `security_risk` — security sensitivity
- `blast_radius` — impact of failure
- `cost_sensitivity` — budget constraints
- `deadline_pressure` — time constraints
- `retry_count` — current retry attempt

## Model Levels

| Level | Name | Use Case | Recommended |
|-------|------|----------|-------------|
| 4 | Critical | Architecture, MathModeler, ModelRepair, Security | FLAGSHIP_MAX |
| 3 | Complex | Workflow, ProblemAgent, ModelExplorer, Validation | FLAGSHIP_XHIGH |
| 2 | Normal | CRUD, Parsers, Provider adapters, Tests | BALANCED |
| 1 | Routine | README, formatting, minor cleanup | FAST |

## Escalation Chain

```
FAST → BALANCED → FLAGSHIP_HIGH → FLAGSHIP_XHIGH → FLAGSHIP_MAX → MULTI_MODEL_REVIEW → HUMAN_REVIEW
```

## Minimum Levels by Task Type

- `mathematical_modeling`: minimum FLAGSHIP_HIGH
- `sandbox_security`: minimum FLAGSHIP_XHIGH
- `documentation`: minimum FAST
- `code_generation`: minimum BALANCED
- `data_analysis`: minimum BALANCED
- `validation`: minimum FLAGSHIP_HIGH

## Quality Gate

Every critical node must pass a Quality Gate checking:
- Schema validation
- Test results
- Solver status
- Constraint checking
- Execution status
- Citation validation
- Artifact existence
- Critical error count

Output: PASS | RETRY | ESCALATE | HUMAN_REVIEW