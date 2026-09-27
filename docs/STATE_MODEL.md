# MathModel AI — State Model

## ProblemState

The unified state object that all agents operate on. No agent maintains private facts.

### Core Fields

```
run_id
problem_id
project_id
title
raw_problem

competition
deadline
remaining_hours

files

background
objectives
subproblems

facts
data_sources
assumptions
ambiguities
constraints

candidate_models
selected_model
backup_model

variables
parameters
units
equations

objective
model_constraints
algorithm

code_files
execution_records
results

validation_results
sensitivity_results
robustness_results

literature
citations

red_team_reports
revisions

figures
tables

paper_state
submission_state

current_stage
status
```

## Workflow State Machine

```
INGEST → UNDERSTAND → DATA → LITERATURE → EXPLORE → SELECT →
MODEL → SOLVE → VALIDATE → SENSITIVITY → ROBUSTNESS → RED_TEAM →
PAPER → FINAL_JURY → SUBMISSION → FINAL
```

### Rollback Support

```
MODEL → SOLVE → VALIDATE → RED_TEAM → FAIL → MODEL_REPAIR → SOLVE → VALIDATE → RED_TEAM
```

Max auto-repair cycles: 3. After 3 failures: HUMAN_REVIEW.

### What the Autopilot Runtime Actually Walks

The runtime names its stages after the work it is doing (`intake`, `codegen`,
`audit`), and `autopilot/ledger.py` maps each name onto the phases above:

```
intake/clarify → INGEST → understand/registries → UNDERSTAND → explore →
EXPLORE → select → SELECT → model → MODEL → codegen/solve → SOLVE →
verify/evidence → VALIDATE → figures/tables/paper → PAPER →
audit → FINAL_JURY → pdf/package → SUBMISSION → final_check → FINAL
```

A run has no separate `DATA` or `LITERATURE` stage, goes to `PAPER` straight
after verification, and returns to `MODEL` when verification fails. Those
edges are declared in `VALID_STAGE_TRANSITIONS` (`api/routes.py`) so one table
describes both the API and a real run.

`select` runs the eligibility gate before the jury (`EligibilityGate` →
`ModelJury`), so a candidate that fails a hard eligibility check is never
scored. The gate's verdict and the number of candidates it admitted are
recorded in the stage detail.

### What a Resume Re-runs

The reasoning stages (`intake`, `understand`, `explore`, `select`, `model`) and
the verified solver run resume from the artifacts on disk. The publication
half resumes too, but only as a unit: `figures`, `tables` and `paper` are
reused when `artifacts/paper/publication_inputs.json` — a fingerprint of the
verified outcome, the model version, the verified statistics, the subproblems
and the output summaries — still matches, and when every recorded figure file
is still present. The figure proposal and the section drafts are model calls,
so without this a resume would replace a delivered paper with differently
worded prose about the same numbers; with a changed verified solution the
fingerprint no longer matches and the publication is written again.

Stages after `paper` (`audit`, `pdf`, `package`, `final_check`) always re-run:
they are deterministic checks over the reused artifacts, and re-running them is
what validates a reused paper.

### Where State Lives

- `<run_dir>/pipeline_state.json` — the resumable run state: which stages
  completed, the clarification answers, the counters. `RunState` reads and
  writes it, and it is the only mechanism that resumes a run.
- `problem_states` (database) — the record of the same run: mapped stage,
  status, stage history, results and verification verdict, written by
  `autopilot/ledger.py` as the run progresses. It is derived from the run
  and is never read back to resume one.

Writing to the database is best-effort: an unreachable database disables the
ledger and leaves a note in the run state rather than failing the run.

## Stage Properties

Every stage has:
- input schema
- output schema
- status (pending, running, completed, failed, skipped)
- retry count
- error messages
- timestamps (created, started, completed)
- evidence references
- artifact references

## Evidence/Claim Tracking

All claims must be traceable:
```
Paper Claim → Result → Validation → ExecutionRecord → Code Version → Mathematical Model → Equation → Assumption/Data/Fact
```

Unverifiable claims: status = UNVERIFIED, excluded from final paper.

## Fact Classification

Every statement must be classified as:
- FACT — verifiable truth from the problem
- DATA — from provided datasets
- ASSUMPTION — modeling assumption
- DERIVATION — mathematically derived
- RESULT — computed output
- EXTERNAL_EVIDENCE — from literature/sources