# MathModel AI — State Model

## ProblemState

The unified state object that all agents operate on. No agent maintains private facts.

### Core Fields

```
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