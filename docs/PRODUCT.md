# MathModel AI — Product Requirements

## Core Capability

Given a mathematical modeling competition problem with attachments, the system autonomously produces a complete, verifiable, competition-ready submission package.

## Input Types

- Competition problem statements (PDF, TXT, Word)
- Data attachments (Excel, CSV, images)
- Competition rules and constraints
- Deadline

## Output

```
submission/
├── paper.pdf
├── paper.tex
├── references.bib
├── figures/
├── tables/
├── code/
├── results/
├── validation/
├── data/
└── README.md
```

## Workflow Stages

```
INGEST → UNDERSTAND → DATA → LITERATURE → EXPLORE → SELECT →
MODEL → SOLVE → VALIDATE → SENSITIVITY → ROBUSTNESS → RED_TEAM →
PAPER → FINAL_JURY → SUBMISSION → FINAL
```

## Design Principles

1. Correctness over complexity
2. Traceability: every number must be traceable to its source
3. Validation: no unverified claims in final output
4. LLM reasons, Python computes — never the reverse
5. Simple models first, complexity only when justified
6. Everything must be verifiable and reproducible