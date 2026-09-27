# MathModel AI — Architecture

## Overview

MathModel AI is an end-to-end AI automation system for mathematical modeling competitions. It processes competition problems, attachments, and rules to produce complete submission packages.

## System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      FastAPI Server                       │
├─────────────────────────────────────────────────────────┤
│  API Routes  │  Agents  │  Providers  │  Router          │
├─────────────────────────────────────────────────────────┤
│  SQLAlchemy ORM  │  Pydantic Schemas  │  Config          │
├─────────────────────────────────────────────────────────┤
│  PostgreSQL  │  File Store (S3/MinIO)  │  Sandbox (Docker)│
└─────────────────────────────────────────────────────────┘
```

## Component Layers

### Layer 1: API & Web
- FastAPI application
- Pydantic request/response schemas
- RESTful endpoints

### Layer 2: Agents
- BaseAgent with standard lifecycle
- Specialized agents for each workflow stage
- All agents operate on unified ProblemState

### Layer 3: Model Providers
- Abstracted LLM access via BaseModelProvider
- Provider implementations: OpenAI, Google, Anthropic, Mock
- Unified interface: generate(), structured_generate(), stream(), get_usage()

### Layer 4: Model Router
- TaskProfile analysis
- RoutingPolicy decision
- ModelRouter orchestration
- Model escalation chain

### Layer 5: Data
- SQLAlchemy ORM models
- Alembic migrations
- PostgreSQL with optional pgvector
- `autopilot/ledger.py` is the single writer for runs: it mirrors each run's
  stage progress, results and verification verdict into its `problem_states`
  row. The run directory keeps the state that resumes a run; the database
  keeps the record of it, and no run reads its own row back.

## Separation of Concerns

| LLM | Python/Solver | Database | File Store |
|-----|---------------|----------|------------|
| Reasoning | Statistics | State | Original files |
| Planning | Calculation | History | Generated code |
| Model Selection | Optimization | Evidence | Figures |
| Mathematical Modeling | Simulation | Versions | Tables |
| Code Generation | Validation | Traceability | Results |
| Review | Sensitivity | | Paper |
| Interpretation | Robustness | | PDF |
| Writing | | | Artifacts |

## Phase 1 Components

- `config.py` — Settings management via pydantic-settings
- `database.py` — SQLAlchemy engine, session, base
- `models/` — ORM models (ProblemState skeleton)
- `providers/` — ModelProvider abstraction + implementations
- `routing/` — ModelRouter + TaskProfile + RoutingPolicy
- `agents/` — BaseAgent
- `api/` — FastAPI routes
- `main.py` — Application entry point