# MathModel AI — AGENTS.md

## Project Goal

End-to-end AI automation system for mathematical modeling competitions.
Input: competition problem + attachments (PDF, Excel, CSV, images, etc.) + rules + deadline.
Output: complete submission package (paper PDF, code, results, figures, tables).

## Highest Principles

```
Correctness > Traceability > Validation > Competition Value > Reliability > Maintainability > Performance > Complexity
```

## Architecture Documents

- `docs/ARCHITECTURE.md` — System architecture and component design
- `docs/PRODUCT.md` — Product requirements and design decisions
- `docs/MODEL_ROUTING.md` — Model routing and escalation design
- `docs/STATE_MODEL.md` — ProblemState and state machine specification
- `docs/AGENT_CONTRACTS.md` — Agent interfaces and contracts
- `docs/TESTING.md` — Testing strategy and patterns
- `docs/design/` — Implementation plans and design decisions

## Current Development Phase

**Phase 1 — Foundation** (in progress)

- FastAPI skeleton
- PostgreSQL + SQLAlchemy
- Configuration management
- ModelProvider abstraction
- ModelRouter
- BaseAgent
- ProblemState skeleton
- Testing infrastructure

## Common Commands

```bash
# Install
pip install -e ".[dev]"

# Run server
uvicorn mathmodel.main:app --reload

# Run tests
pytest

# Run tests with coverage
pytest --cov=mathmodel

# Database migrations
alembic upgrade head
```

## Prohibited Actions

- No untested mass code generation
- No pass/placeholder in core methods
- No "TODO" in critical paths
- No hardcoded results for test passing
- No mock-as-real deception
- No LLM impersonating Python/Solver
- No fabricated literature
- No ignoring test failures
- No unrelated large-scale refactoring

## Agent Development Rules

1. Inspect repository before any change
2. Read AGENTS.md and relevant docs
3. Check git status
4. Understand existing implementation
5. Don't overwrite valid code
6. Don't refactor unrelated modules
7. Determine minimal change scope first
8. Implement, test, debug, document