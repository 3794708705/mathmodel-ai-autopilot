# MathModel AI

End-to-end AI automation system for mathematical modeling competitions.

## Overview

MathModel AI takes a competition problem (PDF, Excel, CSV, images, etc.) along with rules and deadlines, and autonomously produces a complete, verifiable, competition-ready submission package.

## Current Status

**Phase 1 — Foundation** (in progress)

- FastAPI server with REST API
- PostgreSQL + SQLAlchemy ORM
- Configuration management via pydantic-settings
- ModelProvider abstraction (OpenAI, Google, Anthropic, Mock)
- ModelRouter with task profiling and escalation
- BaseAgent with standard lifecycle
- ProblemState skeleton with full workflow state machine

## Quick Start

```bash
# Install dependencies
pip install -e ".[dev]"

# Run the server
uvicorn mathmodel.main:app --reload

# Run tests
pytest

# Run tests with coverage
pytest --cov=mathmodel -v
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/v1/health` | Health check |
| GET | `/api/v1/config` | Configuration info |
| GET | `/api/v1/providers` | List available providers |
| POST | `/api/v1/problems` | Create problem state |
| GET | `/api/v1/problems` | List problem states |
| GET | `/api/v1/problems/{id}` | Get problem state |
| PATCH | `/api/v1/problems/{id}` | Update problem state |
| DELETE | `/api/v1/problems/{id}` | Delete problem state |

## Architecture

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full architecture.

## Development

See [AGENTS.md](AGENTS.md) for development rules and conventions.

## License

MIT