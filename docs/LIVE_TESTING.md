# MathModel AI — Live Testing Guide

## Principle

Normal test suite (`python -m pytest tests/ -q`) must NEVER require
API credentials or network. Live tests are opt-in via markers.

## Markers

- `live` — requires real LLM credentials
- `external` — requires real external service (Crossref etc.)

## Commands

```bash
# Full unit suite (offline, fast):
python -m pytest tests/ -q

# Live LLM tests:
pytest -m live -v

# Live external search tests:
pytest -m external -v
```

## Credentials

- DEEPSEEK_API_KEY — for DeepSeekProvider live tests
- OPENAI_API_KEY — for OpenAIProvider live tests

## Skip Policy

When credentials are absent, live tests SKIP and the run reports
REALITY_NOT_VERIFIED. Skipped live tests never make ExternalRealityGate PASS.

## Secret Safety

- Never write API keys into repo, fixtures, logs, AgentRun, or exception dumps
- After live runs, audit logs for secrets: no key, token, or full secret may appear
