# MathModel AI — Testing Strategy

## Test Layers

### Unit Tests
- Individual functions and methods
- Schema validation
- Configuration loading
- Provider adapters
- Router logic

### Integration Tests
- Database operations
- Provider API calls (with mock)
- Router + Provider integration
- Agent lifecycle

### Workflow Tests
- State transitions
- Agent pipeline
- Error handling and retry

### E2E Tests
- Problem → Model → Code → Execute → Validate
- Problem → Data → Literature → Model → Solve → Validate → RedTeam → Paper → Submission

## Mock Rules

- External API calls use MockProvider when keys are absent
- Mock must explicitly set `is_mock = true`
- Mock only tests: workflow, schema, state, error handling
- Mock cannot claim: model correctness, data analysis success, solver success, literature authenticity

## Test Conventions

- Test files: `test_<module>.py`
- Test functions: `test_<behavior>`
- Fixtures in `conftest.py`
- Database tests use SQLite in-memory
- Async tests use `pytest-asyncio`

## Common Commands

```bash
pytest                          # All tests
pytest tests/test_config.py     # Single module
pytest -v                       # Verbose
pytest --cov=mathmodel          # Coverage
pytest -x                       # Stop on first failure
```