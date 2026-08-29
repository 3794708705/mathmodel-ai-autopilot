# MathModel AI — Agent Contracts

## BaseAgent Interface

All agents inherit from BaseAgent:

```python
class BaseAgent:
    name: str
    role: str
    input_schema: Type[BaseModel]
    output_schema: Type[BaseModel]
    capabilities: list[str]

    async def run(self, state: ProblemState) -> AgentResult
    def validate_output(self, output: BaseModel) -> ValidationResult
    async def retry(self, state: ProblemState, error: AgentError) -> AgentResult
```

## Agent Catalog

### Phase 1 (Foundation)
- BaseAgent only

### Phase 2 (Reasoning Core)
- ProblemAgent — problem understanding, decomposition
- ModelExplorer — candidate model generation
- ModelJury — model evaluation and selection

### Phase 3 (Data + Files)
- DataAgent — data profiling and analysis

### Phase 4 (Mathematical Core)
- MathModeler — mathematical model formalization
- CodeAgent — code generation from models
- SolverAgent — solver execution

### Phase 5 (Verification)
- ValidationAgent — result validation
- SensitivityAgent — sensitivity analysis
- RobustnessAgent — robustness analysis
- RedTeamAgent — adversarial review
- ModelRepairAgent — model repair

### Phase 6 (Paper Pipeline)
- PaperAgent — paper generation
- CitationAgent — citation management
- CitationVerifier — citation verification
- VisualizationAgent — figure/table generation

### Phase 7 (Competition Mode)
- FinalJuryAgent — final review
- SubmissionCheckAgent — submission validation

## Agent Communication

- Agents communicate only through ProblemState
- No agent-to-agent direct calls
- No private agent state
- All outputs are schema-validated
- All runs are recorded in database