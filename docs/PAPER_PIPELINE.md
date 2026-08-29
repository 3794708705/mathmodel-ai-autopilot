# MathModel AI — Evidence to Paper Pipeline (Phase 6)

## Principle

**NO EVIDENCE → NO CLAIM.** The paper is assembled from verified evidence,
never written from raw LLM context.

## Pipeline

```
Verified Evidence
↓
Claim system (ClaimGate)
↓
Literature (CitationVerifier)
↓
Figures / Tables (registries)
↓
PaperIR (structured)
↓
LaTeX renderer
↓
PDF (if compiler available)
↓
Final Jury (cannot override deterministic failures)
↓
Submission Check
```

## Components

| Module | Purpose |
|--------|---------|
| `evidence/` | EvidenceStore, EvidenceRef, Claim, ClaimGate |
| `literature/` | LiteratureRecord, LiteratureStore, Citation, CitationVerifier |
| `documents/` | DocumentRegistry, FigureRecord/Registry, TableRecord/Registry |
| `paper/` | PaperIR, PaperSection, ContentBlock, PaperAgent |
| `paper/renderer.py` | LaTeXRenderer, CompilationRecord |
| `submission/` | CompetitionProfile, SubmissionCheckAgent, UnsupportedClaimChecker, NumericalConsistencyChecker, FinalJuryAgent |
| `domain/codegen.py` | CodeArtifact, CodeMapping, GeneratedFile |
| `agents/code_agent.py` | CodeAgent + model-code drift detection |
| `math_ops/units.py` | Compound unit dimensional analysis |

## Truth Boundaries

- **Numerical consistency:** source 700.0 matches paper "700"/"700.00", fails "701"
- **Claims:** high/critical importance claims need evidence; UNSUPPORTED blocked
- **Citations:** fixture records never enter bibliography; IDENTITY_MISMATCH flagged
- **Figures:** artifact file must exist, be non-empty, have data/execution source
- **Tables:** numeric cells need source_id; row/header counts must match
- **Jury:** cannot override SubmissionCheck BLOCKED

## Environment Status

- LaTeX: renderer implemented; compilation records honest failure without a compiler
- PDF: only claimed when real compile succeeded (PDF_NOT_VERIFIED otherwise)
- SearchProvider: contract only — no real search available
- Real LLM: contracts verified; reasoning quality NOT VERIFIED
