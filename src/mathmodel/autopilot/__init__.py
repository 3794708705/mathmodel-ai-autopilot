"""MathModel AI — CUMCM Autopilot.

One continuous, resumable workflow from uploaded competition files to a
verified paper PDF and support package.
"""

from mathmodel.autopilot.state import (
    ClarificationQuestion,
    RunState,
    RunStatus,
    StageRecord,
    StageStatus,
)
from mathmodel.autopilot.intake import (
    AttachmentInfo,
    ProblemContext,
    ProblemIntake,
    SheetSchema,
    build_clarification_questions,
)
from mathmodel.autopilot.codegen import (
    ExecutionOutcome,
    GeneratedProgram,
    SandboxProgramRunner,
    SolverCodeGenerator,
)
from mathmodel.autopilot.verify import (
    CheckStatus,
    DeterministicVerifier,
    IndependentVerifier,
    VerificationCheck,
    VerificationReport,
)
from mathmodel.autopilot.deliver import (
    FigureBuilder,
    PaperContextPackage,
    PaperWriter,
    SupportPackageBuilder,
    TableBuilder,
    build_cumcm_latex,
    build_pdf,
)
from mathmodel.autopilot.pipeline import AutopilotResult, CUMCMAutopilot

__all__ = [
    "AutopilotResult",
    "AttachmentInfo",
    "CheckStatus",
    "ClarificationQuestion",
    "CUMCMAutopilot",
    "DeterministicVerifier",
    "ExecutionOutcome",
    "FigureBuilder",
    "GeneratedProgram",
    "IndependentVerifier",
    "PaperContextPackage",
    "PaperWriter",
    "ProblemContext",
    "ProblemIntake",
    "RunState",
    "RunStatus",
    "SandboxProgramRunner",
    "SheetSchema",
    "SolverCodeGenerator",
    "StageRecord",
    "StageStatus",
    "SupportPackageBuilder",
    "TableBuilder",
    "VerificationCheck",
    "VerificationReport",
    "build_clarification_questions",
    "build_cumcm_latex",
    "build_pdf",
]
