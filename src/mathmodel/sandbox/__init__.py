"""MathModel AI — Sandbox executor.

Executes AI-generated Python code in an isolated environment.
All generated code must pass through SandboxExecutor — never exec/eval
directly in the host process.
"""

from mathmodel.sandbox.executor import SandboxExecutor, ExecutionRecord, ExecutionStatus
from mathmodel.sandbox.backend import (
    SandboxBackend,
    LocalTestSandboxBackend,
    SandboxLimits,
)

__all__ = [
    "SandboxExecutor",
    "ExecutionRecord",
    "ExecutionStatus",
    "SandboxBackend",
    "LocalTestSandboxBackend",
    "SandboxLimits",
]