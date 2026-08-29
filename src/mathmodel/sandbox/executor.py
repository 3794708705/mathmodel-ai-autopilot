"""MathModel AI — Sandbox executor.

Orchestrates code execution through the sandbox backend.
Validates code before execution and records results.
"""

from __future__ import annotations

import logging
from typing import Optional

from mathmodel.sandbox.backend import (
    ExecutionRecord,
    ExecutionStatus,
    LocalTestSandboxBackend,
    SandboxBackend,
    SandboxLimits,
)

logger = logging.getLogger(__name__)

# Patterns that indicate potentially dangerous code
SUSPICIOUS_PATTERNS = [
    "import os; os.system",
    "import subprocess",
    "import socket",
    "__import__('os')",
    "eval(",
    "exec(",
    "/etc/",
    "C:\\Windows",
    "requests.",
    "urllib.",
    "http.client",
]


class SandboxExecutor:
    """Executes Python code in an isolated sandbox.

    All AI-generated code must pass through this executor.
    No exec/eval in the host process.
    """

    def __init__(
        self,
        backend: Optional[SandboxBackend] = None,
        limits: Optional[SandboxLimits] = None,
    ):
        self._backend = backend or LocalTestSandboxBackend()
        self._limits = limits or SandboxLimits()

    async def execute(
        self,
        code: str,
        input_files: dict[str, str] | None = None,
        agent_run_id: Optional[str] = None,
    ) -> ExecutionRecord:
        """Execute code in the sandbox.

        Args:
            code: Python code to execute
            input_files: Optional dict of filename -> content for input files
            agent_run_id: Optional agent run ID for traceability

        Returns:
            ExecutionRecord with results, stdout, stderr, artifacts
        """
        # Security pre-check
        issues = self._security_check(code)
        if issues:
            record = ExecutionRecord(
                code_hash=ExecutionRecord.hash_code(code),
                agent_run_id=agent_run_id,
                backend=self._backend.backend_name,
                status=ExecutionStatus.SECURITY_VIOLATION,
                stderr=f"Security check failed: {'; '.join(issues)}",
            )
            return record

        # Execute through backend
        record = await self._backend.execute(
            code=code,
            limits=self._limits,
            input_files=input_files,
        )
        record.agent_run_id = agent_run_id
        record.is_mock = not self._backend.production_safe

        return record

    def _security_check(self, code: str) -> list[str]:
        """Check code for suspicious patterns. Returns list of issues."""
        issues = []
        code_lower = code.lower()

        for pattern in SUSPICIOUS_PATTERNS:
            if pattern.lower() in code_lower:
                issues.append(f"Suspicious pattern detected: {pattern}")

        return issues

    @property
    def production_safe(self) -> bool:
        return self._backend.production_safe