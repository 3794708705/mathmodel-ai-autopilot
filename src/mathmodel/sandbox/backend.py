"""MathModel AI — Sandbox backend abstraction.

Defines the interface for sandbox execution backends.
LocalTestSandboxBackend is for controlled testing only.
"""

from __future__ import annotations

import hashlib
import logging
import os
import subprocess
import tempfile
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

logger = logging.getLogger(__name__)


class ExecutionStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    TIMED_OUT = "timed_out"
    MEMORY_EXCEEDED = "memory_exceeded"
    SECURITY_VIOLATION = "security_violation"


@dataclass
class SandboxLimits:
    """Resource limits for sandbox execution."""
    timeout_seconds: int = 30
    max_memory_mb: int = 512
    max_disk_mb: int = 100
    max_processes: int = 1
    network_enabled: bool = False
    read_only_filesystem: bool = True
    allow_subprocess: bool = False


@dataclass
class ExecutionRecord:
    """Record of a sandbox execution."""

    run_id: str = field(default_factory=lambda: f"RUN-{uuid4().hex[:8]}")
    project_id: Optional[str] = None
    agent_run_id: Optional[str] = None
    code_hash: str = ""
    code_version: str = ""
    backend: str = ""
    backend_type: str = ""  # "local_test", "docker", "mock"
    environment: str = ""
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    runtime_seconds: float = 0.0
    exit_code: int = -1
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False
    memory_exceeded: bool = False
    artifacts: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    is_mock: bool = False
    execution_real: bool = False  # True if real Python/subprocess executed
    production_safe: bool = False  # True only for Docker/production backends
    status: ExecutionStatus = ExecutionStatus.PENDING

    @staticmethod
    def hash_code(code: str) -> str:
        return hashlib.sha256(code.encode()).hexdigest()[:16]


class SandboxBackend(ABC):
    """Abstract interface for sandbox execution backends."""

    @abstractmethod
    async def execute(
        self,
        code: str,
        limits: SandboxLimits,
        input_files: dict[str, str] | None = None,
    ) -> ExecutionRecord:
        """Execute code in the sandbox and return an execution record."""
        ...

    @property
    @abstractmethod
    def backend_name(self) -> str:
        ...

    @property
    @abstractmethod
    def production_safe(self) -> bool:
        ...


class LocalTestSandboxBackend(SandboxBackend):
    """Local process sandbox for controlled testing.

    UNSAFE FOR PRODUCTION: executes code in a subprocess on the host.
    Only for testing with trusted code. Does NOT provide true isolation.
    """

    UNSAFE_FOR_PRODUCTION = True

    @property
    def backend_name(self) -> str:
        return "local_test"

    @property
    def production_safe(self) -> bool:
        return False

    async def execute(
        self,
        code: str,
        limits: SandboxLimits,
        input_files: dict[str, str] | None = None,
    ) -> ExecutionRecord:
        """Execute code in a subprocess with basic isolation."""
        record = ExecutionRecord(
            backend=self.backend_name,
            backend_type="local_test",
            code_hash=ExecutionRecord.hash_code(code),
            started_at=datetime.now(timezone.utc),
            status=ExecutionStatus.RUNNING,
            execution_real=True,  # Real subprocess execution
            production_safe=False,  # Local = not production safe
            is_mock=False,  # Not a mock — real execution
        )

        # Create isolated working directory
        work_dir = tempfile.mkdtemp(prefix="sandbox_")

        try:
            # Write code to file
            code_path = Path(work_dir) / "code.py"
            code_path.write_text(code, encoding="utf-8")

            # Write input files if provided
            if input_files:
                for name, content in input_files.items():
                    dest = Path(work_dir) / name
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_text(content, encoding="utf-8")

            # Build minimal environment — only allowlisted vars
            allowed_env_prefixes = (
                "PATH", "PATHEXT", "SYSTEMROOT", "TMP", "TEMP",
                "USERPROFILE", "HOMEDRIVE", "HOMEPATH", "HOME",
                "LANG", "LC_", "PYTHON", "CONDA",
            )
            # Start with minimal env
            sandbox_env = {
                "HOME": work_dir,
                "TMP": work_dir,
                "TEMP": work_dir,
                "PYTHONPATH": "",
                "PYTHONUNBUFFERED": "1",
            }
            # Add allowed host env vars
            for key, value in os.environ.items():
                if any(key.startswith(prefix) for prefix in allowed_env_prefixes):
                    sandbox_env[key] = value

            start = time.time()
            try:
                result = subprocess.run(
                    ["python", str(code_path)],
                    capture_output=True,
                    text=True,
                    timeout=limits.timeout_seconds,
                    cwd=work_dir,
                    env=sandbox_env,
                )
                elapsed = time.time() - start

                record.exit_code = result.returncode
                record.stdout = result.stdout[:100_000]  # Truncate
                record.stderr = result.stderr[:100_000]
                record.runtime_seconds = round(elapsed, 3)
                record.timed_out = False
                record.status = (
                    ExecutionStatus.SUCCESS
                    if result.returncode == 0
                    else ExecutionStatus.FAILED
                )

            except subprocess.TimeoutExpired:
                elapsed = time.time() - start
                record.runtime_seconds = round(elapsed, 3)
                record.timed_out = True
                record.status = ExecutionStatus.TIMED_OUT
                record.stderr = "Execution timed out"

            # Collect artifacts
            artifacts = []
            for p in Path(work_dir).rglob("*"):
                if p.is_file() and p.name != "code.py":
                    artifacts.append(str(p.relative_to(work_dir)))
            record.artifacts = artifacts

        except Exception as e:
            record.status = ExecutionStatus.FAILED
            record.stderr = str(e)

        finally:
            record.finished_at = datetime.now(timezone.utc)
            # Cleanup
            try:
                import shutil
                shutil.rmtree(work_dir, ignore_errors=True)
            except Exception:
                pass

        return record