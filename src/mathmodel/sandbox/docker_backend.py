"""MathModel AI — Phase 7C: Docker Sandbox Backend.

Production-safe isolated execution via Docker containers.
Non-root, network-disabled, ephemeral, resource-limited.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import shutil
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from mathmodel.sandbox.backend import (
    SandboxBackend, SandboxLimits, ExecutionRecord, ExecutionStatus,
)

logger = logging.getLogger(__name__)

# ── Docker Helpers ────────────────────────────────────────────


def _docker_available() -> bool:
    """Check if Docker daemon is reachable."""
    try:
        result = subprocess.run(
            ["docker", "info", "--format", "{{.ServerVersion}}"],
            capture_output=True, text=True, timeout=10,
        )
        return result.returncode == 0
    except Exception:
        return False


def _docker_image_present(image: str) -> bool:
    """Check if the required Docker image exists locally."""
    try:
        result = subprocess.run(
            ["docker", "image", "inspect", image],
            capture_output=True, text=True, timeout=10,
        )
        return result.returncode == 0
    except Exception:
        return False


def _safe_container_name(run_id: str) -> str:
    """Generate a safe container name from a run_id."""
    safe = re.sub(r"[^a-zA-Z0-9_.-]", "_", run_id)[:128]
    return f"sandbox_{safe}"


# ═══════════════════════════════════════════════════════════════
# Docker Sandbox Backend
# ═══════════════════════════════════════════════════════════════

class DockerSandboxBackend(SandboxBackend):
    """Production-safe Docker sandbox for AI-generated code execution.

    Creates ephemeral, non-root, network-isolated containers with
    resource limits. No Docker socket, no host filesystem, no secrets.
    """

    DOCKER_IMAGE = "python:3.12-slim"

    def __init__(
        self,
        image: Optional[str] = None,
        network_enabled: bool = False,
    ):
        self._image = image or self.DOCKER_IMAGE
        self._network_enabled = network_enabled
        self._available = _docker_available()
        self._image_present = _docker_image_present(self._image) if self._available else False

    @property
    def backend_name(self) -> str:
        return "docker"

    @property
    def backend_type(self) -> str:
        return "docker"

    @property
    def production_safe(self) -> bool:
        """Docker backend is production-safe ONLY when all safety checks pass.
        This is evaluated by SandboxSafetyEvaluator, not hardcoded.
        """
        return False  # Evaluated externally — never self-declare

    @property
    def available(self) -> bool:
        return self._available and self._image_present

    async def execute(
        self,
        code: str,
        limits: SandboxLimits,
        input_files: Optional[dict[str, str]] = None,
    ) -> ExecutionRecord:
        """Execute code in an ephemeral Docker container."""
        record = ExecutionRecord(
            backend=self.backend_name,
            backend_type="docker",
            code_hash=ExecutionRecord.hash_code(code),
            started_at=datetime.now(timezone.utc),
            status=ExecutionStatus.RUNNING,
            execution_real=True,
            production_safe=False,  # Evaluated later
            is_mock=False,
            network_isolated=not self._network_enabled,
            filesystem_isolated=True,
            secret_isolated=True,
            non_root=True,
            no_new_privileges=True,
            docker_socket_absent=True,
        )

        if not self._available:
            record.status = ExecutionStatus.BACKEND_UNAVAILABLE
            record.stderr = "Docker daemon not available"
            return record
        if not self._image_present:
            record.status = ExecutionStatus.BACKEND_UNAVAILABLE
            record.stderr = f"Docker image '{self._image}' not present"
            return record

        work_dir = tempfile.mkdtemp(prefix="sandbox_docker_")
        container_name = _safe_container_name(record.run_id)
        container_id = ""

        try:
            # ── Prepare workspace ─────────────────────────────
            code_dir = Path(work_dir) / "code"
            input_dir = Path(work_dir) / "input"
            output_dir = Path(work_dir) / "output"
            code_dir.mkdir()
            input_dir.mkdir()
            output_dir.mkdir()

            # Write generated code
            code_path = code_dir / "code.py"
            code_path.write_text(code, encoding="utf-8")
            code_hash = ExecutionRecord.hash_code(code)
            record.code_hash = code_hash

            # Write input files
            input_hashes = {}
            if input_files:
                for name, content in input_files.items():
                    dest = input_dir / name
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_text(content, encoding="utf-8")
                    input_hashes[name] = hashlib.sha256(
                        content.encode()
                    ).hexdigest()[:16]
            record.metrics["input_hashes"] = input_hashes

            # ── Build Docker command ──────────────────────────
            docker_cmd = [
                "docker", "run",
                "--rm",
                "--name", container_name,
                # Non-root
                "--user", "1000:1000",
                # No new privileges
                "--security-opt", "no-new-privileges=true",
                # Drop all capabilities
                "--cap-drop=ALL",
                # Resource limits
                f"--memory={limits.max_memory_mb}m",
                "--memory-swap", f"{limits.max_memory_mb}m",
                f"--pids-limit={limits.max_processes}",
                # Working directory
                "-w", "/workspace",
                # Mounts: code read-only, input read-only, output writable
                "-v", f"{code_dir}:/workspace/code:ro",
                "-v", f"{input_dir}:/workspace/input:ro",
                "-v", f"{output_dir}:/workspace/output",
                # Environment: minimal allowlist
                "-e", "PYTHONUNBUFFERED=1",
                "-e", "PYTHONDONTWRITEBYTECODE=1",
                "-e", "HOME=/workspace",
                "-e", "TMPDIR=/tmp",
                "-e", "OPENBLAS_NUM_THREADS=1",
                "-e", "MKL_NUM_THREADS=1",
                "-e", "OMP_NUM_THREADS=1",
                # Network: none by default
            ]

            if not self._network_enabled:
                docker_cmd.append("--network=none")

            docker_cmd.append(self._image)
            docker_cmd.extend([
                "python", "/workspace/code/code.py",
            ])

            # ── Execute ───────────────────────────────────────
            start = time.time()
            try:
                result = subprocess.run(
                    docker_cmd,
                    capture_output=True,
                    text=True,
                    timeout=limits.timeout_seconds + 10,  # Docker overhead
                )
                elapsed = time.time() - start

                record.exit_code = result.returncode
                record.stdout = result.stdout[:100_000]
                record.stderr = result.stderr[:100_000]
                record.runtime_seconds = round(elapsed, 3)
                record.timed_out = False

                if result.returncode == 0:
                    record.status = ExecutionStatus.SUCCESS
                elif result.returncode == 137:
                    record.status = ExecutionStatus.OOM_KILLED
                    record.oom_killed = True
                else:
                    record.status = ExecutionStatus.FAILED

            except subprocess.TimeoutExpired:
                elapsed = time.time() - start
                record.runtime_seconds = round(elapsed, 3)
                record.timed_out = True
                record.status = ExecutionStatus.TIMED_OUT
                record.stderr = "Execution timed out"
                # Force cleanup
                self._force_remove(container_name)

            # ── Collect artifacts ─────────────────────────────
            artifacts = []
            output_hashes = {}
            for p in Path(output_dir).rglob("*"):
                if p.is_file() and not p.is_symlink():
                    rel = p.relative_to(output_dir)
                    if ".." in str(rel) or str(rel).startswith("/"):
                        continue
                    if len(artifacts) >= (limits.max_artifact_count or 100):
                        break
                    artifacts.append(str(rel))
                    output_hashes[str(rel)] = hashlib.sha256(
                        p.read_bytes()
                    ).hexdigest()[:16]
            record.artifacts = artifacts
            record.metrics["output_hashes"] = output_hashes

            # ── Verify input integrity ────────────────────────
            if input_files:
                for name, original_hash in input_hashes.items():
                    dest = input_dir / name
                    if dest.exists():
                        current_hash = hashlib.sha256(
                            dest.read_text(encoding="utf-8").encode()
                        ).hexdigest()[:16]
                        if current_hash != original_hash:
                            record.metrics.setdefault("input_modified", []).append(name)

        except Exception as e:
            record.status = ExecutionStatus.SETUP_ERROR
            record.stderr = str(e)

        finally:
            record.finished_at = datetime.now(timezone.utc)
            # Cleanup workspace
            try:
                shutil.rmtree(work_dir, ignore_errors=True)
            except Exception:
                pass
            # Ensure container removed
            self._force_remove(container_name)

        return record

    def _force_remove(self, container_name: str) -> None:
        """Force-remove a container if it still exists."""
        try:
            subprocess.run(
                ["docker", "rm", "-f", container_name],
                capture_output=True, timeout=10,
            )
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════
# Sandbox Safety Evaluator
# ═══════════════════════════════════════════════════════════════

class SandboxSafetyEvaluator:
    """Evaluates whether a sandbox backend is production-safe.

    Hard checks: all must PASS for production_safe=true.
    production_safe is NEVER self-declared by the backend.
    """

    HARD_CHECKS = [
        "network_isolated",
        "secret_isolated",
        "filesystem_isolated",
        "non_root",
        "no_new_privileges",
        "docker_socket_absent",
        "pid_limit",
        "memory_limit",
        "timeout",
        "artifact_containment",
    ]

    def __init__(self, record: ExecutionRecord):
        self._record = record

    def evaluate(self) -> tuple[bool, list[str], list[str]]:
        """Return (production_safe, failures, warnings)."""
        failures = []
        warnings = []

        checks = {
            "network_isolated": getattr(self._record, "network_isolated", False),
            "secret_isolated": getattr(self._record, "secret_isolated", False),
            "filesystem_isolated": getattr(self._record, "filesystem_isolated", False),
            "non_root": getattr(self._record, "non_root", False),
            "no_new_privileges": getattr(self._record, "no_new_privileges", False),
            "docker_socket_absent": getattr(self._record, "docker_socket_absent", False),
            "pid_limit": getattr(self._record, "pid_limit", False),
            "memory_limit": getattr(self._record, "memory_limit", False),
            "timeout": getattr(self._record, "timeout", False),
            "artifact_containment": getattr(self._record, "artifact_containment", False),
        }

        for check in self.HARD_CHECKS:
            if not checks.get(check, False):
                failures.append(f"HARD_CHECK_FAILED: {check}")

        if self._record.backend_type == "local_test":
            failures.append("HARD_CHECK_FAILED: local_test_backend_not_production_safe")

        if self._record.is_mock:
            failures.append("HARD_CHECK_FAILED: mock_execution")

        production_safe = len(failures) == 0
        return production_safe, failures, warnings