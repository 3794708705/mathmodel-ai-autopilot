"""Tests for Phase 3C: Python Sandbox security and execution."""

import asyncio
import pytest

from mathmodel.sandbox.backend import (
    LocalTestSandboxBackend,
    SandboxLimits,
    ExecutionStatus,
    ExecutionRecord,
)
from mathmodel.sandbox.executor import SandboxExecutor


def run_async(coro):
    """Helper to run async code in sync tests."""
    return asyncio.run(coro)


# ═══════════════════════════════════════════════════════════════
# Normal Execution
# ═══════════════════════════════════════════════════════════════

class TestNormalExecution:
    def test_simple_print(self):
        executor = SandboxExecutor()
        record = run_async(executor.execute('print("hello world")'))
        assert record.status == ExecutionStatus.SUCCESS
        assert "hello world" in record.stdout
        assert record.exit_code == 0

    def test_arithmetic(self):
        executor = SandboxExecutor()
        record = run_async(executor.execute("result = 2 + 2\nprint(result)"))
        assert record.status == ExecutionStatus.SUCCESS
        assert "4" in record.stdout

    def test_artifact_generation(self):
        executor = SandboxExecutor()
        code = """
with open("output.txt", "w") as f:
    f.write("artifact content")
print("done")
"""
        record = run_async(executor.execute(code))
        assert record.status == ExecutionStatus.SUCCESS
        assert "output.txt" in record.artifacts

    def test_stdout_stderr_capture(self):
        executor = SandboxExecutor()
        code = """
import sys
print("stdout message")
print("stderr message", file=sys.stderr)
"""
        record = run_async(executor.execute(code))
        assert "stdout message" in record.stdout
        assert "stderr message" in record.stderr


# ═══════════════════════════════════════════════════════════════
# Error Handling
# ═══════════════════════════════════════════════════════════════

class TestErrorHandling:
    def test_syntax_error(self):
        executor = SandboxExecutor()
        record = run_async(executor.execute("print('unclosed string"))
        assert record.status == ExecutionStatus.FAILED
        assert record.exit_code != 0

    def test_runtime_error(self):
        executor = SandboxExecutor()
        record = run_async(executor.execute("x = 1/0"))
        assert record.status == ExecutionStatus.FAILED
        assert record.exit_code != 0

    def test_timeout(self):
        executor = SandboxExecutor(limits=SandboxLimits(timeout_seconds=1))
        code = """
import time
time.sleep(10)
"""
        record = run_async(executor.execute(code))
        assert record.status == ExecutionStatus.TIMED_OUT
        assert record.timed_out is True


# ═══════════════════════════════════════════════════════════════
# Security Tests
# ═══════════════════════════════════════════════════════════════

class TestSecurity:
    def test_block_os_system(self):
        executor = SandboxExecutor()
        code = 'import os; os.system("echo hacked")'
        record = run_async(executor.execute(code))
        assert record.status == ExecutionStatus.SECURITY_VIOLATION

    def test_block_subprocess(self):
        executor = SandboxExecutor()
        code = 'import subprocess; subprocess.run(["echo", "hacked"])'
        record = run_async(executor.execute(code))
        assert record.status == ExecutionStatus.SECURITY_VIOLATION

    def test_block_eval(self):
        executor = SandboxExecutor()
        code = 'eval("__import__(\'os\').system(\'echo hacked\')")'
        record = run_async(executor.execute(code))
        assert record.status == ExecutionStatus.SECURITY_VIOLATION

    def test_block_exec(self):
        executor = SandboxExecutor()
        code = 'exec("print(1)")'
        record = run_async(executor.execute(code))
        assert record.status == ExecutionStatus.SECURITY_VIOLATION

    def test_block_network(self):
        executor = SandboxExecutor()
        code = 'import urllib.request; urllib.request.urlopen("http://example.com")'
        record = run_async(executor.execute(code))
        assert record.status == ExecutionStatus.SECURITY_VIOLATION

    def test_block_file_traversal(self):
        executor = SandboxExecutor()
        code = 'open("/etc/passwd").read()'
        record = run_async(executor.execute(code))
        assert record.status == ExecutionStatus.SECURITY_VIOLATION

    def test_safe_code_passes(self):
        """Normal scientific computing code should pass security checks."""
        executor = SandboxExecutor()
        code = """
import math
import statistics
data = [1, 2, 3, 4, 5]
mean = statistics.mean(data)
std = statistics.stdev(data)
print(f"mean={mean}, std={std}")
with open("result.txt", "w") as f:
    f.write(f"{mean},{std}")
"""
        record = run_async(executor.execute(code))
        assert record.status == ExecutionStatus.SUCCESS

    def test_execution_record_fields(self):
        executor = SandboxExecutor()
        record = run_async(executor.execute("print('test')"))
        assert record.run_id.startswith("RUN-")
        assert record.code_hash != ""
        assert record.started_at is not None
        assert record.finished_at is not None
        assert record.runtime_seconds > 0

    def test_backend_unsafe_flag(self):
        backend = LocalTestSandboxBackend()
        assert backend.UNSAFE_FOR_PRODUCTION is True
        assert backend.production_safe is False

    def test_executor_is_mock_flag(self):
        executor = SandboxExecutor()
        record = run_async(executor.execute("print('test')"))
        # Local backend: real execution, NOT mock
        assert record.is_mock is False
        assert record.execution_real is True
        assert record.production_safe is False
        assert record.backend_type == "local_test"

    def test_real_local_execution_not_mock(self):
        """Real local execution must not be marked as mock."""
        executor = SandboxExecutor()
        record = run_async(executor.execute("print('real')"))
        assert record.execution_real is True
        assert record.is_mock is False

    def test_local_backend_not_production_safe(self):
        """Local backend must remain production_safe=False."""
        executor = SandboxExecutor()
        record = run_async(executor.execute("print('test')"))
        assert record.production_safe is False

    def test_security_violation_not_real_execution(self):
        """Security violation should not be marked as real execution."""
        executor = SandboxExecutor()
        code = 'import os; os.system("echo")'
        record = run_async(executor.execute(code))
        assert record.status == ExecutionStatus.SECURITY_VIOLATION
        assert record.execution_real is False