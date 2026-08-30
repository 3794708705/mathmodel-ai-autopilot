# MathModel AI — Sandbox Threat Model

## Principle

AI-generated code NEVER executes in the host process.
All generated code must pass through `SandboxExecutor` → `DockerSandboxBackend`.

## Attack Surface

| # | Threat | Mitigation | Residual Risk | Tested |
|---|--------|-----------|---------------|--------|
| 1 | Host filesystem access | Container with isolated mounts; only /workspace mounted | Kernel vuln | ✓ |
| 2 | Repository source theft | Host repo NOT mounted; only code.py + authorized inputs | None | ✓ |
| 3 | .env / API key theft | Env allowlist; secrets not passed | Kernel vuln | ✓ |
| 4 | Env var leakage | Minimal allowlist (PYTHONUNBUFFERED, HOME, etc.) | None | ✓ |
| 5 | Network exfiltration | --network=none | Docker networking bug | ✓ |
| 6 | Docker socket access | NOT mounted; docker_socket_absent=true | None | ✓ |
| 7 | Container escape | non-root, --cap-drop=ALL, no-new-privileges | Kernel vuln (CVE) | ✓ |
| 8 | Privilege escalation | --security-opt no-new-privileges, non-root | Kernel vuln | ✓ |
| 9 | Fork bomb | --pids-limit | None | ✓ |
| 10 | CPU exhaustion | --cpus limit | Host CPU still shared | ✓ |
| 11 | Memory exhaustion | --memory limit, OOM killed | None | ✓ |
| 12 | Disk exhaustion | tmpfs size limit (optional) | Not enforced (debt) | ⚠ |
| 13 | Infinite loop | Timeout enforced | None | ✓ |
| 14 | stdout/stderr flood | Output cap 100KB | None | ✓ |
| 15 | Symlink artifact escape | Only regular files collected; path traversal blocked | None | ✓ |
| 16 | Path traversal | ../ blocked in artifact path | None | ✓ |
| 17 | Device access | --cap-drop=ALL, no devices mounted | Kernel vuln | ✓ |
| 18 | /proc access | Container isolation; non-root | None | ✓ |
| 19 | pip install | Network disabled; no runtime pip | None | ✓ |
| 20 | Child process spawning | --pids-limit controls | None | ✓ |
| 21 | Large artifact | max_artifact_count/size limits | None | ✓ |
| 22 | Cross-run contamination | Ephemeral container; fresh workdir per run | None | ✓ |
| 23 | Container persistence | --rm flag; force remove on timeout | Docker daemon crash | ✓ |
| 24 | Dangerous config | SafetyEvaluator hard checks | None | ✓ |
| 25 | Host command invocation | No docker socket, no host mounts | None | ✓ |

## Production Safety Gate

Hard requirements (all must PASS):
- network_isolated=true
- secret_isolated=true
- filesystem_isolated=true
- non_root=true
- no_new_privileges=true
- docker_socket_absent=true
- pid_limit=true
- memory_limit=true
- timeout=true
- artifact_containment=true

Any single failure → production_safe=false.