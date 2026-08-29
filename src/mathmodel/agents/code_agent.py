"""MathModel AI — CodeAgent.

Generates reproducible Python code from a MathematicalModel.
For LP models, prefers the deterministic SimpleLPCompiler path.
CodeAgent is only for models/parts the compiler does not cover.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Optional

from mathmodel.domain.math_model import MathematicalModel, Variable, Parameter, Constraint, Objective, ConstraintRelation, ObjectiveSense
from mathmodel.domain.codegen import (
    CodeArtifact, CodeMapping, CodeGenerationResult, CodeObjectType, GeneratedFile,
)
from mathmodel.solver import SimpleLPCompiler

logger = logging.getLogger(__name__)


class CodeAgent:
    """Generates code from a MathematicalModel (source of truth).

    Never re-reads the raw problem to rebuild a second model.
    """

    name = "CodeAgent"
    role = "Code generation from mathematical models"

    def __init__(self, router=None):
        self._router = router

    def generate(self, model: MathematicalModel) -> CodeGenerationResult:
        """Generate code artifact and mappings from a model.

        For LP models with a supported structure, uses the deterministic
        compiler path. For other models, generates a solver script that
        references the model definition.
        """
        # Compile to check compilability
        try:
            compiled = SimpleLPCompiler.compile(model)
            compilable = True
        except Exception as e:
            logger.warning("Model not compilable via SimpleLPCompiler: %s", e)
            compiled = None
            compilable = False

        mappings: list[CodeMapping] = []
        files: list[GeneratedFile] = []

        if compilable:
            # Deterministic compiler path: generate solver.py that encodes
            # the compiled representation
            code = self._generate_solver_script(model, compiled)
            code_hash = hashlib.sha256(code.encode()).hexdigest()[:16]

            files.append(GeneratedFile(
                path="solver.py",
                purpose="Compiled LP solver script",
                hash=code_hash,
                objectives=[o.objective_id for o in model.objectives],
                constraints=[c.constraint_id for c in model.constraints],
            ))

            # Create mappings for objective and constraints
            for obj in model.objectives:
                mappings.append(CodeMapping(
                    model_id=model.model_id,
                    mathematical_object_id=obj.objective_id,
                    object_type=CodeObjectType.OBJECTIVE,
                    code_artifact_id=f"CODE-{code_hash}",
                    file="solver.py",
                    function="solve",
                    generated_representation=obj.expression,
                    code_hash=code_hash,
                ))
            for con in model.constraints:
                mappings.append(CodeMapping(
                    model_id=model.model_id,
                    mathematical_object_id=con.constraint_id,
                    object_type=CodeObjectType.CONSTRAINT,
                    code_artifact_id=f"CODE-{code_hash}",
                    file="solver.py",
                    function="solve",
                    generated_representation=con.expression,
                    code_hash=code_hash,
                ))

            artifact = CodeArtifact(
                model_id=model.model_id,
                files=files,
                entrypoint="solver.py",
                dependencies=["numpy", "scipy"],
                code_hash=code_hash,
                execution_status="not_executed",
            )
            return CodeGenerationResult(
                artifact=artifact,
                mappings=mappings,
                status="completed",
            )
        else:
            # Model cannot be compiled by the deterministic path.
            # CodeAgent must not invent a second model — block honestly.
            return CodeGenerationResult(
                status="blocked",
                blocked_reason=(
                    "Model cannot be compiled by SimpleLPCompiler and no "
                    "LLM code generation backend is configured."
                ),
            )

    def _generate_solver_script(
        self, model: MathematicalModel, compiled: dict
    ) -> str:
        """Generate a solver script that encodes the compiled LP."""
        lines = [
            '"""Auto-generated LP solver. Source of truth: MathematicalModel."""',
            "import json",
            "import numpy as np",
            "from scipy.optimize import linprog",
            "",
            f"# Compiled from model {model.model_id}",
            f"c = {compiled['c']}",
            f"A_ub = {compiled['A_ub']}",
            f"b_ub = {compiled['b_ub']}",
            f"A_eq = {compiled['A_eq']}",
            f"b_eq = {compiled['b_eq']}",
            f"bounds = {compiled['bounds']}",
            f"variable_names = {compiled['variable_names']}",
            f"maximize = {compiled['maximize']}",
            "",
            "def solve():",
            "    result = linprog(c=c, A_ub=np.array(A_ub) if A_ub else None,",
            "                    b_ub=np.array(b_ub) if b_ub else None,",
            "                    A_eq=np.array(A_eq) if A_eq else None,",
            "                    b_eq=np.array(b_eq) if b_eq else None,",
            "                    bounds=bounds, method='highs')",
            "    obj = float(result.fun) if result.fun is not None else None",
            "    if maximize and obj is not None:",
            "        obj = -obj",
            "    values = {}",
            "    if result.success and result.x is not None:",
            "        values = dict(zip(variable_names, [float(v) for v in result.x]))",
            "    print(json.dumps({'status': int(result.status),",
            "                     'objective': obj, 'values': values}))",
            "",
            "if __name__ == '__main__':",
            "    solve()",
        ]
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
# Model-Code Drift Detection
# ═══════════════════════════════════════════════════════════════

def check_model_code_drift(
    model: MathematicalModel,
    mappings: list[CodeMapping],
) -> list[str]:
    """Check that code mappings cover all core model objects.

    Returns list of drift issues (empty = no drift).
    """
    issues = []

    obj_ids = {o.objective_id for o in model.objectives}
    con_ids = {c.constraint_id for c in model.constraints}

    mapped_obj = {m.mathematical_object_id for m in mappings
                  if m.object_type == CodeObjectType.OBJECTIVE}
    mapped_con = {m.mathematical_object_id for m in mappings
                  if m.object_type == CodeObjectType.CONSTRAINT}

    missing_obj = obj_ids - mapped_obj
    missing_con = con_ids - mapped_con

    if missing_obj:
        issues.append(f"Objectives without code mapping: {sorted(missing_obj)}")
    if missing_con:
        issues.append(f"Constraints without code mapping: {sorted(missing_con)}")

    # Extra mappings for objects not in the model
    extra_obj = mapped_obj - obj_ids
    extra_con = mapped_con - con_ids
    if extra_obj:
        issues.append(f"Mappings for unknown objectives: {sorted(extra_obj)}")
    if extra_con:
        issues.append(f"Mappings for unknown constraints: {sorted(extra_con)}")

    return issues