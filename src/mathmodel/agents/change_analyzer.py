"""MathModel AI — Phase 7D: Change Impact Analyzer.

Deterministic model diff analysis. Compares before/after model
state and classifies changes. NEVER trusts LLM self-declared
classification alone.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from mathmodel.runtime import ChangeClassification


@dataclass
class ChangeImpactResult:
    declared_classification: Optional[ChangeClassification] = None
    observed_classification: ChangeClassification = ChangeClassification.FORMATTING_ONLY
    effective_classification: ChangeClassification = ChangeClassification.FORMATTING_ONLY
    change_summary: str = ""
    affected_ids: list[str] = field(default_factory=list)
    candidate_changed: bool = False
    family_changed: bool = False
    objectives_changed: bool = False
    constraints_changed: bool = False
    variables_changed: bool = False
    parameters_changed: bool = False
    units_changed: bool = False


class ChangeImpactAnalyzer:
    """Deterministic model diff analyzer.

    Compares before/after model snapshots and classifies changes.
    Effective classification = max risk(declared, observed).
    """

    def analyze(
        self,
        before: dict[str, Any],
        after: dict[str, Any],
        declared: Optional[ChangeClassification] = None,
    ) -> ChangeImpactResult:
        result = ChangeImpactResult(declared_classification=declared)

        # Candidate check
        before_cand = before.get("selected_candidate", before.get("candidate_id"))
        after_cand = after.get("selected_candidate", after.get("candidate_id"))
        if before_cand != after_cand:
            result.candidate_changed = True
            result.affected_ids.append(f"candidate:{before_cand}->{after_cand}")

        # Family check
        before_fam = before.get("model_family")
        after_fam = after.get("model_family")
        if before_fam and after_fam and before_fam != after_fam:
            result.family_changed = True
            result.affected_ids.append(f"family:{before_fam}->{after_fam}")

        # Objectives
        before_obj = self._extract_ids(before.get("objectives", []))
        after_obj = self._extract_ids(after.get("objectives", []))
        if before_obj != after_obj:
            result.objectives_changed = True
            result.affected_ids.extend(
                [f"obj:{o}" for o in (before_obj ^ after_obj)]
            )

        # Constraints
        before_con = set(before.get("constraint_ids", []))
        after_con = set(after.get("constraint_ids", []))
        if before_con != after_con:
            result.constraints_changed = True
            result.affected_ids.extend(
                [f"con:{c}" for c in (before_con ^ after_con)]
            )

        # Variables
        before_var = self._extract_ids(before.get("variables", []))
        after_var = self._extract_ids(after.get("variables", []))
        if before_var != after_var:
            result.variables_changed = True
            result.affected_ids.extend(
                [f"var:{v}" for v in (before_var ^ after_var)]
            )

        # Parameters
        before_params = self._extract_param_ids(before.get("parameters", []))
        after_params = self._extract_param_ids(after.get("parameters", []))
        if before_params != after_params:
            result.parameters_changed = True
            if before_params ^ after_params:
                result.affected_ids.extend(
                    [f"param:{p}" for p in (before_params ^ after_params)]
                )

        # Classify observed
        result.observed_classification = self._classify(result)
        result.effective_classification = self._max_risk(
            declared or ChangeClassification.FORMATTING_ONLY,
            result.observed_classification,
        )
        result.change_summary = self._summarize(result)
        return result

    def _extract_ids(self, items: list) -> set[str]:
        ids = set()
        for item in items:
            if isinstance(item, dict):
                for key in ("objective_id", "variable_id", "constraint_id", "equation_id", "id"):
                    if key in item:
                        ids.add(item[key])
            elif hasattr(item, "objective_id"):
                ids.add(item.objective_id)
            elif hasattr(item, "variable_id"):
                ids.add(item.variable_id)
            elif hasattr(item, "constraint_id"):
                ids.add(item.constraint_id)
        return ids

    def _extract_param_ids(self, items: list) -> set[str]:
        ids = set()
        for item in items:
            if isinstance(item, dict):
                ids.add(item.get("parameter_id", item.get("id", str(item))))
            elif hasattr(item, "parameter_id"):
                ids.add(item.parameter_id)
        return ids

    def _classify(self, result: ChangeImpactResult) -> ChangeClassification:
        if result.candidate_changed or result.family_changed:
            return ChangeClassification.MODEL_SWITCH
        if result.objectives_changed:
            return ChangeClassification.MAJOR_MODEL_CHANGE
        if result.constraints_changed:
            obj_count = len(result.affected_ids)
            if obj_count > 3:
                return ChangeClassification.MAJOR_MODEL_CHANGE
            return ChangeClassification.CONSTRAINT_FIX
        if result.parameters_changed:
            return ChangeClassification.PARAMETER_CORRECTION
        if result.variables_changed:
            return ChangeClassification.MAJOR_MODEL_CHANGE
        if result.units_changed:
            return ChangeClassification.PARAMETER_CORRECTION
        return ChangeClassification.FORMATTING_ONLY

    @staticmethod
    def _max_risk(a: ChangeClassification, b: ChangeClassification) -> ChangeClassification:
        order = {
            ChangeClassification.FORMATTING_ONLY: 0,
            ChangeClassification.CITATION_ONLY: 1,
            ChangeClassification.PAPER_ONLY: 2,
            ChangeClassification.CODE_FIX: 3,
            ChangeClassification.PARAMETER_CORRECTION: 4,
            ChangeClassification.CONSTRAINT_FIX: 5,
            ChangeClassification.MINOR_FIX: 6,
            ChangeClassification.MAJOR_MODEL_CHANGE: 7,
            ChangeClassification.MODEL_SWITCH: 8,
        }
        return a if order.get(a, 0) >= order.get(b, 0) else b

    @staticmethod
    def _summarize(result: ChangeImpactResult) -> str:
        parts = []
        if result.candidate_changed:
            parts.append("candidate changed")
        if result.family_changed:
            parts.append("family changed")
        if result.objectives_changed:
            parts.append("objectives changed")
        if result.constraints_changed:
            parts.append("constraints changed")
        if result.parameters_changed:
            parts.append("parameters changed")
        if result.variables_changed:
            parts.append("variables changed")
        return "; ".join(parts) if parts else "no structural changes"