"""MathModel AI — Data reference validator.

Validates that DataAnalysis and DataCleaningPlan references
are valid against actual Dataset/DataProfile state.
"""

from __future__ import annotations

from typing import Optional

from mathmodel.domain.data import (
    DataAnalysis,
    DataCleaningPlan,
    DataProfile,
    Dataset,
    DataTable,
)


class DataReferenceValidator:
    """Validates data references against actual state."""

    def __init__(
        self,
        datasets: Optional[list[Dataset]] = None,
        tables: Optional[list[DataTable]] = None,
        profiles: Optional[list[DataProfile]] = None,
    ):
        self._dataset_ids = {d.dataset_id for d in (datasets or [])}
        self._table_ids = {t.table_id for t in (tables or [])}
        self._column_ids: set[str] = set()
        for p in (profiles or []):
            for c in p.columns:
                self._column_ids.add(c.column_id)

    def validate_analysis(self, analysis: DataAnalysis) -> list[str]:
        """Validate references in a DataAnalysis. Returns list of issues."""
        issues = []

        # Check cleaning plan references
        if analysis.cleaning_plan:
            issues.extend(self._validate_cleaning_plan(analysis.cleaning_plan))

        return issues

    def validate_cleaning_plan(self, plan: DataCleaningPlan) -> list[str]:
        """Validate references in a cleaning plan."""
        issues = []

        if plan.dataset_id and plan.dataset_id not in self._dataset_ids:
            issues.append(
                f"Cleaning plan references unknown dataset_id: {plan.dataset_id}"
            )

        for op in plan.operations:
            # target is column_id or table_id
            target = op.target
            if target.startswith("COL-") and target not in self._column_ids:
                issues.append(
                    f"Operation {op.operation_id} references unknown column: {target}"
                )
            elif target.startswith("TBL-") and target not in self._table_ids:
                issues.append(
                    f"Operation {op.operation_id} references unknown table: {target}"
                )

        return issues

    def validate_profile(self, profile: DataProfile) -> list[str]:
        """Validate references in a DataProfile."""
        issues = []

        if profile.dataset_id and profile.dataset_id not in self._dataset_ids:
            issues.append(
                f"Profile references unknown dataset_id: {profile.dataset_id}"
            )

        if profile.table_id and profile.table_id not in self._table_ids:
            issues.append(
                f"Profile references unknown table_id: {profile.table_id}"
            )

        return issues