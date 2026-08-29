"""MathModel AI — Data profiler.

Computes statistics using Python (numpy/pandas where available).
Never uses LLM for numerical computation.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

from mathmodel.domain.data import (
    ColumnProfile,
    DataProfile,
    DataTable,
    SemanticType,
)

logger = logging.getLogger(__name__)


class DataProfiler:
    """Profiles data tables using real Python computation.

    All numerical statistics are computed by Python, not LLM.
    """

    @staticmethod
    def profile_table(
        table: DataTable,
        rows: list[list[Any]],
        headers: list[str],
    ) -> DataProfile:
        """Profile a data table from rows and headers."""
        start = time.time()

        if not rows:
            return DataProfile(
                dataset_id="",
                table_id=table.table_id,
                row_count=0,
                column_count=len(headers),
            )

        columns = []
        all_col_values = []  # Store column values for later use
        for idx, header in enumerate(headers):
            col_values = [
                row[idx] if idx < len(row) else None
                for row in rows
            ]
            all_col_values.append(col_values)
            col = DataProfiler._profile_column(header, col_values)
            columns.append(col)

        total_missing = sum(c.missing_count for c in columns)
        duplicate_rows = DataProfiler._count_duplicates(rows)

        # Outlier candidates (IQR method)
        outliers = []
        for col, col_values in zip(columns, all_col_values):
            if col.dtype in ("int64", "float64") and col.quantiles:
                q1 = col.quantiles.get("q25", 0)
                q3 = col.quantiles.get("q75", 0)
                iqr = q3 - q1
                if iqr > 0:
                    lower = q1 - 1.5 * iqr
                    upper = q3 + 1.5 * iqr
                    outlier_count = sum(
                        1 for v in col_values
                        if v is not None and isinstance(v, (int, float))
                        and (v < lower or v > upper)
                    )
                    if outlier_count > 0:
                        outliers.append({
                            "column": col.name,
                            "method": "IQR",
                            "count": outlier_count,
                            "lower_bound": lower,
                            "upper_bound": upper,
                        })

        # Correlation candidates (numeric columns only)
        correlations = []
        numeric_cols = [
            (c, DataProfiler._to_numeric(all_col_values[i]))
            for i, c in enumerate(columns)
            if c.dtype in ("int64", "float64")
        ]
        for i in range(len(numeric_cols)):
            for j in range(i + 1, len(numeric_cols)):
                col_i, vals_i = numeric_cols[i]
                col_j, vals_j = numeric_cols[j]
                corr = DataProfiler._pearson_correlation(vals_i, vals_j)
                if corr is not None and abs(corr) > 0.5:
                    correlations.append({
                        "col_a": col_i.name,
                        "col_b": col_j.name,
                        "correlation": round(corr, 4),
                    })

        elapsed = (time.time() - start) * 1000

        return DataProfile(
            dataset_id="",
            table_id=table.table_id,
            row_count=table.row_count,
            column_count=table.column_count,
            columns=columns,
            total_missing=total_missing,
            duplicate_rows=duplicate_rows,
            outlier_candidates=outliers,
            correlation_candidates=correlations,
            computation_time_ms=round(elapsed, 2),
        )

    @staticmethod
    def _profile_column(name: str, values: list[Any]) -> ColumnProfile:
        """Profile a single column."""
        non_null = [v for v in values if v is not None and v != ""]
        missing_count = len(values) - len(non_null)
        missing_rate = missing_count / len(values) if values else 0.0

        dtype, semantic_type = DataProfiler._detect_dtype(non_null)

        col = ColumnProfile(
            name=name,
            original_name=name,
            dtype=dtype,
            semantic_type=semantic_type,
            missing_count=missing_count,
            missing_rate=round(missing_rate, 4),
            unique_count=len(set(str(v) for v in non_null)) if non_null else 0,
            example_values=non_null[:5],
        )

        if dtype in ("int64", "float64"):
            nums = DataProfiler._to_numeric(non_null)
            if nums:
                sorted_nums = sorted(nums)
                col.min = min(nums)
                col.max = max(nums)
                col.mean = sum(nums) / len(nums)
                col.median = sorted_nums[len(sorted_nums) // 2]
                col.std = DataProfiler._std_dev(nums, col.mean)
                col.quantiles = {
                    "q25": sorted_nums[len(sorted_nums) // 4],
                    "q50": col.median,
                    "q75": sorted_nums[3 * len(sorted_nums) // 4],
                }

        return col

    @staticmethod
    def _detect_dtype(values: list[Any]) -> tuple[str, SemanticType]:
        if not values:
            return "object", SemanticType.UNKNOWN
        sample = values[:100]
        int_count = 0
        float_count = 0
        for v in sample:
            try:
                int(str(v))
                int_count += 1
            except (ValueError, TypeError):
                try:
                    float(str(v))
                    float_count += 1
                except (ValueError, TypeError):
                    pass
        if int_count == len(sample):
            return "int64", SemanticType.NUMERIC
        if int_count + float_count == len(sample):
            return "float64", SemanticType.NUMERIC
        unique_ratio = len(set(str(v) for v in values)) / len(values) if values else 0
        if unique_ratio > 0.9 and len(values) > 10:
            return "object", SemanticType.IDENTIFIER
        if len(set(str(v) for v in values)) < min(20, len(values) * 0.3):
            return "object", SemanticType.CATEGORICAL
        return "object", SemanticType.TEXT

    @staticmethod
    def _to_numeric(values: list[Any]) -> list[float]:
        result = []
        for v in values:
            try:
                result.append(float(v))
            except (ValueError, TypeError):
                pass
        return result

    @staticmethod
    def _get_numeric_values(
        col_name: str, values: list[Any], headers: list[str], rows: list[list[Any]]
    ) -> list[float]:
        return DataProfiler._to_numeric(values)

    @staticmethod
    def _std_dev(values: list[float], mean: float) -> float:
        if len(values) < 2:
            return 0.0
        return (sum((v - mean) ** 2 for v in values) / (len(values) - 1)) ** 0.5

    @staticmethod
    def _pearson_correlation(x: list[float], y: list[float]) -> Optional[float]:
        if len(x) < 3 or len(y) < 3:
            return None
        n = min(len(x), len(y))
        x = x[:n]
        y = y[:n]
        mean_x = sum(x) / n
        mean_y = sum(y) / n
        num = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
        den_x = (sum((xi - mean_x) ** 2 for xi in x)) ** 0.5
        den_y = (sum((yi - mean_y) ** 2 for yi in y)) ** 0.5
        if den_x == 0 or den_y == 0:
            return None
        return num / (den_x * den_y)

    @staticmethod
    def _count_duplicates(rows: list[list[Any]]) -> int:
        seen = set()
        dupes = 0
        for row in rows:
            key = tuple(str(v) for v in row)
            if key in seen:
                dupes += 1
            else:
                seen.add(key)
        return dupes