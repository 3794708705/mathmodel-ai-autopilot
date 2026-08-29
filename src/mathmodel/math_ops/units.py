"""MathModel AI — Compound unit and dimensional analysis.

Supports base dimensions and compound units like m/s, kg*m/s^2,
currency/item, m^2. No eval() — safe token grammar parsing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

# Base dimensions
DIMENSION_MAP: dict[str, dict[str, int]] = {
    "m": {"length": 1},
    "km": {"length": 1},
    "cm": {"length": 1},
    "mm": {"length": 1},
    "s": {"time": 1},
    "h": {"time": 1},
    "min": {"time": 1},
    "kg": {"mass": 1},
    "g": {"mass": 1},
    "currency": {"currency": 1},
    "$": {"currency": 1},
    "item": {"count": 1},
    "unit": {"count": 1},
    "k": {"temperature": 1},
    "c": {"temperature": 1},
    "dimensionless": {},
    "1": {},
}

# Scale factors relative to canonical unit
SCALE_MAP: dict[str, float] = {
    "m": 1.0, "km": 1000.0, "cm": 0.01, "mm": 0.001,
    "s": 1.0, "h": 3600.0, "min": 60.0,
    "kg": 1.0, "g": 0.001,
    "currency": 1.0, "$": 1.0,
    "item": 1.0, "unit": 1.0,
    "k": 1.0, "c": 1.0,
    "dimensionless": 1.0, "1": 1.0,
}

# Known compound unit shorthands
COMPOUND_MAP: dict[str, dict[str, int]] = {
    "m/s": {"length": 1, "time": -1},
    "m/s^2": {"length": 1, "time": -2},
    "m/s2": {"length": 1, "time": -2},
    "kg/m^3": {"mass": 1, "length": -3},
    "kg/m3": {"mass": 1, "length": -3},
    "kg*m/s^2": {"mass": 1, "length": 1, "time": -2},
    "currency/item": {"currency": 1, "count": -1},
    "item/hour": {"count": 1, "time": -1},
    "item/h": {"count": 1, "time": -1},
    "currency/time": {"currency": 1, "time": -1},
    "currency/hour": {"currency": 1, "time": -1},
    "currency/h": {"currency": 1, "time": -1},
    "m^2": {"length": 2},
    "m2": {"length": 2},
    "m^3": {"length": 3},
    "m3": {"length": 3},
    "km/h": {"length": 1, "time": -1},
}


@dataclass
class Dimension:
    """A dimensional vector (powers per base dimension)."""
    powers: dict[str, int] = field(default_factory=dict)

    def is_dimensionless(self) -> bool:
        return all(p == 0 for p in self.powers.values())

    def __eq__(self, other) -> bool:
        if not isinstance(other, Dimension):
            return False
        dims = set(self.powers) | set(other.powers)
        return all(
            self.powers.get(d, 0) == other.powers.get(d, 0) for d in dims
        )

    def multiply(self, other: "Dimension") -> "Dimension":
        result = dict(self.powers)
        for d, p in other.powers.items():
            result[d] = result.get(d, 0) + p
        return Dimension(result)

    def divide(self, other: "Dimension") -> "Dimension":
        result = dict(self.powers)
        for d, p in other.powers.items():
            result[d] = result.get(d, 0) - p
        return Dimension(result)

    def to_str(self) -> str:
        if self.is_dimensionless():
            return "dimensionless"
        parts = []
        for d in sorted(self.powers):
            p = self.powers[d]
            if p == 0:
                continue
            parts.append(d if p == 1 else f"{d}^{p}")
        return "*".join(parts) if parts else "dimensionless"


def parse_unit(unit: str) -> Optional[Dimension]:
    """Parse a unit string into a Dimension. Returns None if unparseable."""
    unit = unit.strip().lower()
    if not unit or unit == "unknown":
        return None

    # Known compound shorthand
    if unit in COMPOUND_MAP:
        return Dimension(dict(COMPOUND_MAP[unit]))

    # Single base unit
    if unit in DIMENSION_MAP:
        return Dimension(dict(DIMENSION_MAP[unit]))

    # Parse product/quotient: m/s, kg*m/s^2
    # Split on '*' for products
    product_terms = unit.split("*")
    dimension = Dimension()
    for term in product_terms:
        # Split on '/' for division
        parts = term.split("/")
        for i, part in enumerate(parts):
            # Parse power: m^2 or m2
            match = re.match(r"^([a-z$]+)\^?(-?\d+)?$", part)
            if not match:
                return None
            base = match.group(1)
            power = int(match.group(2)) if match.group(2) else 1
            if base not in DIMENSION_MAP:
                return None
            # Division flips the sign
            if i > 0:
                power = -power
            base_dim = DIMENSION_MAP[base]
            for d, p in base_dim.items():
                dimension.powers[d] = dimension.powers.get(d, 0) + p * power
    return dimension


def units_compatible(unit_a: Optional[str], unit_b: Optional[str]) -> tuple[bool, str]:
    """Check whether two unit strings are dimensionally compatible.

    Returns (compatible, message).
    - Both unparseable (None) → (False, "UNKNOWN") — unknown is not auto-pass.
    - One unknown, one known → (False, "UNKNOWN_VS_KNOWN")
    - Dimensions equal → (True, "OK")
    - Dimensions differ → (False, "MISMATCH: a vs b")
    """
    dim_a = parse_unit(unit_a)
    dim_b = parse_unit(unit_b)

    if dim_a is None or dim_b is None:
        return False, "UNKNOWN"

    if dim_a == dim_b:
        return True, "OK"

    return False, f"MISMATCH: {dim_a.to_str()} vs {dim_b.to_str()}"


def check_equation_units(lhs_unit: Optional[str], rhs_unit: Optional[str]) -> tuple[bool, str]:
    """Check unit consistency of an equation's two sides."""
    return units_compatible(lhs_unit, rhs_unit)


def check_objective_units(objective_units: list[Optional[str]]) -> tuple[bool, str]:
    """Check that all objective terms are dimensionally compatible.

    For multi-objective: incompatible units require normalization.
    """
    if len(objective_units) <= 1:
        return True, "OK"

    first = objective_units[0]
    for u in objective_units[1:]:
        compatible, msg = units_compatible(first, u)
        if not compatible:
            return False, f"Multi-objective unit incompatibility: {msg} (normalization required)"
    return True, "OK"