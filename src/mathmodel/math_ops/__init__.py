"""MathModel AI — Symbol registry, equation registry, unit system, and safe expression parser."""

from __future__ import annotations

import ast
import math
import operator
from enum import Enum
from typing import Any, Callable, Optional

# ═══════════════════════════════════════════════════════════════
# Safe Expression Parser
# ═══════════════════════════════════════════════════════════════

# Allowed AST node types
ALLOWED_NODES = {
    ast.Expression, ast.Constant, ast.Name, ast.Load,
    ast.BinOp, ast.UnaryOp, ast.Add, ast.Sub, ast.Mult, ast.Div,
    ast.Pow, ast.USub, ast.UAdd, ast.Compare, ast.Lt, ast.LtE,
    ast.Gt, ast.GtE, ast.Eq, ast.NotEq, ast.Call, ast.Tuple,
    ast.List, ast.Subscript, ast.Index, ast.Slice,
    ast.Attribute,  # Only for math.xxx
}

# Allowed mathematical functions
ALLOWED_FUNCTIONS = {
    "abs", "min", "max", "sum", "round",
    "sqrt", "exp", "log", "log2", "log10",
    "sin", "cos", "tan", "asin", "acos", "atan",
    "sinh", "cosh", "tanh",
    "ceil", "floor", "trunc",
    "pi", "e", "inf", "nan",
}

# Forbidden function names (even if they look like math)
FORBIDDEN_FUNCTIONS = {
    "__import__", "eval", "exec", "compile", "open",
    "getattr", "setattr", "delattr", "hasattr",
    "globals", "locals", "vars", "dir",
    "type", "object", "super",
}

# Forbidden name patterns
FORBIDDEN_PATTERNS = ["__", "._"]

# Allowed module prefixes for attribute access
ALLOWED_MODULES = {"math", "np", "numpy"}


class ExpressionValidationError(Exception):
    """Raised when an expression fails validation."""
    pass


def validate_expression(expr: str, allowed_symbols: Optional[set[str]] = None) -> ast.Expression:
    """Validate a mathematical expression string.

    Returns the parsed AST if valid.
    Raises ExpressionValidationError if unsafe.
    """
    # Strip whitespace
    expr = expr.strip()
    if not expr:
        raise ExpressionValidationError("Empty expression")

    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        raise ExpressionValidationError(f"Syntax error: {e}")

    allowed = allowed_symbols or set()
    validator = _ExpressionValidator(allowed)
    validator.visit(tree)

    if validator.errors:
        raise ExpressionValidationError("; ".join(validator.errors))

    return tree


class _ExpressionValidator(ast.NodeVisitor):
    """Validates AST nodes against the allowlist."""

    def __init__(self, allowed_symbols: set[str]):
        self.allowed_symbols = allowed_symbols
        self.errors: list[str] = []

    def visit(self, node: ast.AST) -> None:
        # Check node type
        if type(node) not in ALLOWED_NODES:
            self.errors.append(f"Forbidden node type: {type(node).__name__}")
            return

        # Check for forbidden name patterns
        if isinstance(node, ast.Name):
            name = node.id
            if name.startswith("__"):
                self.errors.append(f"Dunder name not allowed: {name}")
                return
            if name in FORBIDDEN_FUNCTIONS:
                self.errors.append(f"Forbidden function: {name}")
                return

        # Check function calls
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                func_name = node.func.id
                if func_name in FORBIDDEN_FUNCTIONS:
                    self.errors.append(f"Forbidden function: {func_name}")
                    return
                if func_name not in ALLOWED_FUNCTIONS:
                    self.errors.append(f"Function not allowed: {func_name}")
                    return
            elif isinstance(node.func, ast.Attribute):
                if isinstance(node.func.value, ast.Name):
                    if node.func.value.id not in ALLOWED_MODULES:
                        self.errors.append(f"Module not allowed: {node.func.value.id}")
                        return
                else:
                    self.errors.append("Complex attribute access not allowed")
                    return
            else:
                self.errors.append("Complex function call not allowed")
                return

        # Check attribute access
        if isinstance(node, ast.Attribute):
            if node.attr.startswith("__"):
                self.errors.append(f"Dunder attribute not allowed: {node.attr}")
                return

        # Check for lambda, comprehensions, imports
        if isinstance(node, (ast.Lambda, ast.ListComp, ast.SetComp, ast.DictComp,
                            ast.GeneratorExp, ast.Import, ast.ImportFrom)):
            self.errors.append(f"Forbidden construct: {type(node).__name__}")
            return

        # Generic visit
        super().visit(node)

    def generic_visit(self, node: ast.AST) -> None:
        if type(node) not in ALLOWED_NODES:
            self.errors.append(f"Forbidden node type: {type(node).__name__}")
            return
        super().generic_visit(node)


# ═══════════════════════════════════════════════════════════════
# Symbol Registry
# ═══════════════════════════════════════════════════════════════

class SymbolEntry:
    """A registered symbol in the model."""

    def __init__(
        self,
        symbol: str,
        entity_id: str,
        entity_type: str,  # "variable", "parameter", "constant"
        meaning: str = "",
        unit: Optional[str] = None,
        domain: str = "",
    ):
        self.symbol = symbol
        self.entity_id = entity_id
        self.entity_type = entity_type
        self.meaning = meaning
        self.unit = unit
        self.domain = domain
        self.aliases: list[str] = []
        self.used_in: list[str] = []


class SymbolRegistry:
    """Registry of all symbols in a mathematical model."""

    def __init__(self):
        self._symbols: dict[str, SymbolEntry] = {}

    def register(
        self,
        symbol: str,
        entity_id: str,
        entity_type: str,
        meaning: str = "",
        unit: Optional[str] = None,
        domain: str = "",
    ) -> SymbolEntry:
        """Register a symbol."""
        if symbol in self._symbols:
            existing = self._symbols[symbol]
            if existing.entity_id != entity_id:
                raise ValueError(
                    f"Symbol '{symbol}' already registered by {existing.entity_id} "
                    f"(type={existing.entity_type}), cannot re-register for {entity_id}"
                )
            return existing

        entry = SymbolEntry(
            symbol=symbol,
            entity_id=entity_id,
            entity_type=entity_type,
            meaning=meaning,
            unit=unit,
            domain=domain,
        )
        self._symbols[symbol] = entry
        return entry

    def lookup(self, symbol: str) -> Optional[SymbolEntry]:
        return self._symbols.get(symbol)

    def resolve(self, symbol: str) -> SymbolEntry:
        entry = self.lookup(symbol)
        if entry is None:
            raise KeyError(f"Symbol '{symbol}' not registered")
        return entry

    def get_all_symbols(self) -> set[str]:
        return set(self._symbols.keys())

    def get_unused(self) -> list[SymbolEntry]:
        return [e for e in self._symbols.values() if not e.used_in]

    def mark_used_in(self, symbol: str, equation_id: str) -> None:
        entry = self.resolve(symbol)
        if equation_id not in entry.used_in:
            entry.used_in.append(equation_id)

    @property
    def symbol_count(self) -> int:
        return len(self._symbols)


# ═══════════════════════════════════════════════════════════════
# Equation Registry
# ═══════════════════════════════════════════════════════════════

class EquationRegistry:
    """Registry of all equations in a mathematical model."""

    def __init__(self):
        self._equations: dict[str, Any] = {}  # equation_id -> Equation

    def register(self, equation: Any) -> None:
        if equation.equation_id in self._equations:
            raise ValueError(f"Duplicate equation ID: {equation.equation_id}")
        self._equations[equation.equation_id] = equation

    def lookup(self, equation_id: str) -> Optional[Any]:
        return self._equations.get(equation_id)

    def validate_dependencies(self) -> list[str]:
        """Check that all equation dependencies exist."""
        issues = []
        all_ids = set(self._equations.keys())
        for eq in self._equations.values():
            for dep_id in eq.dependencies:
                if dep_id not in all_ids:
                    issues.append(f"Equation {eq.equation_id}: unknown dependency {dep_id}")
                if dep_id == eq.equation_id:
                    issues.append(f"Equation {eq.equation_id}: self-dependency")
        return issues

    def get_all(self) -> list[Any]:
        return list(self._equations.values())

    @property
    def equation_count(self) -> int:
        return len(self._equations)


# ═══════════════════════════════════════════════════════════════
# Unit System
# ═══════════════════════════════════════════════════════════════

class Dimension(str, Enum):
    MASS = "M"
    LENGTH = "L"
    TIME = "T"
    CURRENCY = "C"
    COUNT = "N"
    TEMPERATURE = "K"
    DIMENSIONLESS = "1"


class UnitChecker:
    """Validates unit consistency in mathematical expressions."""

    @staticmethod
    def check_compatible(unit_a: Optional[str], unit_b: Optional[str]) -> bool:
        """Check if two units are compatible (same dimension)."""
        if unit_a is None or unit_b is None:
            return True  # Unknown units are not an error, just unknown
        if unit_a == "dimensionless" or unit_b == "dimensionless":
            return True
        return unit_a == unit_b

    @staticmethod
    def check_addition(unit_a: Optional[str], unit_b: Optional[str]) -> list[str]:
        """Check unit compatibility for addition/subtraction."""
        issues = []
        if unit_a and unit_b and unit_a != unit_b:
            if unit_a != "dimensionless" and unit_b != "dimensionless":
                issues.append(f"Unit mismatch: {unit_a} + {unit_b}")
        return issues

    @staticmethod
    def check_equation(lhs_unit: Optional[str], rhs_unit: Optional[str]) -> list[str]:
        """Check both sides of an equation have compatible units."""
        return UnitChecker.check_addition(lhs_unit, rhs_unit)