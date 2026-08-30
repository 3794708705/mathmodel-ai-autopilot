"""MathModel AI — Phase 7D: Competition Budget.

Deterministic budget tracking: token/call/solver/simulation
budgets with reservation, consumption, and reconciliation.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Optional
from uuid import uuid4


# ═══════════════════════════════════════════════════════════════
# Resource Types
# ═══════════════════════════════════════════════════════════════

class ResourceType(str, Enum):
    LLM_CALL = "llm_call"
    LLM_TOKEN = "llm_token"
    SOLVER_SECOND = "solver_second"
    SIMULATION_RUN = "simulation_run"


class ReservationStatus(str, Enum):
    RESERVED = "reserved"
    CONSUMED = "consumed"
    RELEASED = "released"
    EXPIRED = "expired"


# ═══════════════════════════════════════════════════════════════
# Competition Budget
# ═══════════════════════════════════════════════════════════════

@dataclass
class CompetitionBudget:
    competition_budget_id: str = field(default_factory=lambda: f"BUDGET-{uuid4().hex[:8]}")
    # LLM budgets
    total_llm_token_budget: Optional[int] = None
    remaining_llm_token_budget: Optional[int] = None
    total_llm_call_budget: Optional[int] = None
    remaining_llm_calls: Optional[int] = None
    # Solver budgets
    total_solver_seconds: Optional[int] = None
    remaining_solver_seconds: Optional[int] = None
    total_simulation_runs: Optional[int] = None
    remaining_simulation_runs: Optional[int] = None
    # Critical reserve
    critical_reserve_fraction: float = 0.15
    # Wall-clock
    total_wall_clock_seconds: Optional[int] = None
    # Monetary (informational only)
    monetary_budget: Optional[float] = None
    monetary_spend: float = 0.0
    # Metadata
    budget_source: str = "default"
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    version: int = 1

    def __post_init__(self):
        if self.remaining_llm_token_budget is None:
            self.remaining_llm_token_budget = self.total_llm_token_budget
        if self.remaining_llm_calls is None:
            self.remaining_llm_calls = self.total_llm_call_budget
        if self.remaining_solver_seconds is None:
            self.remaining_solver_seconds = self.total_solver_seconds
        if self.remaining_simulation_runs is None:
            self.remaining_simulation_runs = self.total_simulation_runs

    def remaining(self, resource_type: ResourceType) -> Optional[int]:
        map_ = {
            ResourceType.LLM_TOKEN: self.remaining_llm_token_budget,
            ResourceType.LLM_CALL: self.remaining_llm_calls,
            ResourceType.SOLVER_SECOND: self.remaining_solver_seconds,
            ResourceType.SIMULATION_RUN: self.remaining_simulation_runs,
        }
        return map_.get(resource_type)

    def critical_reserve(self, resource_type: ResourceType) -> float:
        """Return the critical reserve amount for a resource."""
        total = self.remaining(resource_type)
        if total is None or total <= 0:
            return 0
        return int(total * self.critical_reserve_fraction)

    def available_for(self, resource_type: ResourceType, is_critical: bool) -> Optional[int]:
        """Return the amount available, considering critical reserve."""
        remaining = self.remaining(resource_type)
        if remaining is None:
            return None
        if is_critical:
            return remaining  # critical tasks can use everything
        reserve = self.critical_reserve(resource_type)
        return max(0, remaining - reserve)


# ═══════════════════════════════════════════════════════════════
# Budget Reservation
# ═══════════════════════════════════════════════════════════════

@dataclass
class BudgetReservation:
    reservation_id: str = field(default_factory=lambda: f"RES-{uuid4().hex[:8]}")
    budget_id: str = ""
    resource_type: ResourceType = ResourceType.LLM_CALL
    amount: int = 0
    owner_run_id: str = ""
    expires_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc) + timedelta(minutes=5))
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    status: ReservationStatus = ReservationStatus.RESERVED


# ═══════════════════════════════════════════════════════════════
# Budget Ledger Entry
# ═══════════════════════════════════════════════════════════════

@dataclass
class BudgetLedgerEntry:
    ledger_entry_id: str = field(default_factory=lambda: f"LE-{uuid4().hex[:8]}")
    reservation_id: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    run_id: str = ""
    agent: str = ""
    task_type: str = ""
    provider: str = ""
    model: str = ""
    runtime_mode: str = ""
    resource_type: ResourceType = ResourceType.LLM_CALL
    reserved_amount: int = 0
    actual_amount: int = 0
    released_amount: int = 0
    remaining_after: int = 0
    reason: str = ""
    status: str = ""


# ═══════════════════════════════════════════════════════════════
# Budget Ledger
# ═══════════════════════════════════════════════════════════════

class BudgetLedger:
    """Thread-safe budget ledger with reservation/consume/reconcile."""

    def __init__(self, budget: CompetitionBudget):
        self._budget = budget
        self._lock = threading.Lock()
        self._reservations: dict[str, BudgetReservation] = {}
        self._entries: list[BudgetLedgerEntry] = []
        self._reconciled: set[str] = set()  # reservation_id → reconciled

    @property
    def budget(self) -> CompetitionBudget:
        return self._budget

    def reserve(
        self,
        resource_type: ResourceType,
        amount: int,
        owner_run_id: str,
        is_critical: bool = False,
    ) -> Optional[BudgetReservation]:
        """Attempt to reserve budget. Returns None if insufficient."""
        with self._lock:
            available = self._budget.available_for(resource_type, is_critical)
            if available is None:
                available = amount  # unlimited
            if available < amount:
                return None

            remaining = self._budget.remaining(resource_type)
            if remaining is not None:
                self._decrement(resource_type, amount)

            reservation = BudgetReservation(
                budget_id=self._budget.competition_budget_id,
                resource_type=resource_type,
                amount=amount,
                owner_run_id=owner_run_id,
            )
            self._reservations[reservation.reservation_id] = reservation
            return reservation

    def reserve_multi(
        self,
        resources: dict[ResourceType, int],
        owner_run_id: str,
        is_critical: bool = False,
    ) -> Optional[list[BudgetReservation]]:
        """Atomic multi-resource reservation. All or nothing."""
        with self._lock:
            # Check all available first
            for rt, amount in resources.items():
                available = self._budget.available_for(rt, is_critical)
                if available is not None and available < amount:
                    return None

            # Allocate all
            reservations = []
            for rt, amount in resources.items():
                remaining = self._budget.remaining(rt)
                if remaining is not None:
                    self._decrement(rt, amount)
                res = BudgetReservation(
                    budget_id=self._budget.competition_budget_id,
                    resource_type=rt,
                    amount=amount,
                    owner_run_id=owner_run_id,
                )
                self._reservations[res.reservation_id] = res
                reservations.append(res)
            return reservations

    def consume(self, reservation_id: str, actual_amount: int) -> BudgetLedgerEntry:
        """Consume a reservation (call was made). actual_amount=0 means UNKNOWN."""
        with self._lock:
            # If actual_amount is 0, treat as UNKNOWN — consume full reserved
            if actual_amount == 0:
                res = self._reservations.get(reservation_id)
                if res:
                    actual_amount = res.amount  # conservative: consume all
            return self._reconcile(reservation_id, actual_amount, "consumed")

    def refill(
        self, resource_type: ResourceType, amount: int, actor: str, reason: str,
    ) -> BudgetLedgerEntry:
        """Authorized budget refill. Records audit trail."""
        with self._lock:
            self._increment(resource_type, amount)
            # Create a dummy reservation for ledger entry
            res = BudgetReservation(
                budget_id=self._budget.competition_budget_id,
                resource_type=resource_type,
                amount=amount,
                owner_run_id=actor,
            )
            self._reservations[res.reservation_id] = res
            entry = self._add_entry(res, 0, 0, f"refill by {actor}: {reason}")
            res.status = ReservationStatus.RELEASED
            return entry

    def release(self, reservation_id: str) -> None:
        """Release a reservation (call was not made)."""
        with self._lock:
            res = self._reservations.get(reservation_id)
            if res and res.status == ReservationStatus.RESERVED:
                self._increment(res.resource_type, res.amount)
                res.status = ReservationStatus.RELEASED
                self._add_entry(res, 0, res.amount, "released")

    def expire_reservations(self) -> int:
        """Expire stale reservations. Returns count of expired."""
        with self._lock:
            now = datetime.now(timezone.utc)
            expired = 0
            for rid, res in list(self._reservations.items()):
                if res.status == ReservationStatus.RESERVED and res.expires_at < now:
                    self._increment(res.resource_type, res.amount)
                    res.status = ReservationStatus.EXPIRED
                    self._add_entry(res, 0, res.amount, "expired")
                    expired += 1
            return expired

    def _reconcile(
        self, reservation_id: str, actual_amount: int, status: str,
    ) -> BudgetLedgerEntry:
        """Reconcile a reservation. Idempotent."""
        if reservation_id in self._reconciled:
            # Return existing entry
            for e in reversed(self._entries):
                if e.reservation_id == reservation_id:
                    return e
            raise ValueError(f"Reservation {reservation_id} reconciled but no entry")

        res = self._reservations.get(reservation_id)
        if not res:
            raise ValueError(f"Unknown reservation: {reservation_id}")

        if res.status != ReservationStatus.RESERVED:
            raise ValueError(f"Reservation {reservation_id} already {res.status.value}")

        # Adjust: actual may differ from reserved
        diff = res.amount - actual_amount
        if diff > 0:
            self._increment(res.resource_type, diff)

        res.status = ReservationStatus.CONSUMED
        entry = self._add_entry(res, actual_amount, diff, status)
        self._reconciled.add(reservation_id)
        return entry

    def _decrement(self, resource_type: ResourceType, amount: int) -> None:
        self._modify(resource_type, -amount)

    def _increment(self, resource_type: ResourceType, amount: int) -> None:
        self._modify(resource_type, amount)

    def _modify(self, resource_type: ResourceType, delta: int) -> None:
        if resource_type == ResourceType.LLM_TOKEN:
            if self._budget.remaining_llm_token_budget is not None:
                self._budget.remaining_llm_token_budget += delta
        elif resource_type == ResourceType.LLM_CALL:
            if self._budget.remaining_llm_calls is not None:
                self._budget.remaining_llm_calls += delta
        elif resource_type == ResourceType.SOLVER_SECOND:
            if self._budget.remaining_solver_seconds is not None:
                self._budget.remaining_solver_seconds += delta
        elif resource_type == ResourceType.SIMULATION_RUN:
            if self._budget.remaining_simulation_runs is not None:
                self._budget.remaining_simulation_runs += delta

    def _add_entry(
        self, res: BudgetReservation, actual: int, released: int, status: str,
    ) -> BudgetLedgerEntry:
        entry = BudgetLedgerEntry(
            reservation_id=res.reservation_id,
            run_id=res.owner_run_id,
            resource_type=res.resource_type,
            reserved_amount=res.amount,
            actual_amount=actual,
            released_amount=released,
            remaining_after=self._budget.remaining(res.resource_type) or 0,
            reason=f"reservation {res.reservation_id} {status}",
            status=status,
        )
        self._entries.append(entry)
        return entry

    @property
    def entries(self) -> list[BudgetLedgerEntry]:
        return list(self._entries)

    @property
    def active_reservations(self) -> list[BudgetReservation]:
        return [r for r in self._reservations.values() if r.status == ReservationStatus.RESERVED]