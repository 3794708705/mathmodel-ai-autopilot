"""Phase 7E Hidden Oracle — Emergency Resource Allocation.

This computes the reference LP solution deterministically using SciPy.
Agent systems MUST NOT see this file during the Dry Run.

Problem: Allocate 3 depots → 5 shelters, minimize transport cost,
satisfy demand with 60% minimum service level, total budget ≤ $50,000.
"""

import numpy as np
from scipy.optimize import linprog

# 3 depots × 5 shelters = 15 decision variables
# x[i,j] = units from depot i to shelter j

costs = np.array([
    # D1→S1..S5
    8, 12, 6, 15, 10,
    # D2→S1..S5
    10, 5, 14, 8, 12,
    # D3→S1..S5
    14, 9, 7, 11, 6,
])

supply = [500, 400, 350]  # total = 1250
demand = [200, 150, 180, 250, 120]  # total = 900
total_demand = 900
total_supply = 1250

# Supply constraints: sum over shelters ≤ depot capacity
A_ub = []
b_ub = []
for i in range(3):
    row = np.zeros(15)
    for j in range(5):
        row[i * 5 + j] = 1
    A_ub.append(row)
    b_ub.append(supply[i])

# Demand constraints: sum over depots ≥ demand
A_eq_bounds = []  # Use bounds for equality effectively
for j in range(5):
    row = np.zeros(15)
    for i in range(3):
        row[i * 5 + j] = -1
    A_ub.append(row)
    b_ub.append(-demand[j])  # -sum ≤ -demand → sum ≥ demand

# Minimum service level: each shelter gets ≥ 60% of demand
for j in range(5):
    row = np.zeros(15)
    for i in range(3):
        row[i * 5 + j] = -1
    A_ub.append(row)
    b_ub.append(-0.6 * demand[j])

# Budget: transport cost ≤ 50000
row = costs.copy()
A_ub.append(row)
b_ub.append(50000)

# Solve
result = linprog(costs, A_ub=np.array(A_ub), b_ub=np.array(b_ub),
                 bounds=[(0, None)] * 15, method='highs')

print("Status:", result.status)
print("Status desc:", {0: "OPTIMAL", 1: "ITERATION_LIMIT", 2: "INFEASIBLE", 3: "UNBOUNDED", 4: "NUMERICAL_ISSUES"}.get(result.status, "UNKNOWN"))
print("Objective (transport cost):", result.fun)
print("Optimal allocations:")
for i in range(3):
    for j in range(5):
        val = result.x[i * 5 + j]
        if val > 0.01:
            print(f"  D{i+1}→S{j+1}: {val:.1f}")

# Compute shelter service levels
print("\nService levels:")
for j in range(5):
    received = sum(result.x[i * 5 + j] for i in range(3))
    print(f"  S{j+1}: received={received:.1f} / demand={demand[j]} ({received/demand[j]*100:.1f}%)")

# Fairness: min ratio / max ratio
ratios = [sum(result.x[i * 5 + j] for i in range(3)) / demand[j] for j in range(5)]
print(f"Fairness ratio: min={min(ratios):.3f} max={max(ratios):.3f}")

# Budget
transport_cost = sum(costs[k] * result.x[k] for k in range(15))
print(f"Transport cost: {transport_cost:.1f}")

# Constraint violations
violations = []
for row, b in zip(A_ub, b_ub):
    v = sum(row[k] * result.x[k] for k in range(15)) - b
    if v > 1e-6:
        violations.append(v)
print(f"Max constraint violation: {max(violations) if violations else 0:.6f}")

# Export oracle result
oracle = {
    "status": result.status,
    "objective": float(result.fun),
    "max_violation": max(violations) if violations else 0.0,
    "feasible": result.success,
    "service_levels": {f"S{j+1}": float(ratios[j]) for j in range(5)},
    "fairness_min": float(min(ratios)),
    "fairness_max": float(max(ratios)),
    "transport_cost": float(transport_cost),
}
import json
with open("tests/fixtures/phase7e_hidden_reference/oracle.json", "w") as f:
    json.dump(oracle, f, indent=2)
print("\nOracle saved.")