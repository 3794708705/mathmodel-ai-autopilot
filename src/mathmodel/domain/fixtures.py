"""MathModel AI — Competition fixtures for Phase 2 testing.

Three competition-style problems used to test Reasoning Core behavior.
NOT for benchmarking model quality — for testing domain logic.
"""

from __future__ import annotations

from mathmodel.domain.analysis import (
    Ambiguity,
    AmbiguityImpact,
    ModelingTaskType,
    ProblemAnalysis,
    Subproblem,
)
from mathmodel.domain.evidence import EvidenceItem, EvidenceType
from mathmodel.domain.candidates import (
    ModelCandidate,
    ModelComponent,
    ModelFamily,
    ComponentRole,
)


# ═══════════════════════════════════════════════════════════════
# Fixture A: Resource Allocation / Optimization
# ═══════════════════════════════════════════════════════════════

FIXTURE_A_PROBLEM = """
A manufacturing company produces three products (P1, P2, P3) using two machines
(M1, M2). Each product requires processing time on both machines. The available
machine hours per week are 40 for M1 and 35 for M2. The processing times (in hours
per unit) are:

        M1    M2
P1      2     1.5
P2      1.5   2
P3      1     1

The profit per unit is $30 for P1, $25 for P2, and $20 for P3. Market demand
limits P1 to at most 15 units per week, and the company must produce at least 5
units of P3 to meet contractual obligations.

Determine the optimal production mix to maximize weekly profit.
"""

FIXTURE_A_ANALYSIS = ProblemAnalysis(
    analysis_id="PA-FIXTURE-A",
    background="Manufacturing production planning with resource constraints",
    core_problem="Determine optimal production quantities to maximize profit given machine time constraints and demand limits",
    objectives=["Maximize weekly profit", "Determine production quantities for P1, P2, P3"],
    subproblems=[
        Subproblem(
            subproblem_id="SUB-A1",
            original_text="Determine the optimal production mix",
            normalized_goal="Maximize profit subject to machine time and demand constraints",
            task_types=[ModelingTaskType.OPTIMIZATION],
            inputs=["Machine hours available", "Processing times", "Profit per unit"],
            expected_outputs=["Optimal quantities for P1, P2, P3", "Maximum profit"],
            constraints=["Machine M1: 40 hours", "Machine M2: 35 hours"],
            priority=1,
        ),
    ],
    evidence=[
        EvidenceItem(
            evidence_id="EVD-A1",
            type=EvidenceType.FACT,
            content="Product P1 requires 2h on M1 and 1.5h on M2",
            source="Problem statement",
            source_location="Processing times table",
            confidence=1.0,
        ),
        EvidenceItem(
            evidence_id="EVD-A2",
            type=EvidenceType.FACT,
            content="Profit per unit: P1=$30, P2=$25, P3=$20",
            source="Problem statement",
            confidence=1.0,
        ),
        EvidenceItem(
            evidence_id="EVD-A3",
            type=EvidenceType.PROPOSED_ASSUMPTION,
            content="Production quantities are continuous (divisible)",
            source="Modeling assumption",
            confidence=0.8,
        ),
        EvidenceItem(
            evidence_id="EVD-A4",
            type=EvidenceType.FACT,
            content="P1 demand limited to 15 units, P3 minimum 5 units",
            source="Problem statement",
            confidence=1.0,
        ),
    ],
    explicit_constraints=[
        "Machine M1: 40 hours/week",
        "Machine M2: 35 hours/week",
        "P1 demand ≤ 15 units",
        "P3 production ≥ 5 units",
    ],
    implicit_conditions=["Production quantities must be non-negative"],
    ambiguities=[
        Ambiguity(
            ambiguity_id="AMB-A1",
            description="Are production quantities continuous or integer?",
            interpretations=["Continuous (LP)", "Integer (IP)"],
            preferred_interpretation="Continuous (LP) — typical for production planning at scale",
            preference_reason="Standard LP formulation unless problem specifies integer constraints",
            confidence=0.7,
            impact=AmbiguityImpact.MEDIUM,
        ),
    ],
    possible_traps=["Forgetting non-negativity constraints", "Confusing LP with IP"],
    modeling_tasks=[ModelingTaskType.OPTIMIZATION],
    confidence=0.9,
)

FIXTURE_A_CANDIDATES = [
    ModelCandidate(
        candidate_id="CAND-A1",
        name="Linear Programming",
        model_family=ModelFamily.LINEAR_PROGRAMMING,
        applicable_subproblems=["SUB-A1"],
        summary="Standard LP formulation with continuous decision variables",
        mathematical_structure="max c^T x s.t. Ax ≤ b, x ≥ 0",
        components=[
            ModelComponent(
                component_id="MC-A1-1",
                name="LP Solver",
                model_family=ModelFamily.LINEAR_PROGRAMMING,
                role=ComponentRole.CORE_MODEL,
                inputs=["Coefficients", "Constraints"],
                outputs=["Optimal solution"],
            ),
        ],
        required_data=["Processing times", "Machine hours", "Profit margins"],
        required_assumptions=["Continuous production quantities"],
        strengths=["Well-understood", "Fast to solve", "Clear interpretation"],
        weaknesses=["Assumes continuity", "No integer constraints"],
        implementation_plan="Use SciPy linprog or Gurobi LP solver",
        validation_plan="Check constraint satisfaction, compare with integer solution",
        risk_flags=[],
    ),
    ModelCandidate(
        candidate_id="CAND-A2",
        name="Integer Linear Programming",
        model_family=ModelFamily.INTEGER_PROGRAMMING,
        applicable_subproblems=["SUB-A1"],
        summary="ILP formulation with integer decision variables",
        mathematical_structure="max c^T x s.t. Ax ≤ b, x ∈ Z⁺",
        components=[
            ModelComponent(
                component_id="MC-A2-1",
                name="ILP Solver",
                model_family=ModelFamily.INTEGER_PROGRAMMING,
                role=ComponentRole.CORE_MODEL,
                inputs=["Coefficients", "Constraints"],
                outputs=["Integer optimal solution"],
            ),
        ],
        required_data=["Processing times", "Machine hours", "Profit margins"],
        required_assumptions=["Integer production quantities required"],
        strengths=["Realistic integer solutions", "Exact"],
        weaknesses=["Computationally harder", "May be slower for large instances"],
        implementation_plan="Use Gurobi MILP solver or OR-Tools",
        validation_plan="Compare with LP relaxation, verify integrality",
        risk_flags=[],
    ),
    ModelCandidate(
        candidate_id="CAND-A3",
        name="Goal Programming",
        model_family=ModelFamily.MULTI_OBJECTIVE,
        applicable_subproblems=["SUB-A1"],
        summary="Multi-objective optimization balancing profit and production constraints",
        mathematical_structure="Minimize weighted deviation from goals",
        components=[
            ModelComponent(
                component_id="MC-A3-1",
                name="Goal Programming Model",
                model_family=ModelFamily.MULTI_OBJECTIVE,
                role=ComponentRole.CORE_MODEL,
                inputs=["Goals", "Weights", "Constraints"],
                outputs=["Compromise solution"],
            ),
        ],
        required_data=["Processing times", "Machine hours", "Profit margins", "Goal priorities"],
        required_assumptions=["Goal priorities can be quantified"],
        strengths=["Handles multiple objectives", "Flexible"],
        weaknesses=["Requires goal weights", "More complex formulation"],
        implementation_plan="Formulate as LP with deviation variables",
        validation_plan="Check if goals are met within tolerances",
        risk_flags=[],
    ),
]

# ═══════════════════════════════════════════════════════════════
# Fixture B: Prediction + Optimization Chain
# ═══════════════════════════════════════════════════════════════

FIXTURE_B_PROBLEM = """
A retail chain wants to optimize inventory levels across 5 stores. Historical
daily sales data (in units) for the past 90 days is provided in the attached
CSV file. Each store has different storage capacity and different demand patterns.
Holding cost is $2 per unit per day, and stockout cost is $10 per unit.
The company wants to:

1. Forecast demand for the next 7 days for each store
2. Determine optimal inventory levels that minimize total cost (holding + stockout)
"""

FIXTURE_B_ANALYSIS = ProblemAnalysis(
    analysis_id="PA-FIXTURE-B",
    background="Retail inventory management with demand forecasting and cost optimization",
    core_problem="Minimize total inventory cost by forecasting demand and optimizing stock levels",
    objectives=[
        "Forecast demand for next 7 days per store",
        "Determine optimal inventory levels minimizing holding + stockout costs",
    ],
    subproblems=[
        Subproblem(
            subproblem_id="SUB-B1",
            original_text="Forecast demand for the next 7 days",
            normalized_goal="Predict daily demand for each store using historical data",
            task_types=[ModelingTaskType.PREDICTION, ModelingTaskType.TIME_SERIES],
            inputs=["90 days of historical daily sales", "Store identifiers"],
            expected_outputs=["7-day demand forecast per store"],
            constraints=[],
            priority=1,
        ),
        Subproblem(
            subproblem_id="SUB-B2",
            original_text="Determine optimal inventory levels",
            normalized_goal="Minimize holding + stockout costs given demand forecasts",
            task_types=[ModelingTaskType.OPTIMIZATION],
            inputs=["Demand forecasts from SUB-B1", "Holding cost=$2/unit/day", "Stockout cost=$10/unit"],
            expected_outputs=["Optimal inventory levels per store per day"],
            constraints=["Store storage capacity limits"],
            dependencies=["SUB-B1"],
            priority=2,
        ),
    ],
    evidence=[
        EvidenceItem(
            evidence_id="EVD-B1",
            type=EvidenceType.DATA,
            content="Historical daily sales data for 90 days across 5 stores",
            source="Attached CSV file",
            confidence=1.0,
        ),
        EvidenceItem(
            evidence_id="EVD-B2",
            type=EvidenceType.FACT,
            content="Holding cost: $2/unit/day, Stockout cost: $10/unit",
            source="Problem statement",
            confidence=1.0,
        ),
    ],
    explicit_constraints=["Store storage capacity (varies by store)"],
    implicit_conditions=["Demand forecasts have uncertainty", "Lead time is zero (assumed)"],
    ambiguities=[],
    possible_traps=["Ignoring forecast uncertainty in optimization", "Not modeling stockout cost correctly"],
    modeling_tasks=[ModelingTaskType.PREDICTION, ModelingTaskType.OPTIMIZATION, ModelingTaskType.TIME_SERIES],
    confidence=0.85,
)

FIXTURE_B_CANDIDATES = [
    ModelCandidate(
        candidate_id="CAND-B1",
        name="ARIMA + Newsvendor",
        model_family=ModelFamily.HYBRID,
        applicable_subproblems=["SUB-B1", "SUB-B2"],
        summary="ARIMA time series forecasting combined with Newsvendor inventory model",
        mathematical_structure="Forecast: ARIMA(p,d,q). Optimization: Newsvendor critical ratio",
        components=[
            ModelComponent(
                component_id="MC-B1-1",
                name="ARIMA Forecaster",
                model_family=ModelFamily.TIME_SERIES,
                role=ComponentRole.CORE_MODEL,
                inputs=["Historical sales data"],
                outputs=["Demand forecasts with uncertainty"],
            ),
            ModelComponent(
                component_id="MC-B1-2",
                name="Newsvendor Optimizer",
                model_family=ModelFamily.STOCHASTIC_PROGRAMMING,
                role=ComponentRole.CORE_MODEL,
                inputs=["Demand forecasts", "Cost parameters"],
                outputs=["Optimal inventory levels"],
                dependencies=["MC-B1-1"],
            ),
        ],
        required_data=["90 days historical sales", "Store capacities"],
        required_assumptions=["Demand follows known distribution", "Zero lead time"],
        strengths=["Classic approach", "Handles uncertainty", "Interpretable"],
        weaknesses=["ARIMA assumes stationarity", "Newsvendor assumes single period"],
        implementation_plan="statsmodels ARIMA + analytical Newsvendor formula",
        validation_plan="Backtest forecasts, compare with actual inventory costs",
        risk_flags=[],
    ),
    ModelCandidate(
        candidate_id="CAND-B2",
        name="LSTM + Stochastic Programming",
        model_family=ModelFamily.HYBRID,
        applicable_subproblems=["SUB-B1", "SUB-B2"],
        summary="Deep learning LSTM forecasting with stochastic programming optimization",
        mathematical_structure="LSTM neural network for forecasting, scenario-based stochastic optimization",
        components=[
            ModelComponent(
                component_id="MC-B2-1",
                name="LSTM Forecaster",
                model_family=ModelFamily.CLASSIFICATION_ML,
                role=ComponentRole.CORE_MODEL,
                inputs=["Historical sales data"],
                outputs=["Demand forecasts with confidence intervals"],
            ),
            ModelComponent(
                component_id="MC-B2-2",
                name="Stochastic Optimizer",
                model_family=ModelFamily.STOCHASTIC_PROGRAMMING,
                role=ComponentRole.CORE_MODEL,
                inputs=["Demand scenarios", "Cost parameters"],
                outputs=["Robust inventory levels"],
                dependencies=["MC-B2-1"],
            ),
        ],
        required_data=["90 days historical sales", "Store capacities"],
        required_assumptions=["Sufficient data for LSTM training", "Scenario generation is adequate"],
        strengths=["Captures non-linear patterns", "Handles complex demand"],
        weaknesses=["Requires careful tuning", "Less interpretable", "Computationally expensive"],
        implementation_plan="PyTorch LSTM + scenario-based linear program",
        validation_plan="Compare LSTM vs ARIMA forecast accuracy, stress test scenarios",
        risk_flags=["high_computational_cost"],
    ),
    ModelCandidate(
        candidate_id="CAND-B3",
        name="Exponential Smoothing + Safety Stock",
        model_family=ModelFamily.HYBRID,
        applicable_subproblems=["SUB-B1", "SUB-B2"],
        summary="Simple exponential smoothing with safety stock optimization",
        mathematical_structure="Holt-Winters forecasting + safety stock = z * σ * √L",
        components=[
            ModelComponent(
                component_id="MC-B3-1",
                name="Exp Smoothing Forecaster",
                model_family=ModelFamily.TIME_SERIES,
                role=ComponentRole.CORE_MODEL,
                inputs=["Historical sales data"],
                outputs=["Demand forecasts"],
            ),
            ModelComponent(
                component_id="MC-B3-2",
                name="Safety Stock Calculator",
                model_family=ModelFamily.STOCHASTIC_PROGRAMMING,
                role=ComponentRole.CORE_MODEL,
                inputs=["Forecast errors", "Service level"],
                outputs=["Safety stock levels"],
                dependencies=["MC-B3-1"],
            ),
        ],
        required_data=["90 days historical sales", "Store capacities"],
        required_assumptions=["Demand variability is stable", "Normal distribution of errors"],
        strengths=["Simple", "Fast", "Easy to explain"],
        weaknesses=["Less accurate for complex patterns", "Assumes normal errors"],
        implementation_plan="statsmodels Holt-Winters + analytical safety stock",
        validation_plan="Compare forecast accuracy, evaluate stockout frequency",
        risk_flags=[],
    ),
]

# ═══════════════════════════════════════════════════════════════
# Fixture C: Network / Routing / Evaluation
# ═══════════════════════════════════════════════════════════════

FIXTURE_C_PROBLEM = """
A delivery company operates a distribution center serving 10 retail locations
in a city. The distances between all locations are given in the attached table.
Each vehicle can carry up to 50 packages and must return to the distribution
center at the end of its route. The company has 3 vehicles available.
Each retail location has a known daily demand (in packages) that must be met.

Determine:
1. The optimal routes for the 3 vehicles
2. The total distance traveled
3. Whether 3 vehicles are sufficient to meet all demands
"""

FIXTURE_C_ANALYSIS = ProblemAnalysis(
    analysis_id="PA-FIXTURE-C",
    background="Vehicle routing with capacity constraints for last-mile delivery",
    core_problem="Minimize total travel distance while satisfying all delivery demands with limited vehicle capacity",
    objectives=[
        "Find optimal routes for 3 vehicles",
        "Minimize total distance traveled",
        "Verify fleet sufficiency",
    ],
    subproblems=[
        Subproblem(
            subproblem_id="SUB-C1",
            original_text="Determine optimal routes for the 3 vehicles",
            normalized_goal="Solve Capacitated Vehicle Routing Problem (CVRP) with 3 vehicles, 10 locations",
            task_types=[ModelingTaskType.ROUTING, ModelingTaskType.OPTIMIZATION, ModelingTaskType.GRAPH_NETWORK],
            inputs=["Distance matrix (10x10)", "Demand per location (10 values)", "Vehicle capacity=50"],
            expected_outputs=["3 vehicle routes", "Total distance"],
            constraints=["Vehicle capacity ≤ 50 packages", "Must return to depot", "All demands must be met"],
            priority=1,
        ),
        Subproblem(
            subproblem_id="SUB-C2",
            original_text="Verify whether 3 vehicles are sufficient",
            normalized_goal="Check if total demand ≤ 3 × 50 and if routing is feasible",
            task_types=[ModelingTaskType.EVALUATION],
            inputs=["Total demand", "Vehicle capacity", "Route feasibility from SUB-C1"],
            expected_outputs=["Feasibility assessment", "Recommendation if more vehicles needed"],
            dependencies=["SUB-C1"],
            priority=2,
        ),
    ],
    evidence=[
        EvidenceItem(
            evidence_id="EVD-C1",
            type=EvidenceType.DATA,
            content="Distance matrix between 10 locations + depot",
            source="Attached table",
            confidence=1.0,
        ),
        EvidenceItem(
            evidence_id="EVD-C2",
            type=EvidenceType.FACT,
            content="Vehicle capacity: 50 packages, Fleet size: 3 vehicles",
            source="Problem statement",
            confidence=1.0,
        ),
    ],
    explicit_constraints=[
        "Vehicle capacity: 50 packages",
        "3 vehicles available",
        "Must return to depot",
        "All locations must be served",
    ],
    implicit_conditions=[
        "Symmetric distances (assumed unless stated otherwise)",
        "Single depot",
        "No time windows",
    ],
    ambiguities=[
        Ambiguity(
            ambiguity_id="AMB-C1",
            description="Can a vehicle make multiple trips?",
            interpretations=["Single trip per vehicle", "Multiple trips allowed"],
            preferred_interpretation="Single trip per vehicle",
            preference_reason="Standard CVRP formulation assumes single trip",
            confidence=0.8,
            impact=AmbiguityImpact.MEDIUM,
        ),
    ],
    possible_traps=["Forgetting to return to depot", "Not checking capacity constraint for each vehicle"],
    modeling_tasks=[ModelingTaskType.ROUTING, ModelingTaskType.OPTIMIZATION, ModelingTaskType.GRAPH_NETWORK],
    confidence=0.9,
)

FIXTURE_C_CANDIDATES = [
    ModelCandidate(
        candidate_id="CAND-C1",
        name="Integer Programming CVRP",
        model_family=ModelFamily.INTEGER_PROGRAMMING,
        applicable_subproblems=["SUB-C1", "SUB-C2"],
        summary="Exact CVRP formulation using integer programming with subtour elimination",
        mathematical_structure="min Σ c_ij x_ij s.t. flow conservation, capacity, subtour elimination",
        components=[
            ModelComponent(
                component_id="MC-C1-1",
                name="CVRP IP Solver",
                model_family=ModelFamily.INTEGER_PROGRAMMING,
                role=ComponentRole.CORE_MODEL,
                inputs=["Distance matrix", "Demands", "Vehicle capacity"],
                outputs=["Optimal routes", "Total distance"],
            ),
        ],
        required_data=["Distance matrix", "Demand per location"],
        required_assumptions=["Symmetric distances", "Single trip per vehicle"],
        strengths=["Provably optimal", "Exact solution"],
        weaknesses=["NP-hard", "May not scale to larger instances"],
        implementation_plan="Gurobi or OR-Tools with MTZ subtour elimination",
        validation_plan="Verify all constraints satisfied, compare with heuristic solutions",
        risk_flags=[],
    ),
    ModelCandidate(
        candidate_id="CAND-C2",
        name="Clarke-Wright Savings Heuristic",
        model_family=ModelFamily.GRAPH_THEORY,
        applicable_subproblems=["SUB-C1", "SUB-C2"],
        summary="Classic savings algorithm for CVRP — fast heuristic solution",
        mathematical_structure="Savings s_ij = c_i0 + c_0j - c_ij, merge routes by decreasing savings",
        components=[
            ModelComponent(
                component_id="MC-C2-1",
                name="Savings Algorithm",
                model_family=ModelFamily.GRAPH_THEORY,
                role=ComponentRole.CORE_MODEL,
                inputs=["Distance matrix", "Demands", "Vehicle capacity"],
                outputs=["Heuristic routes", "Total distance"],
            ),
        ],
        required_data=["Distance matrix", "Demand per location"],
        required_assumptions=["Symmetric distances"],
        strengths=["Fast", "Simple", "Well-known", "Good solutions for small instances"],
        weaknesses=["Not guaranteed optimal", "May miss better solutions"],
        implementation_plan="Custom Python implementation of Clarke-Wright",
        validation_plan="Compare with known optimal for small instances, check capacity constraints",
        risk_flags=[],
    ),
    ModelCandidate(
        candidate_id="CAND-C3",
        name="Network Flow CVRP",
        model_family=ModelFamily.NETWORK_FLOW,
        applicable_subproblems=["SUB-C1", "SUB-C2"],
        summary="Network flow formulation of CVRP with commodity flow constraints",
        mathematical_structure="Flow-based formulation: min Σ c_ij y_ij with commodity flow constraints",
        components=[
            ModelComponent(
                component_id="MC-C3-1",
                name="Network Flow Solver",
                model_family=ModelFamily.NETWORK_FLOW,
                role=ComponentRole.CORE_MODEL,
                inputs=["Distance matrix", "Demands", "Vehicle capacity"],
                outputs=["Optimal routes via flow formulation"],
            ),
        ],
        required_data=["Distance matrix", "Demand per location"],
        required_assumptions=["Symmetric distances", "Flow conservation"],
        strengths=["Alternative exact formulation", "Can be more efficient than IP"],
        weaknesses=["Complex formulation", "Still NP-hard"],
        implementation_plan="NetworkX + Gurobi for flow-based CVRP",
        validation_plan="Verify flow conservation, compare routes with IP formulation",
        risk_flags=["complex_formulation"],
    ),
]