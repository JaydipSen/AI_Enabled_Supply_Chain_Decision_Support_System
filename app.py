import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# ============================================================
# OPTIONAL OPENAI IMPORT
# ============================================================

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI-Enabled Supply Chain Decision Support System",
    page_icon="📦",
    layout="wide",
)


# ============================================================
# TITLE
# ============================================================

st.title("📦 AI-Enabled Supply Chain Decision Support System")

st.markdown(
    """
This decision-support system combines **Monte Carlo simulation,
supplier disruption modeling, inventory analysis, and allocation
optimization** to evaluate supply-chain resilience.

The quantitative analysis is performed entirely in Python.

An OpenAI model can optionally interpret the simulation results and
provide managerial recommendations. If the OpenAI API is unavailable,
the application automatically switches to a **local Python-based
recommendation engine**.
"""
)


# ============================================================
# DATA STRUCTURE
# ============================================================

@dataclass
class SimulationResult:
    total_cost: np.ndarray
    holding_cost: np.ndarray
    shortage_cost: np.ndarray
    emergency_cost: np.ndarray
    shortage_units: np.ndarray
    emergency_units: np.ndarray
    stockout_weeks: np.ndarray
    ending_inventory: np.ndarray


# ============================================================
# DEFAULT SUPPLIER INFORMATION
# ============================================================

DEFAULT_SUPPLIERS = {
    "Supplier A": {
        "unit_cost": 10.0,
        "capacity": 500,
        "lead_time": 1,
        "disruption_probability": 0.12,
        "disruption_duration": 3,
    },
    "Supplier B": {
        "unit_cost": 11.5,
        "capacity": 400,
        "lead_time": 2,
        "disruption_probability": 0.08,
        "disruption_duration": 2,
    },
    "Supplier C": {
        "unit_cost": 14.0,
        "capacity": 250,
        "lead_time": 1,
        "disruption_probability": 0.05,
        "disruption_duration": 2,
    },
}


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("Simulation Parameters")

initial_inventory = st.sidebar.number_input(
    "Initial inventory",
    min_value=0,
    value=800,
    step=50,
)

average_demand = st.sidebar.number_input(
    "Average weekly demand",
    min_value=1,
    value=750,
    step=50,
)

demand_cv = st.sidebar.slider(
    "Demand coefficient of variation",
    min_value=0.01,
    max_value=0.80,
    value=0.15,
    step=0.01,
)

holding_cost_rate = st.sidebar.number_input(
    "Holding cost per unit/week",
    min_value=0.0,
    value=1.50,
    step=0.25,
)

shortage_cost_rate = st.sidebar.number_input(
    "Shortage / lost-sales cost per unit",
    min_value=0.0,
    value=25.0,
    step=1.0,
)

emergency_cost_rate = st.sidebar.number_input(
    "Emergency procurement cost per unit",
    min_value=0.0,
    value=30.0,
    step=1.0,
)

emergency_capacity = st.sidebar.number_input(
    "Emergency procurement capacity/week",
    min_value=0,
    value=150,
    step=10,
)

safety_stock = st.sidebar.number_input(
    "Safety stock",
    min_value=0,
    value=300,
    step=50,
)

horizon = st.sidebar.number_input(
    "Simulation horizon (weeks)",
    min_value=1,
    max_value=104,
    value=12,
    step=1,
)

n_simulations = st.sidebar.number_input(
    "Monte Carlo simulations",
    min_value=100,
    max_value=50000,
    value=5000,
    step=500,
)

seed = st.sidebar.number_input(
    "Random seed",
    min_value=0,
    value=42,
    step=1,
)


# ============================================================
# SUPPLIER CONFIGURATION
# ============================================================

st.sidebar.header("Supplier Configuration")

suppliers = {}

for supplier_name, default in DEFAULT_SUPPLIERS.items():

    st.sidebar.markdown(f"### {supplier_name}")

    unit_cost = st.sidebar.number_input(
        f"{supplier_name} unit cost",
        min_value=0.0,
        value=float(default["unit_cost"]),
        step=0.50,
        key=f"{supplier_name}_cost",
    )

    capacity = st.sidebar.number_input(
        f"{supplier_name} weekly capacity",
        min_value=0,
        value=int(default["capacity"]),
        step=50,
        key=f"{supplier_name}_capacity",
    )

    lead_time = st.sidebar.number_input(
        f"{supplier_name} lead time",
        min_value=0,
        max_value=20,
        value=int(default["lead_time"]),
        step=1,
        key=f"{supplier_name}_lead",
    )

    disruption_probability = st.sidebar.slider(
        f"{supplier_name} disruption probability",
        min_value=0.0,
        max_value=1.0,
        value=float(default["disruption_probability"]),
        step=0.01,
        key=f"{supplier_name}_prob",
    )

    disruption_duration = st.sidebar.number_input(
        f"{supplier_name} disruption duration",
        min_value=1,
        max_value=20,
        value=int(default["disruption_duration"]),
        step=1,
        key=f"{supplier_name}_duration",
    )

    suppliers[supplier_name] = {
        "unit_cost": unit_cost,
        "capacity": capacity,
        "lead_time": lead_time,
        "disruption_probability": disruption_probability,
        "disruption_duration": disruption_duration,
    }


# ============================================================
# AI MODEL SELECTION
# ============================================================

st.sidebar.header("AI Recommendation")

ai_model = st.sidebar.selectbox(
    "OpenAI model",
    [
        "gpt-5.6-luna",
        "gpt-5.6-terra",
        "gpt-5.6-sol",
    ],
)

st.sidebar.caption(
    "OpenAI is optional. If the API is unavailable or credits "
    "are exhausted, the system automatically uses local Python "
    "recommendation logic."
)


# ============================================================
# VALIDATION
# ============================================================

def validate_inputs():

    errors = []

    if average_demand <= 0:
        errors.append("Average demand must be greater than zero.")

    if initial_inventory < 0:
        errors.append("Initial inventory cannot be negative.")

    if safety_stock < 0:
        errors.append("Safety stock cannot be negative.")

    if horizon <= 0:
        errors.append("Simulation horizon must be positive.")

    if n_simulations < 100:
        errors.append("At least 100 simulations are recommended.")

    total_capacity = sum(
        supplier["capacity"]
        for supplier in suppliers.values()
    )

    if total_capacity <= 0:
        errors.append("Total supplier capacity must be greater than zero.")

    return errors


validation_errors = validate_inputs()

if validation_errors:

    for error in validation_errors:
        st.error(error)

    st.stop()


# ============================================================
# DEMAND GENERATION
# ============================================================

def generate_demand(
    rng: np.random.Generator,
    simulations: int,
    weeks: int,
    mean_demand: float,
    cv: float,
) -> np.ndarray:

    if cv <= 0:
        return np.full(
            (simulations, weeks),
            mean_demand,
            dtype=float,
        )

    sigma = np.sqrt(np.log(1 + cv**2))
    mu = np.log(mean_demand) - 0.5 * sigma**2

    demand = rng.lognormal(
        mean=mu,
        sigma=sigma,
        size=(simulations, weeks),
    )

    return np.maximum(
        np.round(demand),
        0,
    )


# ============================================================
# DISRUPTION GENERATION
# ============================================================

def generate_disruptions(
    rng: np.random.Generator,
    simulations: int,
    weeks: int,
    supplier_config: Dict,
) -> np.ndarray:

    probability = supplier_config["disruption_probability"]
    duration = supplier_config["disruption_duration"]

    disruption = np.zeros(
        (simulations, weeks),
        dtype=bool,
    )

    for sim in range(simulations):

        week = 0

        while week < weeks:

            if rng.random() < probability:

                end_week = min(
                    week + duration,
                    weeks,
                )

                disruption[
                    sim,
                    week:end_week,
                ] = True

                week = end_week

            else:
                week += 1

    return disruption


# ============================================================
# ALLOCATION NORMALIZATION
# ============================================================

def normalize_allocation(
    allocation: Dict[str, float]
) -> Dict[str, float]:

    total = sum(allocation.values())

    if total <= 0:
        equal = 1 / len(allocation)
        return {
            supplier: equal
            for supplier in allocation
        }

    return {
        supplier: value / total
        for supplier, value in allocation.items()
    }


# ============================================================
# SUPPLY-CHAIN SIMULATION
# ============================================================

def simulate_policy(
    allocation: Dict[str, float],
    suppliers: Dict,
    demand: np.ndarray,
    initial_inventory: float,
    safety_stock: float,
    holding_cost_rate: float,
    shortage_cost_rate: float,
    emergency_cost_rate: float,
    emergency_capacity: float,
    seed: int = 42,
) -> SimulationResult:

    simulations, weeks = demand.shape

    rng = np.random.default_rng(seed)

    disruptions = {}

    for supplier_name, supplier_config in suppliers.items():

        disruptions[supplier_name] = generate_disruptions(
            rng,
            simulations,
            weeks + 20,
            supplier_config,
        )

    total_cost = np.zeros(simulations)
    holding_cost = np.zeros(simulations)
    shortage_cost = np.zeros(simulations)
    emergency_cost = np.zeros(simulations)

    shortage_units = np.zeros(simulations)
    emergency_units = np.zeros(simulations)
    stockout_weeks = np.zeros(simulations)
    ending_inventory = np.zeros(simulations)

    # --------------------------------------------------------
    # Simulation loop
    # --------------------------------------------------------

    for sim in range(simulations):

        inventory = float(initial_inventory)

        # Pipeline:
        # pipeline[week] = quantity expected to arrive in that week
        pipeline = {}

        sim_holding = 0.0
        sim_shortage = 0.0
        sim_emergency = 0.0

        sim_shortage_units = 0.0
        sim_emergency_units = 0.0
        sim_stockout_weeks = 0

        for week in range(weeks):

            # ------------------------------------------------
            # Receive orders arriving this week
            # ------------------------------------------------

            arrivals = pipeline.pop(week, [])

            for supplier_name, quantity in arrivals:

                supplier_disrupted = disruptions[
                    supplier_name
                ][sim, min(week, disruptions[supplier_name].shape[1] - 1)]

                if not supplier_disrupted:
                    inventory += quantity

            # ------------------------------------------------
            # Demand
            # ------------------------------------------------

            weekly_demand = demand[sim, week]

            available_inventory = inventory

            fulfilled = min(
                available_inventory,
                weekly_demand,
            )

            inventory -= fulfilled

            shortage = max(
                weekly_demand - fulfilled,
                0,
            )

            # ------------------------------------------------
            # Emergency procurement
            # ------------------------------------------------

            emergency_quantity = min(
                shortage,
                emergency_capacity,
            )

            if emergency_quantity > 0:

                inventory += emergency_quantity

                shortage -= emergency_quantity

                sim_emergency += (
                    emergency_quantity
                    * emergency_cost_rate
                )

                sim_emergency_units += emergency_quantity

            # ------------------------------------------------
            # Remaining shortage
            # ------------------------------------------------

            if shortage > 0:

                sim_shortage += (
                    shortage
                    * shortage_cost_rate
                )

                sim_shortage_units += shortage

            # ------------------------------------------------
            # Inventory costs
            # ------------------------------------------------

            sim_holding += (
                max(inventory - safety_stock, 0)
                * holding_cost_rate
            )

            if inventory <= 0:
                sim_stockout_weeks += 1

            # ------------------------------------------------
            # Place supplier orders for future arrival
            # ------------------------------------------------

            for supplier_name, supplier_config in suppliers.items():

                share = allocation.get(
                    supplier_name,
                    0,
                )

                order_quantity = (
                    supplier_config["capacity"]
                    * share
                )

                lead_time = supplier_config[
                    "lead_time"
                ]

                arrival_week = week + lead_time

                if (
                    order_quantity > 0
                    and arrival_week < weeks
                ):

                    pipeline.setdefault(
                        arrival_week,
                        [],
                    ).append(
                        (
                            supplier_name,
                            order_quantity,
                        )
                    )

        # ----------------------------------------------------
        # Final values
        # ----------------------------------------------------

        total_cost[sim] = (
            sim_holding
            + sim_shortage
            + sim_emergency
        )

        holding_cost[sim] = sim_holding
        shortage_cost[sim] = sim_shortage
        emergency_cost[sim] = sim_emergency

        shortage_units[sim] = sim_shortage_units
        emergency_units[sim] = sim_emergency_units
        stockout_weeks[sim] = sim_stockout_weeks
        ending_inventory[sim] = inventory

    return SimulationResult(
        total_cost=total_cost,
        holding_cost=holding_cost,
        shortage_cost=shortage_cost,
        emergency_cost=emergency_cost,
        shortage_units=shortage_units,
        emergency_units=emergency_units,
        stockout_weeks=stockout_weeks,
        ending_inventory=ending_inventory,
    )


# ============================================================
# RESULT SUMMARY
# ============================================================

def summarize_result(
    result: SimulationResult
) -> Dict:

    return {
        "mean_total_cost": float(
            np.mean(result.total_cost)
        ),
        "median_total_cost": float(
            np.median(result.total_cost)
        ),
        "p95_total_cost": float(
            np.percentile(
                result.total_cost,
                95,
            )
        ),
        "mean_holding_cost": float(
            np.mean(result.holding_cost)
        ),
        "mean_shortage_cost": float(
            np.mean(result.shortage_cost)
        ),
        "mean_emergency_cost": float(
            np.mean(result.emergency_cost)
        ),
        "mean_shortage_units": float(
            np.mean(result.shortage_units)
        ),
        "mean_emergency_units": float(
            np.mean(result.emergency_units)
        ),
        "mean_stockout_weeks": float(
            np.mean(result.stockout_weeks)
        ),
        "stockout_probability": float(
            np.mean(result.stockout_weeks > 0)
        ),
        "mean_ending_inventory": float(
            np.mean(result.ending_inventory)
        ),
    }


# ============================================================
# ALLOCATION GRID
# ============================================================

def generate_allocation_grid(
    step: float = 0.10,
) -> List[Dict[str, float]]:

    supplier_names = list(suppliers.keys())

    allocations = []

    values = np.arange(
        0,
        1 + step / 2,
        step,
    )

    for a in values:

        for b in values:

            c = 1 - a - b

            if c < -1e-9:
                continue

            if c > 1 + 1e-9:
                continue

            allocation = {
                supplier_names[0]: round(a, 4),
                supplier_names[1]: round(b, 4),
                supplier_names[2]: round(max(c, 0), 4),
            }

            allocations.append(allocation)

    return allocations


# ============================================================
# OPTIMIZATION
# ============================================================

@st.cache_data
def optimize_allocations(
    suppliers: Dict,
    demand: np.ndarray,
    initial_inventory: float,
    safety_stock: float,
    holding_cost_rate: float,
    shortage_cost_rate: float,
    emergency_cost_rate: float,
    emergency_capacity: float,
    seed: int,
):

    allocation_grid = generate_allocation_grid(
        step=0.10
    )

    records = []

    best_allocation = None
    best_result = None
    best_cost = float("inf")

    for allocation in allocation_grid:

        result = simulate_policy(
            allocation=allocation,
            suppliers=suppliers,
            demand=demand,
            initial_inventory=initial_inventory,
            safety_stock=safety_stock,
            holding_cost_rate=holding_cost_rate,
            shortage_cost_rate=shortage_cost_rate,
            emergency_cost_rate=emergency_cost_rate,
            emergency_capacity=emergency_capacity,
            seed=seed,
        )

        summary = summarize_result(result)

        records.append(
            {
                "Supplier A": allocation["Supplier A"],
                "Supplier B": allocation["Supplier B"],
                "Supplier C": allocation["Supplier C"],
                "Mean Total Cost": summary[
                    "mean_total_cost"
                ],
                "P95 Cost": summary[
                    "p95_total_cost"
                ],
                "Stockout Probability": summary[
                    "stockout_probability"
                ],
                "Emergency Units": summary[
                    "mean_emergency_units"
                ],
                "Shortage Units": summary[
                    "mean_shortage_units"
                ],
            }
        )

        if summary["mean_total_cost"] < best_cost:

            best_cost = summary[
                "mean_total_cost"
            ]

            best_allocation = allocation
            best_result = result

    ranking = pd.DataFrame(records).sort_values(
        "Mean Total Cost"
    )

    return (
        best_allocation,
        best_result,
        ranking,
    )


# ============================================================
# LOCAL PYTHON RECOMMENDATION ENGINE
# ============================================================

def generate_local_recommendation(
    baseline_summary: Dict,
    optimized_summary: Dict,
    optimized_allocation: Dict,
    suppliers: Dict,
) -> str:
    """
    Generate a managerial recommendation without using
    any external AI/API.

    This function deliberately uses only quantitative
    simulation outputs and supplier parameters.
    """

    baseline_cost = baseline_summary["mean_total_cost"]
    optimized_cost = optimized_summary["mean_total_cost"]

    baseline_p95 = baseline_summary["p95_total_cost"]
    optimized_p95 = optimized_summary["p95_total_cost"]

    baseline_stockout = (
        baseline_summary["stockout_probability"]
    )

    optimized_stockout = (
        optimized_summary["stockout_probability"]
    )

    baseline_shortage = (
        baseline_summary["mean_shortage_units"]
    )

    optimized_shortage = (
        optimized_summary["mean_shortage_units"]
    )

    baseline_emergency = (
        baseline_summary["mean_emergency_units"]
    )

    optimized_emergency = (
        optimized_summary["mean_emergency_units"]
    )

    cost_improvement = (
        (baseline_cost - optimized_cost)
        / baseline_cost
        * 100
        if baseline_cost > 0
        else 0
    )

    p95_improvement = (
        (baseline_p95 - optimized_p95)
        / baseline_p95
        * 100
        if baseline_p95 > 0
        else 0
    )

    stockout_improvement = (
        (baseline_stockout - optimized_stockout)
        * 100
    )

    # --------------------------------------------------------
    # Identify dominant supplier
    # --------------------------------------------------------

    dominant_supplier = max(
        optimized_allocation,
        key=optimized_allocation.get,
    )

    dominant_share = optimized_allocation[
        dominant_supplier
    ]

    dominant_config = suppliers[
        dominant_supplier
    ]

    # --------------------------------------------------------
    # Risk diversification
    # --------------------------------------------------------

    active_suppliers = sum(
        share > 0.05
        for share in optimized_allocation.values()
    )

    # --------------------------------------------------------
    # Recommendation text
    # --------------------------------------------------------

    recommendation = []

    recommendation.append(
        "### Local Python-Based Managerial Recommendation"
    )

    recommendation.append(
        "\nThe recommendation below was generated directly "
        "from the Monte Carlo simulation results and does "
        "not require an OpenAI API call."
    )

    # --------------------------------------------------------
    # Cost result
    # --------------------------------------------------------

    if cost_improvement > 0:

        recommendation.append(
            f"\n**1. Cost performance:** "
            f"The optimized allocation reduces expected total "
            f"cost by approximately **{cost_improvement:.1f}%** "
            f"compared with the equal-allocation baseline."
        )

    elif cost_improvement < 0:

        recommendation.append(
            f"\n**1. Cost performance:** "
            f"The optimized allocation increases expected total "
            f"cost by approximately "
            f"**{abs(cost_improvement):.1f}%** compared with "
            f"the equal-allocation baseline. "
            f"The allocation should therefore be reviewed "
            f"before implementation."
        )

    else:

        recommendation.append(
            "\n**1. Cost performance:** "
            "The optimized allocation produces approximately "
            "the same expected cost as the baseline."
        )

    # --------------------------------------------------------
    # Risk
    # --------------------------------------------------------

    if optimized_stockout < baseline_stockout:

        recommendation.append(
            f"\n**2. Service risk:** "
            f"Stockout probability falls from "
            f"**{baseline_stockout:.1%}** to "
            f"**{optimized_stockout:.1%}**."
        )

    elif optimized_stockout > baseline_stockout:

        recommendation.append(
            f"\n**2. Service risk:** "
            f"Stockout probability increases from "
            f"**{baseline_stockout:.1%}** to "
            f"**{optimized_stockout:.1%}**. "
            f"This indicates a potential resilience trade-off."
        )

    else:

        recommendation.append(
            "\n**2. Service risk:** "
            "The optimized allocation produces a similar "
            "stockout probability to the baseline."
        )

    # --------------------------------------------------------
    # Shortage
    # --------------------------------------------------------

    if optimized_shortage < baseline_shortage:

        recommendation.append(
            f"\n**3. Shortage exposure:** "
            f"Expected shortage falls from "
            f"**{baseline_shortage:.1f} units** to "
            f"**{optimized_shortage:.1f} units**."
        )

    elif optimized_shortage > baseline_shortage:

        recommendation.append(
            f"\n**3. Shortage exposure:** "
            f"Expected shortage increases from "
            f"**{baseline_shortage:.1f} units** to "
            f"**{optimized_shortage:.1f} units**."
        )

    # --------------------------------------------------------
    # Emergency procurement
    # --------------------------------------------------------

    if optimized_emergency < baseline_emergency:

        recommendation.append(
            f"\n**4. Emergency procurement:** "
            f"Expected emergency procurement declines from "
            f"**{baseline_emergency:.1f} units** to "
            f"**{optimized_emergency:.1f} units**, "
            f"indicating lower dependence on expensive "
            f"last-minute sourcing."
        )

    elif optimized_emergency > baseline_emergency:

        recommendation.append(
            f"\n**4. Emergency procurement:** "
            f"Expected emergency procurement rises from "
            f"**{baseline_emergency:.1f} units** to "
            f"**{optimized_emergency:.1f} units**. "
            f"Management should assess whether the additional "
            f"resilience justifies this cost."
        )

    # --------------------------------------------------------
    # Supplier recommendation
    # --------------------------------------------------------

    recommendation.append(
        f"\n**5. Supplier allocation:** "
        f"**{dominant_supplier}** receives the largest share "
        f"of the optimized allocation "
        f"(**{dominant_share:.1%}**). "
        f"Its modeled unit cost is "
        f"**{dominant_config['unit_cost']:.2f}**, "
        f"weekly capacity is "
        f"**{dominant_config['capacity']:.0f} units**, "
        f"lead time is "
        f"**{dominant_config['lead_time']} week(s)**, "
        f"and disruption probability is "
        f"**{dominant_config['disruption_probability']:.1%}**."
    )

    # --------------------------------------------------------
    # Diversification
    # --------------------------------------------------------

    if active_suppliers >= 3:

        recommendation.append(
            "\n**6. Resilience:** "
            "The optimized policy maintains meaningful "
            "participation from all three suppliers. "
            "This provides diversification against supplier-specific "
            "disruptions."
        )

    elif active_suppliers == 2:

        recommendation.append(
            "\n**6. Resilience:** "
            "The optimized policy uses two suppliers meaningfully. "
            "This provides some diversification, although "
            "management should monitor concentration risk."
        )

    else:

        recommendation.append(
            "\n**6. Resilience:** "
            "The optimized policy is highly concentrated in one "
            "supplier. While this may improve cost efficiency, "
            "management should consider maintaining a qualified "
            "secondary supplier as a contingency."
        )

    # --------------------------------------------------------
    # Tail-risk statement
    # --------------------------------------------------------

    if p95_improvement > 0:

        recommendation.append(
            f"\n**7. Tail-risk:** "
            f"The 95th-percentile cost decreases by approximately "
            f"**{p95_improvement:.1f}%**, suggesting improved "
            f"performance under adverse scenarios."
        )

    elif p95_improvement < 0:

        recommendation.append(
            f"\n**7. Tail-risk:** "
            f"The 95th-percentile cost increases by approximately "
            f"**{abs(p95_improvement):.1f}%**. "
            f"The optimized policy therefore deserves additional "
            f"stress testing."
        )

    else:

        recommendation.append(
            "\n**7. Tail-risk:** "
            "The 95th-percentile cost is broadly unchanged."
        )

    # --------------------------------------------------------
    # Final managerial recommendation
    # --------------------------------------------------------

    if (
        cost_improvement > 0
        and optimized_stockout <= baseline_stockout
    ):

        recommendation.append(
            "\n### Overall Recommendation\n"
            "The optimized allocation is preferable under the "
            "modeled assumptions because it improves expected "
            "cost performance without worsening the simulated "
            "stockout risk. Management should nevertheless "
            "validate the allocation through additional stress "
            "tests involving higher demand, longer disruptions, "
            "and supplier capacity reductions."
        )

    elif cost_improvement > 0:

        recommendation.append(
            "\n### Overall Recommendation\n"
            "The optimized allocation provides a cost advantage "
            "but introduces a service-risk trade-off. It should "
            "not be implemented solely on cost grounds. "
            "Management should determine an acceptable service "
            "level and consider increasing safety stock or "
            "maintaining additional supplier diversification."
        )

    else:

        recommendation.append(
            "\n### Overall Recommendation\n"
            "The optimized allocation does not provide a clear "
            "cost advantage under the current assumptions. "
            "The baseline allocation should therefore remain "
            "a useful benchmark while management investigates "
            "alternative safety-stock, supplier-capacity, and "
            "disruption assumptions."
        )

    return "\n".join(recommendation)


# ============================================================
# OPENAI API KEY HANDLING
# ============================================================

def get_openai_api_key():

    # First try Streamlit secrets
    try:

        api_key = st.secrets.get(
            "OPENAI_API_KEY"
        )

        if api_key:

            return str(api_key).strip()

    except Exception:
        pass

    # Then try environment variable
    api_key = os.getenv(
        "OPENAI_API_KEY"
    )

    if api_key:
        return api_key.strip()

    return None


# ============================================================
# OPENAI CLIENT
# ============================================================

def get_openai_client():

    if OpenAI is None:

        return None, (
            "The OpenAI Python package is not installed."
        )

    api_key = get_openai_api_key()

    if not api_key:

        return None, (
            "OPENAI_API_KEY was not found."
        )

    try:

        client = OpenAI(
            api_key=api_key
        )

        return client, None

    except Exception as error:

        return None, (
            f"Unable to initialize OpenAI client: {error}"
        )


# ============================================================
# OPENAI PROMPT
# ============================================================

def build_ai_prompt(
    baseline_summary: Dict,
    optimized_summary: Dict,
    optimized_allocation: Dict,
    suppliers: Dict,
) -> str:

    return f"""
You are a careful supply-chain risk consultant.

Interpret the following Monte Carlo simulation results.

Do not invent numerical facts.
Do not recalculate the simulation.
Clearly distinguish quantitative evidence from managerial recommendations.

BASELINE RESULTS
----------------
Mean total cost:
{baseline_summary['mean_total_cost']:.2f}

95th percentile cost:
{baseline_summary['p95_total_cost']:.2f}

Stockout probability:
{baseline_summary['stockout_probability']:.2%}

Mean shortage units:
{baseline_summary['mean_shortage_units']:.2f}

Mean emergency procurement:
{baseline_summary['mean_emergency_units']:.2f}

Mean ending inventory:
{baseline_summary['mean_ending_inventory']:.2f}


OPTIMIZED RESULTS
-----------------
Mean total cost:
{optimized_summary['mean_total_cost']:.2f}

95th percentile cost:
{optimized_summary['p95_total_cost']:.2f}

Stockout probability:
{optimized_summary['stockout_probability']:.2%}

Mean shortage units:
{optimized_summary['mean_shortage_units']:.2f}

Mean emergency procurement:
{optimized_summary['mean_emergency_units']:.2f}

Mean ending inventory:
{optimized_summary['mean_ending_inventory']:.2f}


OPTIMIZED ALLOCATION
--------------------
{optimized_allocation}


SUPPLIER PARAMETERS
-------------------
{suppliers}


Provide a concise managerial recommendation covering:

1. Cost performance
2. Supply disruption risk
3. Stockout/service risk
4. Emergency procurement
5. Supplier diversification
6. Tail-risk / 95th percentile cost
7. Practical implementation recommendation

Use headings and bullet points where useful.
"""


# ============================================================
# OPENAI RECOMMENDATION
# ============================================================

def generate_openai_recommendation(
    client,
    model,
    prompt,
):

    try:

        response = client.responses.create(
            model=model,
            instructions=(
                "You are a careful supply-chain risk consultant. "
                "Interpret quantitative simulation results without "
                "inventing numerical facts. Clearly distinguish "
                "simulation evidence from managerial recommendations."
            ),
            input=prompt,
        )

        text = getattr(
            response,
            "output_text",
            None,
        )

        if text:

            return text, None

        return (
            None,
            "The OpenAI API returned no recommendation text.",
        )

    except Exception as error:

        return None, str(error)


# ============================================================
# MAIN SIMULATION
# ============================================================

st.header("1. Simulation")

if st.button(
    "🚀 Run Monte Carlo Simulation",
    type="primary",
    use_container_width=True,
):

    rng = np.random.default_rng(
        int(seed)
    )

    demand = generate_demand(
        rng=rng,
        simulations=int(n_simulations),
        weeks=int(horizon),
        mean_demand=float(average_demand),
        cv=float(demand_cv),
    )

    # --------------------------------------------------------
    # Baseline allocation
    # --------------------------------------------------------

    baseline_allocation = {
        supplier_name: 1 / len(suppliers)
        for supplier_name in suppliers
    }

    baseline_result = simulate_policy(
        allocation=baseline_allocation,
        suppliers=suppliers,
        demand=demand,
        initial_inventory=initial_inventory,
        safety_stock=safety_stock,
        holding_cost_rate=holding_cost_rate,
        shortage_cost_rate=shortage_cost_rate,
        emergency_cost_rate=emergency_cost_rate,
        emergency_capacity=emergency_capacity,
        seed=int(seed),
    )

    baseline_summary = summarize_result(
        baseline_result
    )

    # --------------------------------------------------------
    # Optimization
    # --------------------------------------------------------

    (
        optimized_allocation,
        optimized_result,
        ranking,
    ) = optimize_allocations(
        suppliers=suppliers,
        demand=demand,
        initial_inventory=initial_inventory,
        safety_stock=safety_stock,
        holding_cost_rate=holding_cost_rate,
        shortage_cost_rate=shortage_cost_rate,
        emergency_cost_rate=emergency_cost_rate,
        emergency_capacity=emergency_capacity,
        seed=int(seed),
    )

    optimized_summary = summarize_result(
        optimized_result
    )

    # --------------------------------------------------------
    # Store results in session state
    # --------------------------------------------------------

    st.session_state["demand"] = demand
    st.session_state["baseline_result"] = baseline_result
    st.session_state["optimized_result"] = optimized_result
    st.session_state["baseline_summary"] = baseline_summary
    st.session_state["optimized_summary"] = optimized_summary
    st.session_state["baseline_allocation"] = baseline_allocation
    st.session_state["optimized_allocation"] = optimized_allocation
    st.session_state["ranking"] = ranking

    st.success(
        "Monte Carlo simulation and allocation optimization completed."
    )


# ============================================================
# DISPLAY RESULTS IF AVAILABLE
# ============================================================

if "optimized_summary" in st.session_state:

    baseline_summary = st.session_state[
        "baseline_summary"
    ]

    optimized_summary = st.session_state[
        "optimized_summary"
    ]

    baseline_result = st.session_state[
        "baseline_result"
    ]

    optimized_result = st.session_state[
        "optimized_result"
    ]

    baseline_allocation = st.session_state[
        "baseline_allocation"
    ]

    optimized_allocation = st.session_state[
        "optimized_allocation"
    ]

    ranking = st.session_state[
        "ranking"
    ]

    # ========================================================
    # EXECUTIVE KPI DASHBOARD
    # ========================================================

    st.header("2. Executive Summary")

    cost_reduction = (
        baseline_summary["mean_total_cost"]
        - optimized_summary["mean_total_cost"]
    )

    cost_reduction_pct = (
        cost_reduction
        / baseline_summary["mean_total_cost"]
        * 100
        if baseline_summary["mean_total_cost"] > 0
        else 0
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Baseline Cost",
            f"${baseline_summary['mean_total_cost']:,.0f}",
        )

    with col2:

        st.metric(
            "Optimized Cost",
            f"${optimized_summary['mean_total_cost']:,.0f}",
            delta=f"{cost_reduction_pct:.1f}%",
        )

    with col3:

        st.metric(
            "Baseline Stockout Risk",
            f"{baseline_summary['stockout_probability']:.1%}",
        )

    with col4:

        st.metric(
            "Optimized Stockout Risk",
            f"{optimized_summary['stockout_probability']:.1%}",
        )

    # ========================================================
    # SUPPLIER ALLOCATION
    # ========================================================

    st.header("3. Recommended Supplier Allocation")

    allocation_df = pd.DataFrame(
        {
            "Supplier": list(
                optimized_allocation.keys()
            ),
            "Allocation": [
                optimized_allocation[supplier]
                for supplier in optimized_allocation
            ],
            "Unit Cost": [
                suppliers[supplier]["unit_cost"]
                for supplier in optimized_allocation
            ],
            "Capacity": [
                suppliers[supplier]["capacity"]
                for supplier in optimized_allocation
            ],
            "Lead Time": [
                suppliers[supplier]["lead_time"]
                for supplier in optimized_allocation
            ],
            "Disruption Probability": [
                suppliers[supplier][
                    "disruption_probability"
                ]
                for supplier in optimized_allocation
            ],
        }
    )

    allocation_display = allocation_df.copy()

    allocation_display["Allocation"] = (
        allocation_display["Allocation"]
        .map(lambda x: f"{x:.1%}")
    )

    allocation_display["Unit Cost"] = (
        allocation_display["Unit Cost"]
        .map(lambda x: f"${x:.2f}")
    )

    allocation_display["Disruption Probability"] = (
        allocation_display[
            "Disruption Probability"
        ]
        .map(lambda x: f"{x:.1%}")
    )

    st.dataframe(
        allocation_display,
        use_container_width=True,
        hide_index=True,
    )

    # --------------------------------------------------------
    # Allocation chart
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(8, 4)
    )

    ax.bar(
        allocation_df["Supplier"],
        allocation_df["Allocation"],
    )

    ax.set_ylabel(
        "Allocation Share"
    )

    ax.set_title(
        "Optimized Supplier Allocation"
    )

    ax.set_ylim(
        0,
        max(
            1,
            allocation_df["Allocation"].max()
            * 1.20,
        ),
    )

    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(
            lambda y, _: f"{y:.0%}"
        )
    )

    st.pyplot(fig)

    # ========================================================
    # BASELINE VS OPTIMIZED
    # ========================================================

    st.header("4. Baseline vs Optimized Policy")

    comparison_df = pd.DataFrame(
        {
            "Metric": [
                "Mean Total Cost",
                "95th Percentile Cost",
                "Mean Holding Cost",
                "Mean Shortage Cost",
                "Mean Emergency Cost",
                "Mean Shortage Units",
                "Mean Emergency Units",
                "Stockout Probability",
                "Mean Ending Inventory",
            ],
            "Baseline": [
                baseline_summary[
                    "mean_total_cost"
                ],
                baseline_summary[
                    "p95_total_cost"
                ],
                baseline_summary[
                    "mean_holding_cost"
                ],
                baseline_summary[
                    "mean_shortage_cost"
                ],
                baseline_summary[
                    "mean_emergency_cost"
                ],
                baseline_summary[
                    "mean_shortage_units"
                ],
                baseline_summary[
                    "mean_emergency_units"
                ],
                baseline_summary[
                    "stockout_probability"
                ],
                baseline_summary[
                    "mean_ending_inventory"
                ],
            ],
            "Optimized": [
                optimized_summary[
                    "mean_total_cost"
                ],
                optimized_summary[
                    "p95_total_cost"
                ],
                optimized_summary[
                    "mean_holding_cost"
                ],
                optimized_summary[
                    "mean_shortage_cost"
                ],
                optimized_summary[
                    "mean_emergency_cost"
                ],
                optimized_summary[
                    "mean_shortage_units"
                ],
                optimized_summary[
                    "mean_emergency_units"
                ],
                optimized_summary[
                    "stockout_probability"
                ],
                optimized_summary[
                    "mean_ending_inventory"
                ],
            ],
        }
    )

    st.dataframe(
        comparison_df,
        use_container_width=True,
        hide_index=True,
    )

    # ========================================================
    # COST DISTRIBUTION
    # ========================================================

    st.header("5. Cost Distribution")

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    ax.boxplot(
        [
            baseline_result.total_cost,
            optimized_result.total_cost,
        ],
        labels=[
            "Baseline",
            "Optimized",
        ],
    )

    ax.set_ylabel(
        "Total Cost"
    )

    ax.set_title(
        "Monte Carlo Total Cost Distribution"
    )

    st.pyplot(fig)

    # ========================================================
    # INVENTORY RISK
    # ========================================================

    st.header("6. Inventory Risk Profile")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Mean Shortage",
            f"{optimized_summary['mean_shortage_units']:,.1f}",
        )

    with col2:

        st.metric(
            "Emergency Procurement",
            f"{optimized_summary['mean_emergency_units']:,.1f}",
        )

    with col3:

        st.metric(
            "Ending Inventory",
            f"{optimized_summary['mean_ending_inventory']:,.1f}",
        )

    # ========================================================
    # COST COMPONENTS
    # ========================================================

    st.header("7. Cost Components")

    cost_component_df = pd.DataFrame(
        {
            "Cost Component": [
                "Holding Cost",
                "Shortage Cost",
                "Emergency Procurement",
            ],
            "Baseline": [
                baseline_summary[
                    "mean_holding_cost"
                ],
                baseline_summary[
                    "mean_shortage_cost"
                ],
                baseline_summary[
                    "mean_emergency_cost"
                ],
            ],
            "Optimized": [
                optimized_summary[
                    "mean_holding_cost"
                ],
                optimized_summary[
                    "mean_shortage_cost"
                ],
                optimized_summary[
                    "mean_emergency_cost"
                ],
            ],
        }
    )

    st.dataframe(
        cost_component_df,
        use_container_width=True,
        hide_index=True,
    )

    # ========================================================
    # TOP ALLOCATIONS
    # ========================================================

    st.header("8. Top Allocation Strategies")

    top_rankings = ranking.head(10).copy()

    top_rankings["Supplier A"] = (
        top_rankings["Supplier A"]
        .map(lambda x: f"{x:.0%}")
    )

    top_rankings["Supplier B"] = (
        top_rankings["Supplier B"]
        .map(lambda x: f"{x:.0%}")
    )

    top_rankings["Supplier C"] = (
        top_rankings["Supplier C"]
        .map(lambda x: f"{x:.0%}")
    )

    top_rankings["Mean Total Cost"] = (
        top_rankings["Mean Total Cost"]
        .map(lambda x: f"${x:,.0f}")
    )

    top_rankings["P95 Cost"] = (
        top_rankings["P95 Cost"]
        .map(lambda x: f"${x:,.0f}")
    )

    top_rankings["Stockout Probability"] = (
        top_rankings[
            "Stockout Probability"
        ]
        .map(lambda x: f"{x:.1%}")
    )

    top_rankings["Emergency Units"] = (
        top_rankings[
            "Emergency Units"
        ]
        .map(lambda x: f"{x:.1f}")
    )

    top_rankings["Shortage Units"] = (
        top_rankings[
            "Shortage Units"
        ]
        .map(lambda x: f"{x:.1f}")
    )

    st.dataframe(
        top_rankings,
        use_container_width=True,
        hide_index=True,
    )

    # ========================================================
    # AI / LOCAL RECOMMENDATION
    # ========================================================

    st.header("9. Managerial Recommendation")

    st.markdown(
        """
The system first attempts to obtain an OpenAI-generated
interpretation. If the API cannot be used, it automatically
falls back to the local Python recommendation engine.
"""
    )

    if st.button(
        "🤖 Generate Managerial Recommendation",
        type="primary",
        use_container_width=True,
    ):

        # ----------------------------------------------------
        # Attempt OpenAI
        # ----------------------------------------------------

        client, client_error = (
            get_openai_client()
        )

        if client is not None:

            prompt = build_ai_prompt(
                baseline_summary=baseline_summary,
                optimized_summary=optimized_summary,
                optimized_allocation=optimized_allocation,
                suppliers=suppliers,
            )

            with st.spinner(
                "Generating AI recommendation..."
            ):

                ai_text, api_error = (
                    generate_openai_recommendation(
                        client=client,
                        model=ai_model,
                        prompt=prompt,
                    )
                )

            if ai_text:

                st.success(
                    f"Recommendation generated using OpenAI "
                    f"({ai_model})."
                )

                st.markdown(
                    ai_text
                )

            else:

                # ------------------------------------------------
                # OPENAI FAILED -> LOCAL FALLBACK
                # ------------------------------------------------

                st.warning(
                    "OpenAI could not generate the recommendation. "
                    "The system has automatically switched to the "
                    "local Python-based recommendation engine."
                )

                with st.expander(
                    "Why was local fallback activated?"
                ):

                    st.caption(
                        api_error
                        or "Unknown OpenAI error."
                    )

                local_recommendation = (
                    generate_local_recommendation(
                        baseline_summary=baseline_summary,
                        optimized_summary=optimized_summary,
                        optimized_allocation=optimized_allocation,
                        suppliers=suppliers,
                    )
                )

                st.markdown(
                    local_recommendation
                )

        else:

            # ----------------------------------------------------
            # NO OPENAI CLIENT -> LOCAL FALLBACK
            # ----------------------------------------------------

            st.info(
                "OpenAI is not currently available. "
                "Using the local Python-based recommendation engine."
            )

            if client_error:

                with st.expander(
                    "OpenAI availability details"
                ):

                    st.caption(
                        client_error
                    )

            local_recommendation = (
                generate_local_recommendation(
                    baseline_summary=baseline_summary,
                    optimized_summary=optimized_summary,
                    optimized_allocation=optimized_allocation,
                    suppliers=suppliers,
                )
            )

            st.markdown(
                local_recommendation
            )

    # ========================================================
    # METHODOLOGY
    # ========================================================

    st.header("10. Methodology")

    with st.expander(
        "View simulation methodology"
    ):

        st.markdown(
            """
### Demand Modeling

Weekly demand is generated using a lognormal distribution
parameterized by the specified mean demand and coefficient
of variation.

### Supplier Disruption Modeling

Each supplier has an independent disruption probability and
disruption duration.

### Lead-Time Modeling

Supplier orders enter a pipeline and become available only
after the supplier's specified lead time.

### Inventory Modeling

The simulation tracks:

- Initial inventory
- Supplier receipts
- Weekly demand
- Safety stock
- Holding inventory
- Stockouts
- Emergency procurement
- Lost sales / shortage

### Cost Modeling

Total cost consists of:

**Total Cost = Holding Cost + Shortage Cost + Emergency Procurement Cost**

### Monte Carlo Simulation

Multiple independent scenarios are simulated to estimate:

- Expected cost
- Tail cost
- Stockout probability
- Expected shortages
- Emergency procurement
- Ending inventory

### Allocation Optimization

The optimizer evaluates supplier allocation combinations
using a 10% allocation grid.

The resulting allocation is therefore a **transparent
heuristic grid-search solution**, not a proof of global
stochastic optimality.

### AI Layer

The numerical simulation and optimization are performed by
Python.

OpenAI is used only as an optional interpretation layer.

If the OpenAI API is unavailable, the system uses a local
Python-based recommendation engine.
"""
        )

    # ========================================================
    # DOWNLOAD RESULTS
    # ========================================================

    st.header("11. Export Results")

    export_df = pd.DataFrame(
        {
            "Metric": [
                "Mean Total Cost",
                "95th Percentile Cost",
                "Stockout Probability",
                "Mean Shortage Units",
                "Mean Emergency Units",
                "Mean Ending Inventory",
            ],
            "Baseline": [
                baseline_summary[
                    "mean_total_cost"
                ],
                baseline_summary[
                    "p95_total_cost"
                ],
                baseline_summary[
                    "stockout_probability"
                ],
                baseline_summary[
                    "mean_shortage_units"
                ],
                baseline_summary[
                    "mean_emergency_units"
                ],
                baseline_summary[
                    "mean_ending_inventory"
                ],
            ],
            "Optimized": [
                optimized_summary[
                    "mean_total_cost"
                ],
                optimized_summary[
                    "p95_total_cost"
                ],
                optimized_summary[
                    "stockout_probability"
                ],
                optimized_summary[
                    "mean_shortage_units"
                ],
                optimized_summary[
                    "mean_emergency_units"
                ],
                optimized_summary[
                    "mean_ending_inventory"
                ],
            ],
        }
    )

    csv_data = export_df.to_csv(
        index=False
    )

    st.download_button(
        label="⬇️ Download Simulation Summary CSV",
        data=csv_data,
        file_name="supply_chain_simulation_summary.csv",
        mime="text/csv",
        use_container_width=True,
    )

    allocation_csv = allocation_df.to_csv(
        index=False
    )

    st.download_button(
        label="⬇️ Download Optimized Allocation CSV",
        data=allocation_csv,
        file_name="optimized_supplier_allocation.csv",
        mime="text/csv",
        use_container_width=True,
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Developed by Jaydip Sen | "
    "AI-Enabled Supply Chain Decision Support System"
)

st.caption(
    "© 2026 Jaydip Sen. All rights reserved. | "
    "For academic, educational, and research purposes."
)