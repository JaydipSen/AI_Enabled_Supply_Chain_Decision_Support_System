# AI-Enabled Supply Chain Decision Support System

An interactive decision-support application for evaluating **supplier disruption risk, inventory performance, procurement decisions, and supply chain resilience** using Monte Carlo simulation, optimization, and AI-assisted managerial interpretation.

The application combines quantitative simulation with an optional Generative AI layer. Python performs the numerical simulation and optimization, while the AI component interprets the results and translates them into practical managerial recommendations.

---

## Project Overview

Modern supply chains are exposed to supplier disruptions, demand variability, capacity constraints, lead-time uncertainty, inventory shortages, and emergency procurement requirements.

This project provides an interactive framework to evaluate these risks and compare alternative supplier allocation strategies.

The system allows users to:

* Configure multiple suppliers
* Model supplier disruption probabilities
* Incorporate disruption duration
* Model supplier-specific lead times
* Simulate variable demand
* Define initial inventory and safety stock
* Model shortage and lost-sales costs
* Include emergency procurement
* Compare baseline and optimized supplier allocations
* Perform Monte Carlo simulations
* Analyze cost and service-risk distributions
* Evaluate tail-risk using percentile measures
* Generate managerial recommendations using AI
* Automatically fall back to a Python-based recommendation engine when the OpenAI API is unavailable

---

## System Architecture

The application follows a hybrid quantitative and AI-assisted architecture:

```text
                    User Inputs
                         │
                         ▼
              Supply Chain Parameters
                         │
                         ▼
             Monte Carlo Simulation
                         │
             ┌───────────┴───────────┐
             │                       │
             ▼                       ▼
       Baseline Strategy       Optimization Search
             │                       │
             └───────────┬───────────┘
                         ▼
                Simulation Results
                         │
             ┌───────────┴───────────┐
             │                       │
             ▼                       ▼
       Statistical Analysis     Risk Metrics
             │                       │
             └───────────┬───────────┘
                         ▼
              Managerial Interpretation
                         │
                  ┌──────┴──────┐
                  │             │
                  ▼             ▼
            OpenAI API      Local Python
            Recommendation   Fallback
```

---

## Methodology

### 1. Supplier Modeling

The model supports multiple suppliers with supplier-specific characteristics:

* Unit procurement cost
* Maximum capacity
* Lead time
* Probability of disruption
* Disruption duration

The default configuration contains three suppliers with different cost, capacity, lead-time, and disruption characteristics.

---

### 2. Demand Variability

Demand is modeled as a stochastic variable rather than a fixed quantity.

Users can specify:

* Average demand
* Demand coefficient of variation
* Planning horizon

This allows the system to evaluate supply chain performance under different demand conditions.

---

### 3. Supplier Disruption

Supplier disruptions are modeled probabilistically using the specified disruption probability and duration.

During simulation, disruption events can affect the availability of supplier shipments and consequently influence:

* Inventory levels
* Shortages
* Emergency procurement
* Total cost
* Service performance

---

### 4. Inventory Dynamics

The model incorporates:

* Initial inventory
* Safety stock
* Supplier lead times
* Incoming orders
* Demand consumption
* Shortage/lost-sales quantities
* Emergency procurement

Supplier lead times are explicitly represented through the order pipeline.

---

### 5. Cost Components

The total supply chain cost includes several components:

```text
Total Cost
   =
Procurement Cost
+ Holding Cost
+ Shortage Cost
+ Emergency Procurement Cost
```

This allows users to evaluate both average performance and unfavorable tail outcomes.

---

## Monte Carlo Simulation

The application uses Monte Carlo simulation to evaluate supply chain performance under repeated stochastic scenarios.

For each simulation run, the model generates different realizations of:

* Demand
* Supplier disruption events
* Inventory movements
* Shipment arrivals
* Shortages
* Emergency procurement requirements

The simulation produces a distribution of possible supply chain outcomes rather than relying on a single deterministic scenario.

Key outputs include:

* Expected total cost
* Cost variability
* P95 total cost
* Stockout probability
* Expected shortage
* Emergency procurement
* Supplier utilization
* Risk exposure

---

## Supplier Allocation Optimization

The system compares a baseline allocation strategy with an optimized allocation strategy.

The optimization searches across feasible supplier allocation combinations and evaluates each candidate using the Monte Carlo simulation framework.

The current implementation uses a **10% allocation grid** to provide a transparent and computationally practical allocation search.

### Important modeling note

The optimization should be interpreted as a **simulation-based heuristic allocation search**, rather than a proof of global stochastic optimality.

Future versions may incorporate more advanced approaches such as:

* Stochastic programming
* Mixed-integer optimization
* Bayesian optimization
* Genetic algorithms
* Reinforcement learning
* Distributionally robust optimization

---

## AI-Assisted Managerial Recommendation

The application includes an optional OpenAI-powered interpretation layer.

The AI does **not** perform the Monte Carlo simulation or numerical optimization.

Instead:

```text
Python
   ↓
Simulation
   ↓
Optimization
   ↓
Quantitative Results
   ↓
AI Interpretation
   ↓
Managerial Recommendation
```

The AI receives the quantitative results generated by Python and provides recommendations regarding:

* Cost efficiency
* Service risk
* Supplier diversification
* Shortage exposure
* Emergency procurement
* Resilience
* Tail-risk management

This separation helps ensure that numerical results remain reproducible and transparent.

---

## Local Recommendation Fallback

The application does not depend completely on the OpenAI API.

If the OpenAI API is unavailable because of:

* Missing API key
* Insufficient API credits
* Invalid API key
* Network failure
* API service error
* OpenAI package unavailable

the application automatically generates a **Python-based managerial recommendation** from the simulation results.

Therefore, the core simulation and decision-support functionality remains available even without an active OpenAI API connection.

---

## Baseline vs. Optimized Strategy

The application enables users to compare:

### Baseline Strategy

A predefined supplier allocation, such as equal allocation across suppliers.

### Optimized Strategy

An allocation identified through the simulation-based allocation search.

The comparison can help assess:

* Cost reduction
* Service-level improvement
* Stockout risk
* Shortage reduction
* Emergency procurement
* Resilience improvement
* Tail-risk reduction

---

## Application Interface

The Streamlit interface provides interactive controls for configuring:

### Supply Chain Parameters

* Initial inventory
* Average demand
* Demand variability
* Holding cost
* Shortage cost
* Emergency procurement cost
* Emergency procurement capacity
* Safety stock
* Planning horizon
* Number of simulations
* Random seed

### Supplier Parameters

For each supplier:

* Unit cost
* Capacity
* Lead time
* Disruption probability
* Disruption duration

### AI Parameters

Users can select the desired OpenAI model when API access is available.

---

## Technologies Used

* **Python**
* **Streamlit**
* **NumPy**
* **Pandas**
* **Matplotlib**
* **OpenAI API**
* **Monte Carlo Simulation**
* **Stochastic Modeling**
* **Simulation-Based Optimization**

---

## Project Structure

```text
AI_Enabled_Supply_Chain_Decision_Support_System/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
└── .streamlit/
    └── secrets.toml
```

> `secrets.toml` should remain local and must never be committed to GitHub.

---

## Local Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/AI_Enabled_Supply_Chain_Decision_Support_System.git
```

Navigate to the project directory:

```bash
cd AI_Enabled_Supply_Chain_Decision_Support_System
```

Install the required packages:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

The application will open in your browser.

---

## OpenAI API Configuration

OpenAI API access is optional because the application includes a local Python fallback.

If you want to enable AI-generated managerial recommendations, create:

```text
.streamlit/secrets.toml
```

and add:

```toml
OPENAI_API_KEY = "YOUR_API_KEY"
```

**Never commit this file to GitHub.**

The `.gitignore` file should contain:

```text
.streamlit/secrets.toml
```

---

## ☁️ Streamlit Community Cloud Deployment

The application can be deployed using Streamlit Community Cloud.

General deployment process:

1. Push the project to GitHub.
2. Sign in to Streamlit Community Cloud.
3. Connect your GitHub account.
4. Select the repository.
5. Select the `main` branch.
6. Select `app.py` as the application entry point.
7. Configure the OpenAI API key through Streamlit's Secrets settings if AI recommendations are required.
8. Deploy the application.

The OpenAI API key should **not** be stored directly in the GitHub repository.

---

## Potential Applications

The framework can support decision-making in areas such as:

* Supplier risk management
* Procurement planning
* Inventory management
* Supply chain resilience
* Disruption analysis
* Sourcing strategy
* Emergency procurement planning
* Scenario analysis
* Operations management education
* Supply chain research

---

## Future Enhancements

Potential extensions include:

* Multi-echelon inventory optimization
* Multi-product supply chains
* Multi-period procurement decisions
* Correlated supplier disruptions
* Geographic disruption modeling
* Transportation disruption
* Supplier reliability learning
* Dynamic safety-stock optimization
* Risk-adjusted objective functions
* CVaR-based optimization
* Advanced stochastic optimization
* Reinforcement-learning-based procurement
* Interactive scenario comparison
* Historical demand calibration
* Real-time external data integration

---

## Model Limitations

The current version is intended primarily for **academic, educational, and research purposes**.

The simulation results depend on the assumptions and parameter values supplied by the user.

Important considerations include:

* Disruption probabilities are assumed inputs.
* Demand is generated according to the specified stochastic assumptions.
* The allocation optimizer uses a discrete 10% allocation grid.
* The optimization is a simulation-based heuristic rather than a guarantee of global optimality.
* Real-world supply chains may involve dependencies and correlations not represented in the current model.
* AI-generated recommendations should be interpreted alongside the underlying quantitative results.

---

## Research and Educational Use

This application can be used as a practical demonstration of concepts including:

* Monte Carlo simulation
* Stochastic supply chain modeling
* Inventory management
* Supplier risk analysis
* Operations management
* Procurement optimization
* Supply chain resilience
* Decision-support systems
* Generative AI-assisted analytics

It can also serve as a foundation for further academic research into AI-enabled supply chain decision-making.

---

## Author

**Jaydip Sen**

AI-Enabled Supply Chain Decision Support System

Developed for academic, educational, and research applications.

---

## License

No open-source license is currently specified for this repository.

All rights reserved unless otherwise stated.

---

## Acknowledgment

This project demonstrates the integration of **quantitative simulation, optimization, and Generative AI** to support data-driven supply chain decision-making.

The central principle of the system is:

> **Simulate quantitatively. Optimize transparently. Interpret intelligently.**
