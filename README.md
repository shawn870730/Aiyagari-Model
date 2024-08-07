# Aiyagari-Bewley-Huggett-Imrohoglu Model Simulation

This repository contains two implementations of the Aiyagari-Bewley-Huggett-Imrohoglu model, a classic economic model used to analyze household decision-making under income uncertainty. The model simulates the consumption and savings behavior of agents over a finite time horizon.

## Files

1. **analytical_solution.py**: This script solves the model using an analytical approach. It derives the consumption path using the Euler equation and backward induction. The solution is derived mathematically, and the results illustrate how agents optimize their utility under different asset constraints.

2. **numerical_solution.py**: This script solves the model using a numerical optimization method. It employs the Sequential Least Squares Quadratic Programming (SLSQP) algorithm from SciPy to maximize the agent's utility while satisfying budget constraints and asset limits. This approach provides a flexible way to handle various constraints and scenarios.

## Objective

The goal is to solve and simulate the finite horizon model using two different approaches to analyze:

- The mean and variance of income, consumption, and assets over time.
- The life cycle evolution of these economic variables.
- The impact of varying minimum asset levels on household behavior.

## Model Details

- **Objective Function**: Maximizes the expected utility of consumption over the horizon.
- **Budget Constraint**: Ensures that consumption plus next period's assets equals current assets and income.
- **Autoregressive Income Process**: Income follows a log-normal autoregressive process.

## Parameters

- Interest rate: $\( r = 0.05 \)$
- Discount factor: \( \beta = 0.95 \)
- Time horizon: \( T = 40 \)
- Minimum asset levels: \( a \in \{-40, -10, 0\} \)
- Autoregressive coefficient: \( \rho = 0.9 \)
- Income shock standard deviation: \( \sigma = 0.1 \)
- Initial income: \( y_1 = 1 \)
- Initial assets: \( a_1 = 0 \)
- Terminal asset condition: \( a_{T+1} = 0 \)

## Usage

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/aiyagari-model-simulation.git
   cd aiyagari-model-simulation
   ```

2. **Run the analytical solution**:
   ```bash
   python analytical_solution.py
   ```

3. **Run the numerical solution**:
   ```bash
   python numerical_solution.py
   ```

## Results

- **Analytical Solution**: Provides closed-form solutions for optimal consumption paths and demonstrates the effect of constraints on agent behavior.
- **Numerical Solution**: Uses simulation to show dynamic behavior under different economic conditions, offering insights into income distribution and consumption smoothing.

## Future Enhancements

- Explore endogenous labor supply decisions to capture work-leisure trade-offs.
- Analyze the impact of income inequality and redistributive policies.
- Integrate CVXPY for enhanced optimization and visualization capabilities.
