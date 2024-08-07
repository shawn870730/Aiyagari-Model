# Aiyagari-Bewley-Huggett-Imrohoglu Model Simulation

This repository contains implementations of the Aiyagari-Bewley-Huggett-Imrohoglu (ABHI) model, a cornerstone in economic modeling for analyzing household decision-making under income uncertainty. The project includes both analytical and numerical solutions, accompanied by a detailed report.

## Files

1. **ABHI_model_analytical.py**: This script solves the ABHI model using an analytical approach. It derives the optimal consumption path through mathematical solutions and backward induction, demonstrating how agents optimize utility under different constraints.

2. **ABHI_model_numerical.py**: This script employs a numerical optimization approach to solve the model. Utilizing the Sequential Least Squares Quadratic Programming (SLSQP) algorithm from SciPy, it maximizes agent utility while adhering to budget and asset constraints.

3. **ABHI_model_report.pdf**: This document provides an in-depth explanation of the model, methodology, and results. It includes mathematical derivations, simulation insights, and potential improvements for the model.

## Objective

The project's goal is to solve and simulate the ABHI model using both analytical and numerical methods to explore:

- The mean and variance of income, consumption, and assets over time.
- The life cycle evolution of these economic variables.
- The impact of different minimum asset levels on household decision-making.

## Model Details

- **Objective Function**: Maximize the expected utility of consumption over a finite time horizon.
- **Budget Constraint**: Ensure consumption plus next period's assets equals current assets and income.
- **Autoregressive Income Process**: Income follows a log-normal autoregressive process.

## Parameters

- Interest rate: $\( r = 0.05 \)$
- Discount factor: $\( \beta = 0.95 \)$
- Time horizon: $\( T = 40 \)$
- Minimum asset levels: $\( a \in \{-40, -10, 0\} \)$
- Autoregressive coefficient: $\( \rho = 0.9 \)$
- Income shock standard deviation: $\( \sigma = 0.1 \)$
- Initial income: $\( y_1 = 1 \)$
- Initial assets: $\( a_1 = 0 \)$
- Terminal asset condition: $\( a_{T+1} = 0 \)$

## Usage

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/aiyagari-model-simulation.git
   cd aiyagari-model-simulation
   ```

2. **Run the analytical solution**:
   ```bash
   python ABHI_model_analytical.py
   ```

3. **Run the numerical solution**:
   ```bash
   python ABHI_model_numerical.py
   ```

## Results

- **Analytical Solution**: Provides closed-form solutions for consumption paths and illustrates the effect of constraints on agent behavior.
- **Numerical Solution**: Demonstrates dynamic behavior under various economic conditions, offering insights into income distribution and consumption smoothing.

## Documentation

For a comprehensive overview of the model, methodology, and results, refer to the [ABHI_model_report.pdf](./ABHI_model_report.pdf) file.

## Future Enhancements

- Introduce endogenous labor supply decisions to capture work-leisure trade-offs.
- Analyze the impact of income inequality and redistributive policies.
- Integrate CVXPY for enhanced optimization and visualization capabilities.

