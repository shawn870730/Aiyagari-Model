import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import time

##########################  numerical  ############################

def solve_income(y1, T, rho, sigma): #income_path
    yt = np.empty(T)
    yt[0] = y1
    epsilon_t = np.random.normal(0, sigma, T - 1)
    for t in range(T-1):
        yt[t+1] = yt[t]**rho * np.exp(epsilon_t[t])
    return yt

def utility(ct, beta, T):
    return -np.sum([beta ** (t + 1) * np.log(ct[t]) for t in range(T)])

def budget_constraint(ct, yt, r, T):
    at = np.zeros(T + 1)
    for t in range(T):
        at[t + 1] = (1 + r) * at[t] + yt[t] - ct[t]
    return at[-1]  # Ensure terminal condition a_T = 0

def non_negative(ct): #constraint for positive consumption
    return ct

def non_negative_assets(ct, yt, r, a_min, T): #constraint for sufficient assets
    at = np.zeros(T + 1)
    for t in range(T):
        at[t + 1] = (1 + r) * at[t] + yt[t] - ct[t]
    return at[1:] - a_min

def run_simulation(y1, r, beta, T, rho, sigma, a_min, c0):
    yt = solve_income(y1, T, rho, sigma)
    constraints = [
        {'type': 'eq', 'fun': lambda ct: budget_constraint(ct, yt, r, T)}, # Budget constraint
        {'type': 'ineq', 'fun': lambda ct: non_negative(ct)}, # ct > 0
        {'type': 'ineq', 'fun': lambda ct: non_negative_assets(ct, yt, r, a_min, T)} # at > a_min
    ]

    result = minimize(utility, c0, args=(beta, T), method='SLSQP', constraints=constraints,
                      bounds=[(0, None) for _ in range(T)], options={'maxiter': 100, 'ftol': 1e-6})

    if result.success:
        optimal_c = result.x
        optimal_a = np.zeros(T + 1)
        for t in range(T):
            optimal_a[t + 1] = (1 + r) * optimal_a[t] + yt[t] - optimal_c[t]
        return yt, optimal_c, optimal_a
    else:
        return None

# Parameters
r = 0.05
beta = 0.95
T = 40
rho = 0.9
sigma = 0.1
y1 = 1
a_min = -10

# Initial guess
c0 = np.ones(T)

n_simulations = 100000

start_time = time.time()

yt_simulations = np.zeros((n_simulations, T))

for i in range(n_simulations):
    yt_simulations[i, :] = solve_income(y1, T, rho, sigma)

mean_yt = np.mean(yt_simulations, axis=0)
var_yt = np.var(yt_simulations, axis=0)

results = Parallel(n_jobs=-1, backend="loky")(
    delayed(run_simulation)(y1, r, beta, T, rho, sigma, a_min, c0) for _ in range(n_simulations))

# Filter out failed optimizations
results = [res for res in results if res is not None]

if results:
    yt_simulations = np.array([res[0] for res in results])
    ct_simulations = np.array([res[1] for res in results])
    at_simulations = np.array([res[2] for res in results])
    print("Utility: ", utility(np.mean(ct_simulations, axis=0), beta, T))

    mean_ct = np.mean(ct_simulations, axis=0)
    var_ct = np.var(ct_simulations, axis=0)
    mean_at = np.mean(at_simulations, axis=0)
    var_at = np.var(at_simulations, axis=0)

    # Plotting
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(mean_yt, label='Mean of Income')
    plt.fill_between(range(T), mean_yt - var_yt, mean_yt + var_yt, alpha=0.2, label='Variance of Income')
    plt.xlabel('Time')
    plt.ylabel('Income')
    plt.title(f'Income: a_min = {a_min}')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(mean_ct, label='Mean of Consumption')
    plt.fill_between(range(T), mean_ct - var_ct, mean_ct + var_ct, alpha=0.2, label='Variance of Consumption')
    plt.xlabel('Time')
    plt.ylabel('Consumption')
    plt.title(f'Consumption: a_min = {a_min}')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(mean_at, label='Mean of Assets')
    plt.fill_between(range(T + 1), mean_at - var_at, mean_at + var_at, alpha=0.2, label='Variance of Assets')
    plt.xlabel('Time')
    plt.ylabel('Assets')
    plt.title(f'Assets: a_min = {a_min}')
    plt.legend()

    plt.tight_layout()
    plt.show()

end_time = time.time()
runtime = end_time - start_time
print(f"Finished in {runtime:.2f} seconds")
