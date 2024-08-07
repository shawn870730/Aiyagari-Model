import numpy as np
import matplotlib.pyplot as plt
import time

########################### mathematical ###########################
def solve_income(y1, T, rho, sigma): # yt path
    yt = np.empty(T)
    yt[0] = y1
    epsilon_t = np.random.normal(0, sigma, T - 1)
    for t in range(1, T):
        yt[t] = yt[t - 1] ** rho * np.exp(epsilon_t[t - 1])
    return yt

def ct_star(beta, r, yt, t, T): # ct_star path
    numerator = beta ** (t - 1) * (1 - beta)
    denominator = (1 - beta ** T)
    sum_term = sum((1 + r) ** (t - s) * yt[s - 1] for s in range(1, T + 1))
    return numerator / denominator * sum_term

def decision_process(beta, r, yt, T, a_min):
    at = np.zeros(T)
    ct_star_values = np.zeros(T)

    for t in range(1, T):
        ct_star_values[t - 1] = ct_star(beta, r, yt, t, T)

        at[t] = (1 + r) * at[t - 1] + yt[t - 1] - ct_star_values[t - 1]
        if at[t] < a_min:
            at[t] = a_min
            ct_star_values[t - 1] = (1 + r) * at[t - 1] + yt[t - 1] - a_min

    ct_star_values[T - 1] = (1 + r) * at[T - 1] + yt[T - 1]
    return at, ct_star_values

def utility(ct_star, beta, T):
    return np.sum([beta ** (t + 1) * np.log(ct_star[t]) for t in range(T)])

# Parameters
r = 0.05
beta = 0.95
T = 40
a_min_values = [-40, -10, 0]
rho = 0.90
sigma = 0.1
y1 = 1
n_simulations = 100000

start_time = time.time()

# Simulation
yt_simulations = [solve_income(y1, T, rho, sigma) for _ in range(n_simulations)]
results = {a_min: {'yt': yt_simulations, 'ct': [], 'at': []} for a_min in a_min_values}

for a_min in a_min_values:
    for yt in yt_simulations:
        at, ct = decision_process(beta, r, yt, T, a_min)
        results[a_min]['ct'].append(ct)
        results[a_min]['at'].append(at)

# Calculate means and variances
mean_results = {a_min: {'yt': [], 'ct': [], 'at': []} for a_min in a_min_values}
var_results = {a_min: {'yt': [], 'ct': [], 'at': []} for a_min in a_min_values}

for a_min in a_min_values:
    mean_results[a_min]['yt'] = np.mean(results[a_min]['yt'], axis=0)
    mean_results[a_min]['ct'] = np.mean(results[a_min]['ct'], axis=0)
    mean_results[a_min]['at'] = np.mean(results[a_min]['at'], axis=0)
    var_results[a_min]['yt'] = np.var(results[a_min]['yt'], axis=0)
    var_results[a_min]['ct'] = np.var(results[a_min]['ct'], axis=0)
    var_results[a_min]['at'] = np.var(results[a_min]['at'], axis=0)

# Plotting Mean and Variance
for a_min in a_min_values:
    plt.figure(figsize=(10, 12))

    x_values = range(1, T + 1)

    # Finding y-axis limits for consistent and symmetric scaling
    y_max = max(np.max(mean_results[a_min]['yt'] + np.sqrt(var_results[a_min]['yt'])),
                np.max(mean_results[a_min]['ct'] + np.sqrt(var_results[a_min]['ct'])),
                np.max(mean_results[a_min]['at'] + np.sqrt(var_results[a_min]['at'])))

    y_min = min(np.min(mean_results[a_min]['yt'] - np.sqrt(var_results[a_min]['yt'])),
                np.min(mean_results[a_min]['ct'] - np.sqrt(var_results[a_min]['ct'])),
                np.min(mean_results[a_min]['at'] - np.sqrt(var_results[a_min]['at'])))

    y_limit = max(abs(y_max), abs(y_min))

    plt.subplot(3, 1, 1)
    plt.plot(x_values, mean_results[a_min]['yt'], label='Mean Income')
    plt.fill_between(x_values, mean_results[a_min]['yt'] - np.sqrt(var_results[a_min]['yt']),
                     mean_results[a_min]['yt'] + np.sqrt(var_results[a_min]['yt']), alpha=0.2)
    plt.title(f'Mean and Variance of Income for a_min={a_min}')
    plt.xlabel('Time')
    plt.ylabel('Income')
    plt.ylim(0.5, 1.5)
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.plot(x_values, mean_results[a_min]['ct'], label='Mean Consumption')
    plt.fill_between(x_values, mean_results[a_min]['ct'] - np.sqrt(var_results[a_min]['ct']),
                     mean_results[a_min]['ct'] + np.sqrt(var_results[a_min]['ct']), alpha=0.2)
    plt.title(f'Mean and Variance of Consumption for a_min={a_min}')
    plt.xlabel('Time')
    plt.ylabel('Consumption')
    plt.ylim(0.7, 1.3)
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 1, 3)
    plt.plot(x_values, mean_results[a_min]['at'], label='Mean Assets')
    plt.fill_between(x_values, mean_results[a_min]['at'] - np.sqrt(var_results[a_min]['at']),
                     mean_results[a_min]['at'] + np.sqrt(var_results[a_min]['at']), alpha=0.2)
    plt.title(f'Mean and Variance of Assets for a_min={a_min}')
    plt.xlabel('Time')
    plt.ylabel('Assets')
    plt.ylim(-y_limit, y_limit)
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

# Plotting Variance
for a_min in a_min_values:
    plt.figure(figsize=(10, 8))

    plt.subplot(3, 1, 1)
    plt.plot(x_values, var_results[a_min]['yt'], label='Variance of Income')
    plt.title(f'Variance of Income for a_min={a_min}')
    plt.xlabel('Time')
    plt.ylabel('Variance')
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.plot(x_values, var_results[a_min]['ct'], label='Variance of Consumption')
    plt.title(f'Variance of Consumption for a_min={a_min}')
    plt.xlabel('Time')
    plt.ylabel('Variance')
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 1, 3)
    plt.plot(x_values, var_results[a_min]['at'], label='Variance of Assets')
    plt.title(f'Variance of Assets for a_min={a_min}')
    plt.xlabel('Time')
    plt.ylabel('Variance')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

end_time = time.time()
print(f"Runtime: {end_time - start_time} seconds")
