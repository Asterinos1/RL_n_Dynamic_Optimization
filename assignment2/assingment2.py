import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv('stocks.csv')

# Assuming the file has columns for each stock's daily percentage changes
# e.g., stock_1, stock_2, ..., stock_K

# Get the number of stocks (K) and days (T)
K = data.shape[1]
T = data.shape[0]

def multiplicative_weights(data, eta=0.1):
    K, T = data.shape[1], data.shape[0]
    weights = np.ones(K)
    cumulative_regret = np.zeros(T)
    cumulative_profit = np.zeros(T)

    for t in range(T):
        probabilities = weights / np.sum(weights)
        chosen_stock = np.random.choice(K, p=probabilities)
        reward = data.iloc[t, chosen_stock] / 100.0

        # Update weights
        weights[chosen_stock] *= np.exp(eta * reward)

        # Calculate regret and profit
        best_possible = np.max(data.iloc[t] / 100.0)
        cumulative_regret[t] = cumulative_regret[t - 1] + (best_possible - reward) if t > 0 else (best_possible - reward)
        cumulative_profit[t] = cumulative_profit[t - 1] + reward if t > 0 else reward

    return cumulative_regret, cumulative_profit

# Run the algorithm
cumulative_regret, cumulative_profit = multiplicative_weights(data)

# Plot results
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(cumulative_regret)
plt.title('Cumulative Regret')
plt.xlabel('Days')
plt.ylabel('Regret')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(cumulative_profit)
plt.title('Cumulative Profit')
plt.xlabel('Days')
plt.ylabel('Profit')
plt.grid(True)

plt.show()


def multiplicative_weights_with_fees(data, fees, eta=0.1):
    K, T = data.shape[1], data.shape[0]
    weights = np.ones(K)
    cumulative_regret = np.zeros(T)
    cumulative_profit = np.zeros(T)

    for t in range(T):
        probabilities = weights / np.sum(weights)
        chosen_stock = np.random.choice(K, p=probabilities)
        reward = (data.iloc[t, chosen_stock] - fees[chosen_stock]) / 100.0

        # Update weights
        weights[chosen_stock] *= np.exp(eta * reward)

        # Calculate regret and profit
        best_possible = np.max((data.iloc[t] - fees) / 100.0)
        cumulative_regret[t] = cumulative_regret[t - 1] + (best_possible - reward) if t > 0 else (best_possible - reward)
        cumulative_profit[t] = cumulative_profit[t - 1] + reward if t > 0 else reward

    return cumulative_regret, cumulative_profit

# Define transaction fees
fees = np.linspace(0.005, 0.05, K) * 100  # 0.5%, 1%, ..., K%

# Run the algorithm with fees
cumulative_regret_fees, cumulative_profit_fees = multiplicative_weights_with_fees(data, fees)

# Plot results
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(cumulative_regret, label='Without Fees')
plt.plot(cumulative_regret_fees, label='With Fees')
plt.title('Cumulative Regret')
plt.xlabel('Days')
plt.ylabel('Regret')
plt.grid(True)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(cumulative_profit, label='Without Fees')
plt.plot(cumulative_profit_fees, label='With Fees')
plt.title('Cumulative Profit')
plt.xlabel('Days')
plt.ylabel('Profit')
plt.grid(True)
plt.legend()

plt.show()

# Task 3: Bandits with Transaction Fees
def exp3(data, fees, eta=0.1, gamma=0.1):
    K, T = data.shape[1], data.shape[0]
    weights = np.ones(K)
    cumulative_regret = np.zeros(T)
    cumulative_profit = np.zeros(T)

    for t in range(T):
        probabilities = (1 - gamma) * (weights / np.sum(weights)) + (gamma / K)
        chosen_stock = np.random.choice(K, p=probabilities)
        reward = (data.iloc[t, chosen_stock] - fees[chosen_stock]) / 100.0

        # Update weights
        estimated_reward = reward / probabilities[chosen_stock]
        weights[chosen_stock] *= np.exp(eta * estimated_reward)

        # Calculate regret and profit
        best_possible = np.max((data.iloc[t] - fees) / 100.0)
        cumulative_regret[t] = cumulative_regret[t - 1] + (best_possible - reward) if t > 0 else (best_possible - reward)
        cumulative_profit[t] = cumulative_profit[t - 1] + reward if t > 0 else reward

    return cumulative_regret, cumulative_profit

# Run the EXP3 algorithm with fees
cumulative_regret_bandit, cumulative_profit_bandit = exp3(data, fees)

# Plot results
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(cumulative_regret, label='Experts Without Fees')
plt.plot(cumulative_regret_fees, label='Experts With Fees')
plt.plot(cumulative_regret_bandit, label='Bandit With Fees')
plt.title('Cumulative Regret')
plt.xlabel('Days')
plt.ylabel('Regret')
plt.grid(True)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(cumulative_profit, label='Experts Without Fees')
plt.plot(cumulative_profit_fees, label='Experts With Fees')
plt.plot(cumulative_profit_bandit, label='Bandit With Fees')
plt.title('Cumulative Profit')
plt.xlabel('Days')
plt.ylabel('Profit')
plt.grid(True)
plt.legend()

plt.show()
