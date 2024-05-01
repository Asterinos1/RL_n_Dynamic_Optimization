import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load stock data
data = pd.read_csv('stocks.csv')
num_stocks = data.shape[1]
num_days = data.shape[0]

# Initial weights and parameters
weights = np.ones(num_stocks)
eta = 0.05  # Learning rate
wealth = 1
cumulative_profits = []
cumulative_regret = []

# Best possible strategy (for regret calculation)
best_stock_performance = data.cumsum().max(axis=1)

for t in range(num_days):
    # Normalize weights to get probabilities
    probabilities = weights / np.sum(weights)
    
    # Choose stock based on probabilities and get the gain/loss
    stock_performance_today = data.iloc[t]
    expected_gain = np.dot(probabilities, stock_performance_today)
    
    # Update wealth and record profit
    wealth += expected_gain / 100  # convert percentage to actual profit
    cumulative_profits.append(wealth)
    
    # Update weights
    weights *= np.exp(eta * stock_performance_today / 100)
    
    # Calculate regret
    optimal_wealth = 1 + best_stock_performance[t] / 100
    cumulative_regret.append(optimal_wealth - wealth)

# Plotting results
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(cumulative_regret, label='Cumulative Regret')
plt.title('Cumulative Regret over Time')
plt.xlabel('Days')
plt.ylabel('Regret')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(cumulative_profits, label='Cumulative Profits')
plt.title('Cumulative Profits over Time')
plt.xlabel('Days')
plt.ylabel('Profits')
plt.legend()

plt.tight_layout()
plt.show()

# Transaction costs for each stock
transaction_costs = 0.5 + np.arange(num_stocks) * 0.5

# Updated weight and wealth calculations
wealth = 1
cumulative_profits_with_fees = []

for t in range(num_days):
    probabilities = weights / np.sum(weights)
    stock_performance_today = data.iloc[t]
    effective_gain = stock_performance_today - transaction_costs
    expected_gain = np.dot(probabilities, effective_gain)
    
    wealth += expected_gain / 100
    cumulative_profits_with_fees.append(wealth)
    weights *= np.exp(eta * effective_gain / 100)

# Plotting new results against the old results
plt.figure(figsize=(12, 6))
plt.plot(cumulative_profits, label='Profits without Fees')
plt.plot(cumulative_profits_with_fees, label='Profits with Fees')
plt.title('Comparison of Cumulative Profits')
plt.xlabel('Days')
plt.ylabel('Profits')
plt.legend()
plt.show()

# Reset weights and wealth
weights = np.ones(num_stocks)
wealth = 1
cumulative_profits_bandit = []

for t in range(num_days):
    probabilities = weights / np.sum(weights)
    chosen_stock = np.random.choice(num_stocks, p=probabilities)
    actual_gain = data.iloc[t, chosen_stock] - transaction_costs[chosen_stock]
    
    wealth += actual_gain / 100
    cumulative_profits_bandit.append(wealth)
    
    # Update only the chosen stock's weight
    weights[chosen_stock] *= np.exp(eta * actual_gain / 100)

# Plotting bandit results
plt.plot(cumulative_profits_bandit, label='Bandit with Fees')
plt.title('Bandit Cumulative Profits with Fees')
plt.xlabel('Days')
plt.ylabel('Profits')
plt.legend()
plt.show()
