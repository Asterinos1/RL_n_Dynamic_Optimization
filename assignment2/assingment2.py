import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load data from CSV
data = pd.read_csv('stocks.csv')

# Number of stocks (K) and days (T)
K = data.shape[1]
T = data.shape[0]

# Initial weights
weights = np.ones(K)

# Learning rate
eta = 0.1

# To store cumulative profit and regrets
cumulative_profits = np.zeros(T)
cumulative_regret = np.zeros(T)

# Total optimal profit (if best stock was chosen every day)
optimal_profit = data.max(axis=1).sum()

# Simulation of the trading process
profit = 0
for t in range(T):
    # Normalize weights to get probabilities
    probabilities = weights / weights.sum()
    
    # Choose stock based on probabilities
    stock_choice = np.random.choice(K, p=probabilities)
    
    # Get today's stock performance
    stock_performance = data.iloc[t, stock_choice]
    
    # Update total profit
    profit += stock_performance
    cumulative_profits[t] = profit
    
    # Calculate regret (difference from optimal)
    cumulative_regret[t] = optimal_profit - cumulative_profits[t]
    
    # Update weights based on performance
    weights *= np.exp(eta * data.iloc[t] / 100)  # Dividing by 100 to convert percentage to fraction

# Plotting cumulative profit
plt.figure(figsize=(14, 7))
plt.subplot(1, 2, 1)
plt.plot(cumulative_profits, label='Cumulative Profit')
plt.title('Cumulative Profits Over Time')
plt.xlabel('Days')
plt.ylabel('Profit in Euros')
plt.legend()

# Plotting cumulative regret
plt.subplot(1, 2, 2)
plt.plot(cumulative_regret, label='Cumulative Regret')
plt.title('Cumulative Regret Over Time')
plt.xlabel('Days')
plt.ylabel('Regret in Euros')
plt.legend()

plt.tight_layout()
plt.show()
