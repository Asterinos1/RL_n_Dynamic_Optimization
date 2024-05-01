import numpy as np
import matplotlib.pyplot as plt

# Simulated stock_changes data, replace with actual data loading from 'stocks.csv'
# stock_changes = np.random.randn(2000, 10) * 0.05  # 2000 days, 10 stocks

# Load actual data from file
import pandas as pd
stock_changes = pd.read_csv('stocks.csv').values

num_days, num_stocks = stock_changes.shape
weights = np.ones(num_stocks)
learning_rate = 0.1
total_profits = []
cumulative_profit = 0

for day in range(num_days):
    weighted_profits = weights * stock_changes[day]
    chosen_stock = np.argmax(weighted_profits)
    profit = stock_changes[day, chosen_stock]
    
    # Update weights
    weights *= np.exp(learning_rate * stock_changes[day])
    weights /= np.sum(weights)  # Normalize weights

    cumulative_profit += profit
    total_profits.append(cumulative_profit)

# Plotting cumulative profits
plt.figure(figsize=(10, 5))
plt.plot(total_profits, label='Cumulative Profits')
plt.title('Cumulative Profits Over Time')
plt.xlabel('Days')
plt.ylabel('Profit (Euros)')
plt.legend()
plt.show()

transaction_costs = np.arange(0.5, 5.5, 0.5) / 100  # Transaction costs for 10 stocks

cumulative_profit_with_costs = 0
total_profits_with_costs = []

for day in range(num_days):
    weighted_profits = weights * (stock_changes[day] - transaction_costs)
    chosen_stock = np.argmax(weighted_profits)
    profit = stock_changes[day, chosen_stock] - transaction_costs[chosen_stock]
    
    weights *= np.exp(learning_rate * (stock_changes[day] - transaction_costs))
    weights /= np.sum(weights)  # Normalize weights

    cumulative_profit_with_costs += profit
    total_profits_with_costs.append(cumulative_profit_with_costs)

# Plotting comparison of cumulative profits with and without transaction costs
plt.figure(figsize=(10, 5))
plt.plot(total_profits, label='Cumulative Profits without Costs')
plt.plot(total_profits_with_costs, label='Cumulative Profits with Costs')
plt.title('Comparison of Cumulative Profits')
plt.xlabel('Days')
plt.ylabel('Profit (Euros)')
plt.legend()
plt.show()


cumulative_profit_bandit = 0
total_profits_bandit = []

for day in range(num_days):
    probabilities = weights / np.sum(weights)
    chosen_stock = np.random.choice(np.arange(num_stocks), p=probabilities)
    profit = stock_changes[day, chosen_stock] - transaction_costs[chosen_stock]
    
    # Update only the chosen stock's weight
    weights[chosen_stock] *= np.exp(learning_rate * (profit))
    weights /= np.sum(weights)  # Normalize weights

    cumulative_profit_bandit += profit
    total_profits_bandit.append(cumulative_profit_bandit)

# Plotting comparison of cumulative profits with and without transaction costs
plt.figure(figsize=(10, 5))
plt.plot(total_profits_with_costs, label='Cumulative Profits with Costs (Experts)')
plt.plot(total_profits_bandit, label='Cumulative Profits with Costs (Bandit)')
plt.title('Comparison of Cumulative Profits')
plt.xlabel('Days')
plt.ylabel('Profit (Euros)')
plt.legend()
plt.show()
