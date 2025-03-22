import numpy as np
import matplotlib.pyplot as plt

#student id -> 2020030107 -> seed:07

T = 1000  
alpha = 0.5
beta = 1.5
price_options = ['p0', 'p1', 'p2', 'p3', 'p4']
P = np.random.uniform(1.5, 2.5)  

prices = [
    alpha**2 * P,  # p0
    alpha * P,     # p1
    P,             # p2
    beta * P,      # p3
    beta**2 * P    # p4
]

np.random.seed(7)  
weights = np.ones(5)  
eta = 0.1

rewards = np.zeros(T)
picked_prices = np.zeros(T)
cumulative_rewards = np.zeros(T)
regrets = np.zeros(T)

"""
Regarding question 1:

Since the customer always agrees to pay after choosing,
it would be best to always suggest him the biggest price.
However we are tasked to assume this is not true
Then how do we proceed?
"""
best_possible_reward = np.max(prices)

for t in range(T):
    
    probabilities = weights / np.sum(weights)
    chosen_price_index = np.random.choice(5, p=probabilities)
    picked_prices[t] = prices[chosen_price_index]
    
    
    competitor_price = np.random.uniform(0, 4)
    user_picked_you = np.random.choice([True, False])  

    reward = prices[chosen_price_index] if user_picked_you else 0
    rewards[t] = reward

    
    if reward > 0:  # only update if reward is earned
        weights[chosen_price_index] *= np.exp(eta * reward / np.sum(prices))
    
    cumulative_rewards[t] = np.sum(rewards[:t + 1])
    
    total_possible_reward = best_possible_reward * (t + 1)
    regrets[t] = total_possible_reward - cumulative_rewards[t]

    print(f"Iteration {t+1}: Weights: {weights}")


plt.plot(cumulative_rewards, label='Cumulative Reward')
plt.plot(regrets, label='Regret', linestyle='--')
plt.title('Profit and Regret over time')
plt.xlabel('Rounds (t)')
plt.ylabel('Cumulative Value')
plt.grid(True)
plt.legend()
plt.show()

final_probabilities = weights / np.sum(weights)
print("Final weights:", weights)
print("Final price probabilities:", final_probabilities)
print("Total reward after T rounds:", np.sum(rewards))
print("Total regret after T rounds:", regrets[-1])
