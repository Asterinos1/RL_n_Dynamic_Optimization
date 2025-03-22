import numpy as np
import matplotlib.pyplot as plt

# student id -> 2020030107 -> seed:07

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

best_possible_reward = np.max(prices)
naive_rounds = T // 4

for t in range(T):
    probabilities = weights / np.sum(weights)
    chosen_price_index = np.random.choice(5, p=probabilities)
    picked_prices[t] = prices[chosen_price_index]
    
    if t < naive_rounds:
        competitor_price = np.random.uniform(0, 4)
        user_picked_you = np.random.choice([True, False])  # Random choice
        reward = prices[chosen_price_index] if user_picked_you else 0
    else:
        competitor_price = np.random.uniform(0, 4)
        if prices[chosen_price_index] <= competitor_price:
            reward = prices[chosen_price_index]
        else:
            reward = 0

    rewards[t] = reward

    
    if reward > 0:
        l_t = 0  
    else:
        l_t = 1  

    #w(t+1) = w(t) * (1 - eta)^l(t)
    weights[chosen_price_index] *= (1 - eta) ** l_t
    
    cumulative_rewards[t] = np.sum(rewards[:t + 1])
    total_possible_reward = best_possible_reward * (t + 1)
    regrets[t] = total_possible_reward - cumulative_rewards[t]

    #debugging print to check weight prices
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
