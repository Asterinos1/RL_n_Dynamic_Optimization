import numpy as np
import matplotlib.pyplot as plt

# student id -> 2020030107 -> seed:07

T = 100  
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

# Question 1 logic remains for the first T/4 rounds (user is naive)
best_possible_reward = np.max(prices)

# T/4 is the point where user becomes smart
naive_rounds = T // 4

for t in range(T):
    probabilities = weights / np.sum(weights)
    chosen_price_index = np.random.choice(5, p=probabilities)
    picked_prices[t] = prices[chosen_price_index]
    
    if t < naive_rounds:
        # User picks randomly between you and competitor (naive user)
        competitor_price = np.random.uniform(0, 4)
        user_picked_you = np.random.choice([True, False])  # Random choice

        # If user picks you, you earn the chosen price, otherwise you get 0
        reward = prices[chosen_price_index] if user_picked_you else 0
    else:
        # User becomes smart: picks the lower price between you and the competitor
        competitor_price = np.random.uniform(0, 4)
        
        # User picks the lower price (smart user)
        if prices[chosen_price_index] <= competitor_price:
            reward = prices[chosen_price_index]
        else:
            reward = 0

    # Update MW weights
    rewards[t] = reward
    if reward > 0:  # only update if reward is earned
        weights[chosen_price_index] *= np.exp(eta * reward / np.sum(prices))
    
    # Store cumulative rewards
    cumulative_rewards[t] = np.sum(rewards[:t + 1])
    
    # Calculate regret as the difference between the best possible reward and actual cumulative reward
    total_possible_reward = best_possible_reward * (t + 1)
    regrets[t] = total_possible_reward - cumulative_rewards[t]

    # Print weights after each iteration
    print(f"Iteration {t+1}: Weights: {weights}")

# Plot the cumulative reward and regret
plt.plot(cumulative_rewards, label='Cumulative Reward')
plt.plot(regrets, label='Regret', linestyle='--')
plt.title('Profit and Regret over time')
plt.xlabel('Rounds (t)')
plt.ylabel('Cumulative Value')
plt.grid(True)
plt.legend()
plt.show()

# Print final observations
final_probabilities = weights / np.sum(weights)
print("Final weights:", weights)
print("Final price probabilities:", final_probabilities)
print("Total reward after T rounds:", np.sum(rewards))
print("Total regret after T rounds:", regrets[-1])
