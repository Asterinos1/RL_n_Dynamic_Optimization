import numpy as np
import matplotlib.pyplot as plt

# student id -> 2020030107 -> seed:07

T = 10000  
alpha = 0.5
beta = 1.5
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

for t in range(T):
    #we calculate the probabilities based on the weights and probabilisticaly make a choice.
    probabilities = weights / np.sum(weights)
    chosen_price_index = np.random.choice(5, p=probabilities)
    picked_prices[t] = prices[chosen_price_index]
    
    #get a random price for our competitor
    competitor_price = np.random.uniform(0, 4)
    #in this case, the user selects randomly between us and the competitor.
    user_picked_you = np.random.choice([True, False])
    reward = prices[chosen_price_index] if user_picked_you else 0
    rewards[t] = reward

    #setting the loss function values.
    #this was based on the professor's pdf regarding the experts.
    if reward > 0:
        l_t = 0  #no loss if we actually recieve reward.
    else:
        l_t = 1  #we have loss if we don't get the reward.
    
    #w(t+1) = w(t) * (1 - eta)^l(t)
    #update all weights.
    for i in range(len(weights)):
        weights[i] *= (1 - eta) ** (1 if i == chosen_price_index and reward == 0 else 0)

    cumulative_rewards[t] = np.sum(rewards[:t + 1])
    total_possible_reward = best_possible_reward * (t + 1)
    regrets[t] = total_possible_reward - cumulative_rewards[t]
    #debugging print
    #print(f"Iteration {t+1}:\nWeights: {weights}")

plt.plot(cumulative_rewards, label='total profit')
plt.plot(regrets, label='regret', linestyle='--')
plt.title('profit and regret over time')
plt.xlabel('rounds')
plt.ylabel('total profit')
plt.grid(True)
plt.legend()
plt.show()

final_probabilities = weights / np.sum(weights)
print("Final weights:", weights)
print("Final price probabilities:", final_probabilities)
print("Total reward after T rounds:", np.sum(rewards))
print("Total regret after T rounds:", regrets[-1])
