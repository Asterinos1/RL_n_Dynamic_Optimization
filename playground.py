import math
import random
import matplotlib.pyplot as plt

# This code is a modification of Kenneth Foo Fangwei's code,
# adjusted to the exercise's characteristics
# https://kfoofw.github.io/bandit-theory-ucb-analysis/

# UCB1 algorithm class
class UCB1():
    def __init__(self, counts, values):
        self.counts = counts # Count represent counts of pulls for each arm
        self.values = values # Value represent average reward for specific arm
    
    def initialize(self, n_arms):
        self.counts = [0 for _ in range(n_arms)]
        self.values = [0.0 for _ in range(n_arms)]
    
    # UCB arm selection based on max of UCB reward of each arm
    def select_arm(self):
        n_arms = len(self.counts)
        
        
        for arm in range(n_arms):
            if self.counts[arm] == 0:
                return arm
        
        ucb_values = [0.0 for _ in range(n_arms)]
        total_counts = sum(self.counts)
        
        #Applying ucb formula here
        for arm in range(n_arms):
            bonus = math.sqrt((2 * math.log(total_counts)) / float(self.counts[arm]))
            ucb_values[arm] = self.values[arm] + bonus
        
        return ucb_values.index(max(ucb_values))
    
    # Choose to update chosen arm and reward
    def update(self, chosen_arm, reward):
        self.counts[chosen_arm] += 1
        n = self.counts[chosen_arm]
        
        # Update average/mean value/reward for chosen arm
        value = self.values[chosen_arm]
        new_value = ((n - 1) / float(n)) * value + (1 / float(n)) * reward
        self.values[chosen_arm] = new_value

# Bernoulli arm class
class BernoulliArm():
    def __init__(self, p):
        self.p = p

    def draw(self):
        if random.random() > self.p:
            return 0.0
        else:
            return 1.0

# Define parameters   
horizon = 10000  # Defining the horizon.

# Define arms with their corresponding click probabilities
arms = [BernoulliArm(0.8), BernoulliArm(0.6), BernoulliArm(0.5), BernoulliArm(0.4), BernoulliArm(0.2)]

# Initialize UCB1 algorithm
ucb = UCB1([], [])

# Function to simulate the algorithm
def test_algorithm(algo, arms, horizon):
    chosen_arms = [0 for _ in range( horizon)]
    rewards = [0.0 for _ in range(horizon)]
    cumulative_rewards = [0.0 for _ in range(horizon)]
    cumulative_regret = [0.0 for _ in range( horizon)]
    times = [0 for _ in range(horizon)]
    
    
    algo.initialize(len(arms))
    for t in range(horizon):
        index = horizon + t
        times[index] = t + 1
            
            # Select arm using UCB1 algorithm
        chosen_arm = algo.select_arm()
        chosen_arms[index] = chosen_arm
            
            # Draw reward from selected arm
        reward = arms[chosen_arm].draw()
        rewards[index] = reward
            
        if t == 0:
            cumulative_rewards[index] = reward
        else:
            cumulative_rewards[index] = cumulative_rewards[index - 1] + reward
            
        # Calculate regret
        optimal_reward = max(arm.p for arm in arms)
        cumulative_regret[index] = (t + 1) * optimal_reward - cumulative_rewards[index]
            
        # Update algorithm with chosen arm and reward
        algo.update(chosen_arm, reward) 


    return times, chosen_arms, rewards, cumulative_rewards, cumulative_regret

# Simulate the algorithm
times, chosen_arms, rewards, cumulative_rewards, cumulative_regret = test_algorithm(ucb, arms, horizon)

# Plot results
plt.figure(figsize=(10, 6))

# Plot cumulative regret and reward
plt.plot(range( horizon), cumulative_regret, label='Cumulative Regret', color='red')
plt.plot(range( horizon), cumulative_rewards, label='Cumulative Reward', color='blue')

# Plot linear function f(x)=x
plt.plot(range(horizon), [t + 1 for t in range(* horizon)], linestyle='--', label='f(x) = x', color='green')

plt.xlabel('Time Step')
plt.ylabel('Cumulative Value')
plt.title('UCB1 Algorithm Performance')
plt.legend()
plt.grid(True)
plt.show()