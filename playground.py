import math
import random
import matplotlib.pyplot as plt

# UCB1 algorithm class
class UCB1():
    def __init__(self, counts, values):
        self.counts = counts  # Count represent counts of pulls for each arm
        self.values = values  # Value represent average reward for specific arm
    
    def initialize(self, n_arms, num_user_types):
        self.counts = [[0 for _ in range(n_arms)] for _ in range(num_user_types)]  # Counts for each user type
        self.values = [[0.0 for _ in range(n_arms)] for _ in range(num_user_types)]  # Values for each user type
        self.ucb_values = [[0.0 for _ in range(n_arms)] for _ in range(num_user_types)]  # UCB values for each user type
    
    # UCB arm selection based on max of UCB reward of each arm
    def select_arm(self, user_type_index):
        n_arms = len(self.counts[user_type_index])
        
        # Prioritize arms that haven't been played at all for this user type.
        for arm in range(n_arms):
            if self.counts[user_type_index][arm] == 0:
                return arm
        
        # Update UCB values for this user type
        total_counts = sum(self.counts[user_type_index])
        for arm in range(n_arms):
            bonus = math.sqrt((2 * math.log(total_counts)) / float(self.counts[user_type_index][arm]))
            self.ucb_values[user_type_index][arm] = self.values[user_type_index][arm] + bonus
        
        return self.ucb_values[user_type_index].index(max(self.ucb_values[user_type_index]))
    
    # Choose to update chosen arm and reward
    def update(self, chosen_arm, reward, user_type_index):
        self.counts[user_type_index][chosen_arm] += 1
        n = self.counts[user_type_index][chosen_arm]
        
        # Update average/mean value/reward for chosen arm and user type
        value = self.values[user_type_index][chosen_arm]
        new_value = ((n - 1) / float(n)) * value + (1 / float(n)) * reward
        self.values[user_type_index][chosen_arm] = new_value

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
num_users = 1000  # Number of users
horizon = 1000    # Defining the horizon

# Define click probabilities for different user types and articles
user_types = {
    'female_over_25': [0.8, 0.6, 0.5, 0.4, 0.2],
    'male_over_25': [0.2, 0.4, 0.5, 0.6, 0.8],
    'male_female_under_25': [0.2, 0.4, 0.8, 0.6, 0.5]
}

# Initialize UCB1 algorithm
ucb = UCB1([], [])

# Function to simulate the algorithm for a uniform distribution of users
def test_algorithm(algo, num_users, horizon):
    chosen_arms = [[] for _ in range(len(user_types))]  # Chosen arms for each user type
    rewards = [[] for _ in range(len(user_types))]      # Rewards for each user type
    cumulative_rewards = [[] for _ in range(len(user_types))]  # Cumulative rewards for each user type
    cumulative_regret = [[] for _ in range(len(user_types))]   # Cumulative regret for each user type
    times = [t + 1 for t in range(horizon)]
    
    first_user_type = list(user_types.keys())[0]  # Get the first user type from the dictionary
    algo.initialize(len(user_types[first_user_type]), len(user_types))  # Initialize algorithm with the number of arms and user types

    for t in range(horizon):
        # Generate a new user randomly
        user_type_index = random.randint(0, len(user_types) - 1)  # Choose a random user type
        click_probabilities = user_types[list(user_types.keys())[user_type_index]]
        arms = [BernoulliArm(p) for p in click_probabilities]
        
        #for user_type in range(len(user_types)):
        # Select arm using UCB algorithm for this user type
        chosen_arm = algo.select_arm(user_type_index)
        chosen_arms[user_type_index].append(chosen_arm)
            
        # Draw reward from selected arm using Bernoulli.
        # Success -> reward = 1
        # Fail -> reward = 0
        reward = arms[chosen_arm].draw()
        rewards[user_type_index].append(reward)
            
        if t == 0:
            cumulative_rewards[user_type_index].append(reward)
        else:
            cumulative_rewards[user_type_index].append(cumulative_rewards[user_type_index][-1] + reward)
            
        # Calculate regret
        optimal_reward = max(click_probabilities)
        cumulative_regret[user_type_index].append((t + 1) * optimal_reward - cumulative_rewards[user_type_index][-1])
            
        # Update algorithm with chosen arm and reward for this user type
        algo.update(chosen_arm, reward, user_type_index)
    
    return times, chosen_arms, rewards, cumulative_rewards, cumulative_regret

# Simulate the algorithm for a uniform distribution of users
times, chosen_arms, rewards, cumulative_rewards, cumulative_regret = test_algorithm(ucb, num_users, horizon)

# Plot results for each user type
plt.figure(figsize=(12, 8))

for i, user_type in enumerate(user_types):
    plt.plot(times, cumulative_regret[i], label='Cumulative Regret - {}'.format(user_type.replace('_', ' ').title()))

plt.xlabel('Time Step')
plt.ylabel('Cumulative Regret')
plt.title('UCB Algorithm Performance')
plt.legend()
plt.grid(True)
plt.show()
