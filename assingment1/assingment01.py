import math
import random
import matplotlib.pyplot as plt

# This code is a modification of Kenneth Foo Fangwei's code,
# adjusted to the exercise's characteristics
# As of now it works for a single user type
# https://kfoofw.github.io/bandit-theory-ucb-analysis/

# UCB1 algorithm class
# We keep track for the stats of each user type separately
# female over 25 -> female
# male over 25 -> male
# male/female under 25 -> kid
class UCB1():
    def __init__(self, counts_female, values_female, counts_male, 
                values_male, counts_kid, values_kid,counts_kid2, values_kids2 ):
        self.counts_female = counts_female
        self.values_female = values_female
        
        self.counts_male = counts_male
        self.values_male = values_male
        
        self.counts_kid = counts_kid
        self.values_kid = values_kid
    
        self.counts_kid2 = counts_kid2
        self.values_kids2 = values_kids2
    
    def initialize(self, n_arms):
        self.counts_female = [0 for _ in range(n_arms)]
        self.values_female = [0.0 for _ in range(n_arms)]
        
        self.counts_male = [0 for _ in range(n_arms)]
        self.values_male = [0.0 for _ in range(n_arms)]
        
        self.counts_kid = [0 for _ in range(n_arms)]
        self.values_kid = [0.0 for _ in range(n_arms)]
        
        self.counts_kid2 = [0 for _ in range(n_arms)]
        self.values_kid2 = [0.0 for _ in range(n_arms)]
        
    
    def select_arm(self, user_type):
        if user_type == 'female_over_25':
            counts = self.counts_female
            values = self.values_female
        elif user_type == 'male_over_25':
            counts = self.counts_male
            values = self.values_male
        elif user_type == 'female_under_25':
            counts = self.counts_kid
            values = self.values_kid
        elif user_type == 'male_under_25':
            counts = self.counts_kid
            values = self.values_kid
        else:
            raise ValueError("Invalid user type")

        n_arms = len(counts)
        for arm in range(n_arms):
            if counts[arm] == 0:
                return arm

        ucb_values = [0.0 for _ in range(n_arms)]
        total_counts = sum(counts)

        for arm in range(n_arms):
            bonus = math.sqrt((2 * math.log(total_counts)) / float(counts[arm]))
            ucb_values[arm] = values[arm] + bonus

        return ucb_values.index(max(ucb_values))
    
    def update(self, chosen_arm, reward, user_type):
        if user_type == 'female_over_25':
            counts = self.counts_female
            values = self.values_female
        elif user_type == 'male_over_25':
            counts = self.counts_male
            values = self.values_male
        elif user_type == 'female_under_25':
            counts = self.counts_kid
            values = self.values_kid
        elif user_type == 'male_under_25':
            counts = self.counts_kid
            values = self.values_kid
        else:
            raise ValueError("Invalid user type")
        
        counts[chosen_arm] += 1
        n = counts[chosen_arm]
        value = values[chosen_arm]
        new_value = ((n - 1) / float(n)) * value + (1 / float(n)) * reward
        values[chosen_arm] = new_value

# Bernoulli arm class
# The reward for success returns 1 else 0
class BernoulliArm():
    def __init__(self, p):
        self.p = p

    def draw(self):
        if random.random() > self.p:
            return 0.0
        else:
            return 1.0

# Define parameters
num_sims = 1   # How many times to run the simulation
horizon = 10000  # Defining the horizon

# Define arms for female over 25, male over 25, and kids (under 25)
arms_female_over_25 = [BernoulliArm(0.8), BernoulliArm(0.6), BernoulliArm(0.5), BernoulliArm(0.4), BernoulliArm(0.2)]
arms_male_over_25 = list(reversed(arms_female_over_25))
arms_female_under_25 = [BernoulliArm(0.2), BernoulliArm(0.4), BernoulliArm(0.8), BernoulliArm(0.6), BernoulliArm(0.5)]

#new kid
arms_male_under_25 = [BernoulliArm(0.2), BernoulliArm(0.4), BernoulliArm(0.8), BernoulliArm(0.6), BernoulliArm(0.5)]

# Initialize UCB1 algorithm
ucb = UCB1([], [], [], [], [], [], [], [])

# Function to simulate the algorithm
def test_algorithm(algo, arms_female_over_25, arms_male_over_25, arms_female_under_25, arms_male_under_25, num_sims, horizon):
    chosen_arms = [0 for _ in range(num_sims * horizon)]
    rewards = [0.0 for _ in range(num_sims * horizon)]
    cumulative_rewards = [0.0 for _ in range(num_sims * horizon)]
    cumulative_regret = [0.0 for _ in range(num_sims * horizon)]
    sim_nums = [0 for _ in range(num_sims * horizon)]
    times = [0 for _ in range(num_sims * horizon)]
    
    #Ignore this
    for sim in range(num_sims):
        algo.initialize(len(arms_female_over_25))  # Initialize for female arms
        
        for t in range(horizon):
            
            index = sim * horizon + t
            sim_nums[index] = sim
            times[index] = t + 1
            
            # female_over_25
            # male_over_25
            # Randomly choose user type for this round
            user_type = random.choice(['female_over_25', 'male_over_25', 'female_under_25', 'male_under_25'])
            # For debugging purposes
            # print(f"Current user is a {user_type}")
            
            # Select arm using UCB algorithm based on user type
            if user_type == 'female_over_25':
                chosen_arm = algo.select_arm('female_over_25')
                reward = arms_female_over_25[chosen_arm].draw()
            elif user_type == 'male_over_25':
                chosen_arm = algo.select_arm('male_over_25')
                reward = arms_male_over_25[chosen_arm].draw()
            elif user_type == 'female_under_25':
                chosen_arm = algo.select_arm('female_under_25')
                reward = arms_female_under_25[chosen_arm].draw()
            else:
                chosen_arm = algo.select_arm('male_under_25')
                reward = arms_male_under_25[chosen_arm].draw()
            
            chosen_arms[index] = chosen_arm
            rewards[index] = reward
            
            if t == 0:
                cumulative_rewards[index] = reward
            else:
                cumulative_rewards[index] = cumulative_rewards[index - 1] + reward
            
            # Calculate regret
            if user_type == 'female_over_25':
                optimal_reward = max(arm.p for arm in arms_female_over_25)
            elif user_type == 'male_over_25':
                optimal_reward = max(arm.p for arm in arms_male_over_25)
            elif user_type == 'female_under_25':
                optimal_reward = max(arm.p for arm in arms_female_under_25)
            else:
                optimal_reward = max(arm.p for arm in arms_male_under_25)
                
            cumulative_regret[index] = (t + 1) * optimal_reward - cumulative_rewards[index]
            
            # Update algorithm with chosen arm and reward
            if user_type == 'female_over_25':
                algo.update(chosen_arm, reward, 'female_over_25')
            elif user_type == 'male_over_25':
                algo.update(chosen_arm, reward, 'male_over_25')
            elif user_type == 'female_under_25':
                algo.update(chosen_arm, reward, 'female_under_25')
            else:
                algo.update(chosen_arm, reward, 'male_under_25')
    
    return sim_nums, times, chosen_arms, rewards, cumulative_rewards, cumulative_regret

# Simulate the algorithm
sim_nums, times, chosen_arms, rewards, cumulative_rewards, cumulative_regret = test_algorithm(ucb, arms_female_over_25, arms_male_over_25, arms_female_under_25, arms_male_under_25, num_sims, horizon)

# Plot results
plt.figure(figsize=(10, 6))



# Plot cumulative regret and reward
plt.plot(range(num_sims * horizon), cumulative_regret, label='Cumulative Regret', color='red')
#plt.plot(range(num_sims * horizon), cumulative_rewards, label='Cumulative Reward', color='blue')

# Plot linear function f(x)=x to compare with previous results
plt.plot(range(num_sims * horizon), [t + 1 for t in range(num_sims * horizon)], linestyle='--', label='f(x) = x', color='green')

plt.xlabel('Time Step')
plt.ylabel('Cumulative Value')
plt.title('UCB Algorithm Performance')
plt.legend()
plt.grid(True)