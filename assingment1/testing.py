import numpy as np
import matplotlib.pyplot as plt

class UCBAlgorithm:
    """
    This class implements the UCB algorithm.
    The algorithm maintains estimates of the expected rewards for each arm
    and uses these estimates to select the arm with the highest Upper Confidence
    Bound (UCB) value at each round.

    Parameters:
    - num_articles (int): The number of arms (articles) in the multi-armed bandit.

    Methods:
    - get_ucb(article_index, total_rounds):
        Computes the Upper Confidence Bound (UCB) value for a given arm at
        the current round based on its historical rewards and exploration bonus.
        Parameters:
        * article_index (int): The index of the arm for which to compute the UCB value.
        * total_rounds (int): The total number of rounds completed so far.
        Returns:
        * ucb_value (float): The UCB value for the specified arm.

    - select_article(total_rounds):
        Selects the arm with the highest Upper Confidence Bound (UCB) value
        at the current round.
        Parameters:
        * total_rounds (int): The total number of rounds completed so far.
        Returns:
        * selected_article (int): The index of the arm selected for this round.

    - update(article_index, reward):
        Updates the historical records of rewards for the selected arm based on
        the observed reward in the current round.
        Parameters:
        * article_index (int): The index of the arm selected for this round.
        * reward (int): The reward obtained by selecting the specified arm.
    """
    def __init__(self, num_articles):
        self.num_articles = num_articles
        self.num_selections = np.zeros(num_articles)
        self.sum_rewards = np.zeros(num_articles)
        self.total_reward = 0
        self.cumulative_reward = []

    def get_ucb(self, article_index, total_rounds):
        if self.num_selections[article_index] == 0:
            return float('inf')  # Select unselected articles first
        avg_reward = self.sum_rewards[article_index] / self.num_selections[article_index]
        exploration_bonus = np.sqrt(2 * np.log(total_rounds) / self.num_selections[article_index])
        return avg_reward + exploration_bonus

    def select_article(self, total_rounds):
        ucb_values = [self.get_ucb(i, total_rounds) for i in range(self.num_articles)]
        return np.argmax(ucb_values)

    def update(self, article_index, reward):
        self.num_selections[article_index] += 1
        self.sum_rewards[article_index] += reward
        self.total_reward += reward
        self.cumulative_reward.append(self.total_reward)

# Defining user-news preferences
preferences = {
    'female_over_25': [0.8, 0.6, 0.5, 0.4, 0.2],
    'male_over_25': [0.2, 0.4, 0.5, 0.6, 0.8],
    'under_25': [0.2, 0.4, 0.8, 0.6, 0.5]
}

# Define optimal rewards for each article based on the highest click probability
optimal_rewards = [max(preferences[key]) for key in preferences]

# Define number of articles and total rounds.
num_articles = 5
total_rounds = 1000

# Initialize UCB algorithm.
ucb_algorithm = UCBAlgorithm(num_articles)

# Keeping track of cumulative regret and cumulative reward.
cumulative_regret = []
cumulative_reward = []

# Simulate user interactions and update algorithm
for round in range(1, total_rounds + 1):
    # Simulate user characteristics
    # Selecting a random age group.
    user_characteristics = np.random.choice(['female_over_25', 'male_over_25', 'under_25'])
    
    # Get the chosen age group's characteristics.
    click_probabilities = preferences[user_characteristics]

    # Select article using UCB algorithm.
    # Store its index in selected_article.
    selected_article = ucb_algorithm.select_article(round)

    # Calculate optimal reward for this round based on user's characteristics
    optimal_reward = max(click_probabilities)

    # Simulate user click based on selected article's click probability
    click_probability = click_probabilities[selected_article]
    if np.random.rand() < click_probability:
        reward = 1
    else:
        reward = 0

    # Update algorithm with the observed reward
    ucb_algorithm.update(selected_article, reward)

    # Calculate regret
    chosen_reward = ucb_algorithm.sum_rewards[selected_article]/ucb_algorithm.num_selections[selected_article] \
        if ucb_algorithm.num_selections[selected_article] > 0 else 0
    regret = optimal_reward - chosen_reward

    # Update cumulative regret and cumulative reward
    cumulative_regret.append(cumulative_regret[-1] + regret if round > 1 else regret)
    cumulative_reward.append(ucb_algorithm.total_reward)

# Evaluate algorithm's performance
total_reward = ucb_algorithm.total_reward
print(f"Cumulative total reward: {total_reward}")

# Plot cumulative regret and cumulative reward along with the linear function f(x) = x
plt.plot(range(1, total_rounds + 1), cumulative_regret, label='Cumulative Regret')
plt.plot(range(1, total_rounds + 1), cumulative_reward, label='Cumulative Reward')
plt.plot(range(1, total_rounds + 1), range(1, total_rounds + 1), label='f(x) = x', linestyle='--')
plt.xlabel('Rounds')
plt.ylabel('Value')
plt.title('Cumulative Regret and Cumulative Reward over Rounds')
plt.grid()
plt.legend()
plt.show()