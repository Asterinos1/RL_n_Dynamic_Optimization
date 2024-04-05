import random
import numpy as nm
import time

class User:
    def __init__(self, gender, age):
        self.gender = gender
        self.age = age

#method to generate users.
#There are 4 types of users here:
#Male, above 25.
#Male, under 25.
#Female, above 25.
#Female, under 25.
#They all have 1/4 chance to appear (Could be done in a better way.)

def generate_users(num_users):
    users = []
    genders = ['female', 'male']  # Only female and male genders are considered
    num_above_25 = num_users // 2  # Number of users with age above 25
    num_below_25 = num_users - num_above_25  # Number of users with age below 25

    above_25_users = []
    below_25_users = []

    for _ in range(num_above_25):
        gender = random.choice(genders)
        age = random.randint(26, 70)  # Ages above 25
        user = User(gender, age)
        above_25_users.append(user)

    for _ in range(num_below_25):
        gender = random.choice(genders)
        age = random.randint(12, 25)  # Ages below or equal to 25
        user = User(gender, age)
        below_25_users.append(user)

    users.extend(above_25_users)
    users.extend(below_25_users)

    #return users, above_25_users, below_25_users
    return users


if __name__ == "__main__":
    num_users = 10000  # Change this to the desired number of users
    
    T = 1000
    #We have K = 5 arms.
    
    rewards_of_arm_i = {} #key is the arm
    
    rewards_dic = {}
    
    for i in range(100):
        # Generate a random number between 0 and 1
        random_number = random.random()

        # Determine the value based on the random number
        if random_number < 0.5:
            value = 0
        else:
            value = 1
            
        rewards_dic[i] = value
        
    #let's calculate the mean.
    
    #Get the denominator.
    times_arm_i_was_selected = 0
    
    
    
    for index, value in rewards_dic.items():
        if(value==1):
            times_arm_i_was_selected+=1
            
    nominator = times_arm_i_was_selected
    
    mean_estimate_m = nominator-5/times_arm_i_was_selected
    mean_estimate_N = times_arm_i_was_selected
    print(mean_estimate_m)
    #print(rewards_dic)
    #print(f"This arm was selected {times_arm_i_was_selected} times.")
    
    upper_confidence_bound = mean_estimate_m + nm.sqrt((2*nm.log(T))/mean_estimate_N)
    
    
    print(upper_confidence_bound)
    
    ucb_list = [28.94, 15.72, 83.05, 44.19, 7.63, 90.12, 55.38, 36.87, 71.56, 9.82]
    
    print(f"Current UCB list: {ucb_list}")
    
    best_arm = ucb_list[0]
    
    for i in ucb_list:
        if i>best_arm:
            best_arm=i
    
    print(best_arm)
    