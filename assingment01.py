import random
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
    
    '''
    all_users, above_25_users, below_25_users = generate_users(num_users)
    

    # Count the number of users based on gender and age categories
    male_above_25_count = sum(1 for user in above_25_users if user.gender == 'male')
    female_above_25_count = sum(1 for user in above_25_users if user.gender == 'female')
    male_below_25_count = sum(1 for user in below_25_users if user.gender == 'male')
    female_below_25_count = sum(1 for user in below_25_users if user.gender == 'female')

    # Print the results
    print("Male with age above 25:", male_above_25_count)
    print("Female with age above 25:", female_above_25_count)
    print("Male with age below 25:", male_below_25_count)
    print("Female with age below 25:", female_below_25_count)
    
    # Print the first 10 users
    for i, user in enumerate(all_users[:100]):
        print(f"User {i+1}: Gender: {user.gender}, Age: {user.age}")
    '''
    #We care for all_users only, there others are for debugging purposes.
    #all_users, above_25_users, below_25_users = generate_users(num_users)
    all_users = generate_users(num_users)
    print(all_users[2].gender)
