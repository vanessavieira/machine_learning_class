import random

N_SIMULATIONS = 1000
POLICY_RESULTS = 0

def prizedoor_and_guess():
    prizedoor = random.sample(range(1, 4), 1)
    guess = random.sample(range(1, 4), 1)
    return prizedoor, guess

def show_goat_door(prizedoor, guess):
    goat_door = random.sample(range(1, 4), 1)

    while True:

        if (goat_door != prizedoor) & (goat_door != guess):
            return goat_door
        else:
            goat_door = random.sample(range(1, 4), 1)

def not_switch_policy(prizedoor, guess):
    if (guess == prizedoor):
        return True
    else:
        return False

def switch_policy(prizedoor, guess, goat_door):
    new_guess = random.sample(range(1, 4), 1)

    while True:

        if (new_guess == guess == goat_door) | (new_guess == goat_door) | (new_guess == guess):
            new_guess = random.sample(range(1, 4), 1)
        else:
            if (new_guess == prizedoor):
                return True
            else:
                return False

# Main script

switch_results = 0
not_switch_results = 0

for sim in range(1000):
    prizedoor, guess = prizedoor_and_guess()
    goat_door = show_goat_door(prizedoor, guess)

    if (not_switch_policy(prizedoor, guess) == True):
        not_switch_results = not_switch_results + 1

    if (switch_policy(prizedoor, guess, goat_door) == True):
        switch_results = switch_results + 1

not_switch_results_percentage = (not_switch_results/1000)*100
switch_results_percentage = (switch_results/1000)*100

print("Percentage when you don't change the door: " + str(not_switch_results_percentage) + "\n")
print("Percentage when you always change the door: " + str(switch_results_percentage) + "\n")