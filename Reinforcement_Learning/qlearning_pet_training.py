# qlearning_pet_training.py

import numpy as np 
import random 

# Environment setup 
goal = 5 
positions = list(range(goal + 1)) 
actions = [-1, 1]  # left or right 

# Q-table initialization 
Q = np.zeros((len(positions), len(actions))) 

# Hyperparameters 
alpha = 0.1    # learning rate 
gamma = 0.9    # discount factor 
epsilon = 0.2  # exploration rate 
episodes = 20 

for episode in range(episodes): 
    state = 0  # start position 
    while state != goal: 
        # Choose action: explore or exploit 
        if random.uniform(0, 1) < epsilon: 
            action_idx = random.choice([0, 1]) 
        else: 
            action_idx = np.argmax(Q[state]) 

        action = actions[action_idx] 
        next_state = max(0, min(goal, state + action))  # stay in bounds 

        # Reward: 1 if goal reached, else 0 
        reward = 1 if next_state == goal else 0 

        # Q-learning update 
        Q[state, action_idx] = Q[state, action_idx] + alpha * ( 
            reward + gamma * np.max(Q[next_state]) - Q[state, action_idx] 
        ) 

        state = next_state 

# Display learned Q-table 
print("Learned Q-table:") 
print(Q) 

# Test the learned policy 
state = 0 
steps = [state] 
while state != goal: 
    action_idx = np.argmax(Q[state]) 
    state += actions[action_idx] 
    steps.append(state) 

print("Path taken by agent to reach goal:", steps) 
