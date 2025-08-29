# streamlit_qlearning_grid_obstacles_toggle.py 

import numpy as np 
import random 
import matplotlib.pyplot as plt 
import streamlit as st 
import time 

# Grid size 
grid_size = 6 
start_state = (0, 0) 
goal_state = (5, 5) 

actions = ["U", "D", "L", "R"] 

# Define obstacles (cells agent cannot enter) 
obstacles = {(1, 1), (2, 2), (3, 1), (4, 3), (2, 4)} 

# Q-table initialization 
Q = {} 
for row in range(grid_size): 
    for col in range(grid_size): 
        if (row, col) not in obstacles: 
            Q[(row, col)] = {a: 0.0 for a in actions} 

# Hyperparameters 
alpha = 0.1 
gamma = 0.9 
epsilon = 0.2 
episodes = 500 

# Reward function 
def get_reward(state): 
    if state == goal_state: 
        return 20 
    elif state in obstacles: 
        return -10 
    else: 
        return -1 

# Step function 
def step(state, action): 
    row, col = state 
    if action == "U" and row > 0: 
        next_state = (row - 1, col) 
    elif action == "D" and row < grid_size - 1: 
        next_state = (row + 1, col) 
    elif action == "L" and col > 0: 
        next_state = (row, col - 1) 
    elif action == "R" and col < grid_size - 1: 
        next_state = (row, col + 1) 
    else: 
        next_state = state 

    # Block obstacle movement 
    if next_state in obstacles: 
        next_state = state 
    return next_state 

# Training Q-learning 
def train_qlearning(): 
    for ep in range(episodes): 
        state = start_state 
        while state != goal_state: 
            # ε-greedy policy 
            if random.uniform(0, 1) < epsilon: 
                action = random.choice(actions) 
            else: 
                action = max(Q[state], key=Q[state].get) 

            next_state = step(state, action) 
            reward = get_reward(next_state) 

            Q[state][action] += alpha * ( 
                reward + gamma * max(Q[next_state].values()) - Q[state][action] 
            ) 

            state = next_state 

# Path with trained agent 
def get_trained_path(): 
    state = start_state 
    path = [state] 
    visited = set() 
    while state != goal_state and len(path) < 100: 
        action = max(Q[state], key=Q[state].get) 
        state = step(state, action) 
        if state in visited:  # prevent infinite loop 
            break 
        path.append(state) 
        visited.add(state) 
    return path 

# Path with random agent 
def get_random_path(): 
    state = start_state 
    path = [state] 
    for _ in range(50):  # limit random moves 
        action = random.choice(actions) 
        state = step(state, action) 
        path.append(state) 
        if state == goal_state: 
            break 
    return path 

# Draw the grid 
def draw_grid(state, path): 
    fig, ax = plt.subplots(figsize=(6, 6)) 
    ax.set_xticks(np.arange(0, grid_size + 1, 1)) 
    ax.set_yticks(np.arange(0, grid_size + 1, 1)) 
    ax.grid(True) 
    ax.set_xlim(0, grid_size) 
    ax.set_ylim(0, grid_size) 
    ax.invert_yaxis() 

    # Obstacles 
    for (r, c) in obstacles: 
        ax.add_patch(plt.Rectangle((c, r), 1, 1, color="black")) 

    # Start + Goal 
    ax.text(start_state[1] + 0.5, start_state[0] + 0.5, "S", ha="center", va="center", fontsize=16) 
    ax.text(goal_state[1] + 0.5, goal_state[0] + 0.5, "   ", ha="center", va="center", fontsize=16) 

    # Path trace 
    for p in path: 
        ax.plot(p[1] + 0.5, p[0] + 0.5, "bo", markersize=10, alpha=0.3) 

    # Agent 
    ax.plot(state[1] + 0.5, state[0] + 0.5, "ro", markersize=20) 

    return fig 

# Streamlit UI 
st.title("Reinforcement Learning - Grid World with Obstacles") 
st.write("Compare how a **Trained Agent** vs a **Random Agent** behaves.") 

mode = st.radio("Choose Agent Mode:", ["Trained Agent", "Random Agent"]) 

if st.button("Run Simulation"): 
    if mode == "Trained Agent": 
        train_qlearning() 
        path = get_trained_path() 
    else: 
        path = get_random_path() 

    placeholder = st.empty() 
    for i, state in enumerate(path): 
        fig = draw_grid(state, path[:i + 1]) 
        placeholder.pyplot(fig) 
        time.sleep(0.5) 

    if path[-1] == goal_state: 
        st.success("Agent reached the goal! ") 
    else: 
        st.error("Agent did not reach the goal.") 
