import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time

# Constants
GRID_SIZE = 3  # 3x3 grid
ACTIONS = ["up", "down", "left", "right"]
GAMMA = 0.9  # Discount factor
THETA = 1e-6  # Convergence threshold

# Initialize the grid world
def initialize_grid(size):
    grid = np.zeros((size, size))
    return grid

# Define rewards and transitions
def get_rewards_and_transitions(size):
    rewards = np.zeros((size, size))
    transitions = {}

    # Randomly set a goal state with a high reward
    goal_state = (np.random.randint(0, size), np.random.randint(0, size))
    rewards[goal_state] = 10

    # Define transitions (deterministic for simplicity)
    for i in range(size):
        for j in range(size):
            transitions[(i, j)] = {}
            for action in ACTIONS:
                if action == "up":
                    next_state = (max(i - 1, 0), j)
                elif action == "down":
                    next_state = (min(i + 1, size - 1), j)
                elif action == "left":
                    next_state = (i, max(j - 1, 0))
                elif action == "right":
                    next_state = (i, min(j + 1, size - 1))
                transitions[(i, j)][action] = next_state

    return rewards, transitions, goal_state

# Policy Evaluation
def policy_evaluation(policy, values, rewards, transitions, gamma, theta):
    while True:
        delta = 0
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                if (i, j) == goal_state:
                    continue
                v = values[i, j]
                action = policy[i, j]
                next_state = transitions[(i, j)][action]
                values[i, j] = rewards[next_state] + gamma * values[next_state]
                delta = max(delta, abs(v - values[i, j]))
        if delta < theta:
            break
    return values

# Policy Improvement
def policy_improvement(policy, values, rewards, transitions, gamma):
    policy_stable = True
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            if (i, j) == goal_state:
                continue
            old_action = policy[i, j]
            action_values = {}
            for action in ACTIONS:
                next_state = transitions[(i, j)][action]
                action_values[action] = rewards[next_state] + gamma * values[next_state]
            policy[i, j] = max(action_values, key=action_values.get)
            if old_action != policy[i, j]:
                policy_stable = False
    return policy, policy_stable

# Visualize the grid
def visualize_grid(values, policy, goal_state):
    fig, ax = plt.subplots()
    ax.set_xticks(np.arange(GRID_SIZE))
    ax.set_yticks(np.arange(GRID_SIZE))
    ax.grid(True)

    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            if (i, j) == goal_state:
                ax.text(j, i, "Goal", ha="center", va="center", color="red")
            else:
                ax.text(j, i, f"{policy[i, j]}\n{values[i, j]:.2f}", ha="center", va="center")

    st.pyplot(fig)

# Streamlit App
st.title("Policy Iteration Animation")
st.write("This app demonstrates the Policy Iteration algorithm on a small grid world.")

# Initialize the grid world
rewards, transitions, goal_state = get_rewards_and_transitions(GRID_SIZE)
values = initialize_grid(GRID_SIZE)
policy = np.random.choice(ACTIONS, size=(GRID_SIZE, GRID_SIZE))

# Run Policy Iteration
if st.button("Run Policy Iteration"):
    policy_stable = False
    iteration = 0

    while not policy_stable:
        st.write(f"Iteration {iteration + 1}")
        values = policy_evaluation(policy, values, rewards, transitions, GAMMA, THETA)
        policy, policy_stable = policy_improvement(policy, values, rewards, transitions, GAMMA)
        visualize_grid(values, policy, goal_state)
        time.sleep(1)  # Pause for animation
        iteration += 1

    st.write("Policy Iteration converged to the optimal policy!")
