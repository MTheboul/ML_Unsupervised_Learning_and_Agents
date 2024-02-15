import numpy as np
from agent import Agent

def yann_simon_policy(agent: Agent) -> str:
    epsilon = 1
    actions = ["left", "right", "none"]
    best_reward = agent.known_rewards[np.argmax(agent.known_rewards)]

    if best_reward != 0:
        epsilon = (100 - best_reward) / 100

    if np.random.uniform(0, 1) < epsilon:
        action = np.random.choice(actions)
    else:
        action = get_best_action(agent)
    return action

def get_best_action(agent: Agent):
    best_reward_pos = np.argmax(agent.known_rewards)

    if best_reward_pos == agent.position:
        return "none"
    elif best_reward_pos > agent.position:
        return "right"
    else:
        return "left"