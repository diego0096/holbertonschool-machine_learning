#!/usr/bin/env python3
"""Train monte carlo value estimation"""


def monte_carlo(env, V, policy, episodes=5000, max_steps=100,
                alpha=0.1, gamma=.99, first=False):
    """Train monte carlo value estimation"""
    success = 0
    for episode in range(episodes):
        state = env.reset()
        ep_rewards = []
        states = [state]
        for step in range(max_steps):
            action = policy(state)
            state, reward, done, info = env.step(action)
            if first and state in states:
                continue
            ep_rewards.append(reward)
            states.append(state)
            if reward > 0:
                success += 1
            if done:
                break
        total_return = 0
        for state, reward in zip(states[:-1][::-1], ep_rewards[::-1]):
            total_return = total_return * gamma + reward
            V[state] += alpha * (total_return - V[state])
    return V
