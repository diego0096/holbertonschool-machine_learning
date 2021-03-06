#!/usr/bin/env python3
"""Train a policy based Monte Carlo"""


from policy_gradient import policy_gradient
import numpy as np


def train(env, nb_episodes, alpha=0.00045, gamma=0.98, show_result=False):
    """Train a policy based Monte Carlo"""
    scores = []
    weights = np.random.rand(env.observation_space.shape[0],
                             env.action_space.n)
    for episode in range(nb_episodes):
        state = env.reset()[None, :]
        grads = []
        rewards = []
        actions = []
        done = 0
        while not done:
            if show_result and not episode % 1000:
                env.render()
            action, grad = policy_gradient(state, weights)
            state, reward, done, info = env.step(action)
            grads.append(grad)
            rewards.append(reward)
            actions.append(action)
        total_reward = 0
