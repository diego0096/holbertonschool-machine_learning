#!/usr/bin/env python3
"""3-q_learning.py"""


import numpy as np
import time


def play(env, Q, max_steps=100):
    """function that has the trained agent play an episode"""
    state = env.reset()
    done = False
    time.sleep(1)
    for step in range(max_steps):
        env.render()
        time.sleep(3.0)
        action = np.argmax(Q[state, :])
        new_state, reward, done, info = env.step(action)
        if done is True:
            env.render()
            break
        state = new_state
    env.close()
    return reward
