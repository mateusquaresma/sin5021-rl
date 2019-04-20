
"""
# Applies the value iteration algorithm on a MDP of the following format
# [[s0 s1 s2 s3 G] => probabilistic row, every action may execute with probability of 0.5
# [s5 61 s7 s8 s9]] => deterministic row, every action executes with the probability of 1.0
#
# Rewards were fixed and not modeled in a matrix
# For each cell the following actions are possible [North, South, East, West]
"""

import numpy as np
from value_iteration import apply_value_iteration
from iterative_policy_evaluation_1 import evaluate

goal = 4
s0 = 0
states = np.array(range(10))
actions = np.array(range(4))  # North, South, East, West

# The reward must be contained inside the or it should be represented as its own matrix
# whenever the action's rewards weren't always equal to each other
reward = -1

T = {
    (0, 0): [(0, 1.0, reward)],
    (0, 1): [(5, 0.5, reward), (0, 0.5, reward)],
    (0, 2): [(1, 0.5, reward), (0, 0.5, reward)],
    (0, 3): [(0, 1.0, reward)],

    (1, 0): [(1, 1.0, reward)],
    (1, 1): [(6, 0.5, reward), (1, 0.5, reward)],
    (1, 2): [(2, 0.5, reward), (1, 0.5, reward)],
    (1, 3): [(0, 0.5, reward), (1, 0.5, reward)],

    (2, 0): [(2, 1.0, reward)],
    (2, 1): [(7, 0.5, reward), (2, 0.5, reward)],
    (2, 2): [(3, 0.5, reward), (2, 0.5, reward)],
    (2, 3): [(1, 0.5, reward), (2, 0.5, reward)],

    (3, 0): [(3, 1.0, reward)],
    (3, 1): [(8, 0.5, reward), (3, 0.5, reward)],
    (3, 2): [(4, 0.5, reward), (3, 0.5, reward)],
    (3, 3): [(2, 0.5, reward), (3, 0.5, reward)],

    (4, 0): [(4, 1.0, 0)],
    (4, 1): [(4, 1.0, 0)],
    (4, 2): [(4, 1.0, 0)],
    (4, 3): [(4, 1.0, 0)],

    (5, 0): [(0, 1.0, reward)],
    (5, 1): [(5, 1.0, reward)],
    (5, 2): [(6, 1.0, reward)],
    (5, 3): [(5, 1.0, reward)],

    (6, 0): [(1, 1.0, reward)],
    (6, 1): [(6, 1.0, reward)],
    (6, 2): [(7, 1.0, reward)],
    (6, 3): [(5, 1.0, reward)],

    (7, 0): [(2, 1.0, reward)],
    (7, 1): [(7, 1.0, reward)],
    (7, 2): [(8, 1.0, reward)],
    (7, 3): [(6, 1.0, reward)],

    (8, 0): [(3, 1.0, reward)],
    (8, 1): [(8, 1.0, reward)],
    (8, 2): [(9, 1.0, reward)],
    (8, 3): [(7, 1.0, reward)],

    (9, 0): [(4, 1.0, reward)],
    (9, 1): [(9, 1.0, reward)],
    (9, 2): [(9, 1.0, reward)],
    (9, 3): [(8, 1.0, reward)],
}

# iterate(states, actions, T, 1, 15, (2, 5))
mdp_values, arg_values = apply_value_iteration(states, actions, T, 1)
# mdp_values = evaluate(states, actions, T, 1, 0.001, (2, 5))
print(np.reshape(mdp_values, (2, 5)))
print(np.reshape(arg_values, (2, 5)))
