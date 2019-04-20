import numpy as np
from iterative_policy_evaluation import evaluate
from utils import compute_q_value
from value_iteration import apply_value_iteration


def build_new_policy(argmax_values):
    policy = {}
    for s, a in enumerate(argmax_values):
        policy[(s, a)] = T[(s, a)]
    return policy


def iterate(states, actions, transitions, init_policy, epsilon=0.001):

    max_values = np.zeros(len(states))
    argmax_values = np.zeros(len(states))

    # evaluate initial policy - evaluate(states, policy_2, 0.9, 0.01, (2, 5))
    policy = init_policy
    policy_values = evaluate(states, policy)
    policy_actions = []
    while True:

        tmp_max_values = max_values.copy()
        tmp_max_values.fill(float('-inf'))
        tmp_argmax_values = argmax_values.copy()
        tmp_argmax_values.fill(float('-inf'))

        # tries to improve

        # for state in states:
        #     for action in actions:
        #         possible_transitions = transitions[(state, action)]
        #         q_value = compute_q_value(max_values, possible_transitions, 0.9)
        #
        #         if q_value > tmp_max_values[state]:
        #             tmp_max_values[state] = q_value
        #             tmp_argmax_values[state] = action

        max_values, argmax_values = apply_value_iteration(states, actions, transitions)

        new_policy = build_new_policy(argmax_values)
        new_policy_values = evaluate(states, new_policy)

        if np.sum(new_policy_values) > np.sum(policy_values):
            policy = new_policy
            policy_values = new_policy_values
            policy_actions = argmax_values

        if np.abs(np.sum(new_policy_values) - np.sum(policy_values)) < epsilon:
            break

    return policy, policy_actions


reward = -1
policy_1 = {
    (0, 2): [(1, 0.5, reward), (0, 0.5, reward)],
    (1, 2): [(2, 0.5, reward), (1, 0.5, reward)],
    (2, 2): [(3, 0.5, reward), (2, 0.5, reward)],
    (3, 2): [(4, 0.5, reward), (3, 0.5, reward)],
    (4, 0): [(4, 1.0, 0)],
    (5, 2): [(6, 1.0, reward)],
    (6, 2): [(7, 1.0, reward)],
    (7, 2): [(8, 1.0, reward)],
    (8, 2): [(9, 1.0, reward)],
    (9, 0): [(4, 1.0, reward)]
}

policy_2 = {
    (0, 1): [(5, 0.5, reward), (0, 0.5, reward)],
    (1, 2): [(2, 0.5, reward), (1, 0.5, reward)],
    (2, 2): [(3, 0.5, reward), (2, 0.5, reward)],
    (3, 2): [(4, 0.5, reward), (3, 0.5, reward)],
    (4, 0): [(4, 1.0, 0)],
    (5, 2): [(6, 1.0, reward)],
    (6, 2): [(7, 1.0, reward)],
    (7, 2): [(8, 1.0, reward)],
    (8, 2): [(9, 1.0, reward)],
    (9, 0): [(4, 1.0, reward)]
}

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

states = np.array(range(10))
actions = np.array(range(4))
v, p_actions = iterate(states, actions, T, policy_1)
pvs = evaluate(states, v)
print(v)
print(np.reshape(p_actions, (2, 5)))
print(np.reshape(pvs, (2, 5)))
