import numpy as np


def mdp_max(v, s, transitions):
    pr_sum = 0

    for fs, pr, r in transitions:
        pr_sum += pr * (r + v[fs])

    return pr_sum


def iterate(states, actions, transitions, n=10, grid_shape=(2, 3)):
    values = np.zeros(len(states))
    for i in range(n):
        temp_values = values.copy()
        for state in states:
            temp = float('-inf')
            for action in actions:
                possible_transitions = transitions[(state, action)]
                max_for_action = mdp_max(values, state, possible_transitions)
                temp = max([temp, max_for_action])

            temp_values[state] = temp

        values = temp_values
        print("----- iteration = %s -----" % (i,))
        print(np.reshape(values, grid_shape))
