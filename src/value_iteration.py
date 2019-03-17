import numpy as np


def compute_value(v, transitions, gamma):
    """
    sp = s-prime (s'), future state
    pr = probability of moving to the next state, denoted by sp
    r = reward received when executing the action, might be a positive or negative number
    """
    pr_sum = 0

    for sp, pr, r in transitions:
        pr_sum += pr * (r + gamma * v[sp])

    return pr_sum


def iterate(states, actions, transitions, gamma=1, n=10, grid_shape=(2, 3)):
    """
    :param states:
    :param actions:
    :param transitions:
    :param gamma: discount factor
    :param n: number of iterations
    :param grid_shape:
    :return:
    """
    values = np.zeros(len(states))
    for i in range(n):
        temp_values = values.copy()
        for state in states:
            temp = float('-inf')
            for action in actions:
                possible_transitions = transitions[(state, action)]
                max_for_action = compute_value(values, possible_transitions, gamma)
                temp = max([temp, max_for_action])

            temp_values[state] = temp

        values = temp_values
        print("----- iteration = %s -----" % (i,))
        print(np.reshape(values, grid_shape))
