import numpy as np
from utils import compute_q_value


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
                max_for_action = compute_q_value(values, possible_transitions, gamma)
                temp = max([temp, max_for_action])

            temp_values[state] = temp

        values = temp_values
        print("----- iteration = %s -----" % (i,))
        print(np.reshape(values, grid_shape))
