import numpy as np
from utils import compute_q_value


def apply_value_iteration(states, actions, transitions, gamma=0.999, epsilon=0.001):
    """
    :param states:
    :param actions:
    :param transitions:
    :param gamma: discount factor
    :param epsilon:
    :return:
    """
    values = np.zeros(len(states))
    arg_values = np.zeros(len(states))
    while True:
        temp_values = values.copy()
        temp_arg_values = arg_values.copy()
        for state in states:
            temp = float('-inf')
            action_star = float('-inf')
            for action in actions:
                possible_transitions = transitions[(state, action)]
                q_value = compute_q_value(values, possible_transitions, gamma)

                if q_value > temp:
                    temp = q_value
                    action_star = action

            temp_values[state] = temp
            temp_arg_values[state] = action_star

        residuals = np.abs(np.sum(values) - np.sum(temp_values))
        values = temp_values
        arg_values = temp_arg_values

        if residuals < epsilon:
            break

    return values, arg_values


