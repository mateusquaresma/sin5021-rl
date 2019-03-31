import numpy as np


def compute_value(v, transitions, gamma):
    """
    sp = s-prime (s'), future state
    pr = probability of moving to the next state, denoted by sp
    r = reward received when executing the action, might be a positive or negative number
    """
    pr_sum = 0

    for sp, pr, r in transitions:
        current = pr * (r + gamma * v[sp])
        pr_sum += current

    return pr_sum


def evaluate(states, actions, transitions, gamma=1, epsilon=0.001, grid_shape=(2, 3)):
    """
    :param states:
    :param actions:
    :param transitions:
    :param gamma: discount factor
    :param epsilon: max residual value
    :param grid_shape:
    :return:
    """
    previous_values = np.zeros(len(states))
    for i in range(50):
        current_values = np.zeros(len(states))
        for state in states:
            temp = float('-inf')
            for action in actions:
                possible_transitions = transitions[(state, action)]
                max_for_action = compute_value(previous_values, possible_transitions, gamma)
                temp = max([temp, max_for_action])

            current_values[state] = temp

        residuals = np.abs(current_values - previous_values)
        previous_values = current_values
        # print("----- iteration = %s -----" % (i,))
        # print(np.reshape(previous_values, grid_shape))

        if np.max(residuals) < epsilon:
            break

    return previous_values
