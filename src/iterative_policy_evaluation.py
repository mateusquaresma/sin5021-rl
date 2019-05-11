import numpy as np
from utils import compute_q_value


def evaluate(states, policy, gamma=0.99, epsilon=0.01):
    """
    :param states:
    :param actions:
    :param policy:
    :param gamma: discount factor
    :param epsilon: max residual value
    :param grid_shape:
    :return:
    """
    actual_values = np.zeros(len(states))
    for i in range(50):
        current_values = np.zeros(len(states))
        for (state, action), possible_transitions in policy.items():
            current_values[state] = compute_q_value(actual_values, possible_transitions, gamma)

        residuals = np.abs(current_values - actual_values)
        actual_values = current_values
        # print("----- iteration = %s -----" % (i,))
        # print(np.reshape(actual_values, grid_shape))

        if np.max(residuals) < epsilon:
            break

    return actual_values


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

# states = np.array(range(10))
# policy_1_values, value_1 = evaluate(states, policy_1, 0.9, 0.01)
# policy_2_values, value_2 = evaluate(states, policy_2, 0.9, 0.01)
# print(policy_1_values)
# print(value_1)
# print(policy_2_values)
# print(value_2)
