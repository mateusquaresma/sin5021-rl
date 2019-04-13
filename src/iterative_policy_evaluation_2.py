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


def evaluate(states, actions, gamma=0.9, epsilon=0.001, grid_shape=(2, 3)):
    """
    :param states:
    :param actions:
    :param transitions:
    :param gamma: discount factor
    :param epsilon: max residual value
    :param grid_shape:
    :return:
    """

    reward = -1
    policy = {
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

    policies = [policy, policy_2]

    best_policy = {}
    best_policy_value = float('-inf')

    for current_policy in policies:
        previous_values = np.zeros(len(states))
        for i in range(50):
            current_values = np.zeros(len(states))
            for state in states:
                temp = float('-inf')
                for action in actions:
                    try:
                        possible_transitions = current_policy[(state, action)]
                        max_for_action = compute_value(previous_values, possible_transitions, gamma)
                        temp = max([temp, max_for_action])
                    except KeyError:
                        continue

                current_values[state] = temp

            residuals = np.abs(current_values - previous_values)
            previous_values = current_values
            # print("----- iteration = %s -----" % (i,))
            # print(np.reshape(previous_values, grid_shape))

            if np.max(residuals) < epsilon:
                break

        current_policy_value = np.sum(previous_values)
        if current_policy_value > best_policy_value:
            best_policy_value = current_policy_value
            best_policy = current_policy

    return best_policy, best_policy_value


states = np.array(range(10))
actions = np.array(range(4))  # North, South, East, West
best_policy, best_policy_value = evaluate(states, actions, 0.9, 0.01, (2, 5))
print(best_policy_value)
print(best_policy)
