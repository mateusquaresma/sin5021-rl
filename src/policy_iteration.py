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


def iterate(states, actions, transitions, gamma=1, n=10, grid_shape=(2, 3)):

    max_values = np.zeros(len(states))
    argmax_values = np.zeros(len(states))

    for i in range(n):
        temp_max_values = max_values.copy()
        temp_argmax_values = argmax_values.copy()

        for state in states:
            vp = float('-inf')  # v'= v(s') best value
            ap = -1  # a' = π(s) = best action
            for action in actions:
                possible_transitions = transitions[(state, action)]
                max_for_action = compute_value(max_values, possible_transitions, gamma)
                # temp = max([temp, max_for_action])
                if max_for_action > vp:
                    vp = max_for_action
                    ap = action

            temp_max_values[state] = vp
            temp_argmax_values[state] = ap

        max_values = temp_max_values
        argmax_values = temp_argmax_values

        print("----- iteration = %s (max) -----" % (i,))
        print(np.reshape(max_values, grid_shape))
        print("----- iteration = %s (argmax) -----" % (i,))
        print(np.reshape(argmax_values, grid_shape))


states = np.array(range(10))
actions = np.array(range(4))  # North, South, East, West
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
    (9, 3): [(8, 1.0, reward)]
}

iterate(states, actions, T, 1, 15, (2, 5))
