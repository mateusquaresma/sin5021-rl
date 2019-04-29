import numpy as np
from iterative_policy_evaluation import evaluate
from utils import compute_q_value


def build_random_policy(states, actions, transitions):
    policy = {}
    random_actions = np.random.randint(min(actions), max(actions) + 1, len(states))

    for s, a in list(zip(states, random_actions)):
        policy[(s, a)] = transitions[(s, a)]
    return policy


def build_new_policy(transitions, argmax_values):
    policy = {}
    for s, a in enumerate(argmax_values):
        policy[(s, a)] = transitions[(s, a)]
    return policy


def apply_policy_iteration(states, actions, transitions, init_policy, goal_state=0, gamma=1.0, epsilon=0.001, grid_shape=(2, 3)):

    # max_values = np.zeros(len(states))
    # argmax_values = np.zeros(len(states))

    policy = build_random_policy(states, actions, transitions)
    policy_values = evaluate(states, policy)
    policy_actions = np.zeros(len(states))
    while True:
        new_policy_values = policy_values.copy()
        new_policy_argmax_values = policy_actions.copy()

        # tries to improve

        for state in states:
            vp = float('-inf')  # v'= v(s') best value
            ap = -1  # a' = π(s) = best action
            for action in actions:
                possible_transitions = transitions[(state, action)]
                max_for_action = compute_q_value(policy_values, possible_transitions, gamma)
                # temp = max([temp, max_for_action])
                if max_for_action > vp:
                    vp = max_for_action
                    ap = action

            print(state)
            new_policy_values[state] = vp
            new_policy_argmax_values[state] = ap

        policy_values_sum = np.sum(policy_values)
        new_policy_values_sum = np.sum(new_policy_values)

        if new_policy_values_sum > policy_values_sum:
            policy = build_new_policy(transitions, new_policy_argmax_values)
            policy_values = new_policy_values
            policy_actions = new_policy_argmax_values

        if np.abs(new_policy_values_sum - policy_values_sum) < epsilon:
            break

        # policy_values = temp_max_values
        # policy_actions = temp_argmax_values

        # print("----- iteration = %s (max) -----" % (i,))
        # print(np.reshape(max_values, grid_shape))
        # print("----- iteration = %s (argmax) -----" % (i,))
        # mapping = {0: 'N', 1: 'S', 2: 'E', 3: 'W'}
        # printable_argmax = []
        # for (idx, k) in enumerate(argmax_values):
        #     if idx == goal_state:
        #         printable_argmax.append('0')
        #     else:
        #         printable_argmax.append(mapping[k])
        # print(np.reshape(np.array(printable_argmax), grid_shape))
    return policy, policy_actions


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

# apply_policy_iteration(states, actions, T, policy_1, 4, 0.99, 0.01, (2, 5))
