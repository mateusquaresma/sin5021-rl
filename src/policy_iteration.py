import numpy as np
from iterative_policy_evaluation import evaluate
from utils import compute_q_value
from utils import build_new_policy
from utils import build_random_policy


def apply_policy_iteration(states, actions, transitions, gamma=0.999, epsilon=0.001):

    # max_values = np.zeros(len(states))
    # argmax_values = np.zeros(len(states))

    policy = build_random_policy(states, actions, transitions)
    policy_values = evaluate(states, policy, gamma=gamma, epsilon=epsilon)
    policy_actions = np.zeros(len(states))
    count = 0
    while True:
        count += 1
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

            new_policy_values[state] = vp
            new_policy_argmax_values[state] = ap

        policy_values = evaluate(states, policy, gamma=gamma, epsilon=epsilon)
        new_policy = build_new_policy(transitions, new_policy_argmax_values)
        new_policy_values = evaluate(states, new_policy, gamma=gamma, epsilon=epsilon)

        policy_values_sum = np.sum(policy_values)
        new_policy_values_sum = np.sum(new_policy_values)

        if new_policy_values_sum > policy_values_sum:
            policy = new_policy
            policy_values = new_policy_values
            policy_actions = new_policy_argmax_values

        if np.abs(new_policy_values_sum - policy_values_sum) < epsilon:
            break
    print("convergence in %s iterations" % (count, ))
    return policy, policy_actions


# states = np.array(range(10))
# actions = np.array(range(4))  # North, South, East, West
# reward = -1
#
# T = {
#     (0, 0): [(0, 1.0, reward)],
#     (0, 1): [(5, 0.5, reward), (0, 0.5, reward)],
#     (0, 2): [(1, 0.5, reward), (0, 0.5, reward)],
#     (0, 3): [(0, 1.0, reward)],
#
#     (1, 0): [(1, 1.0, reward)],
#     (1, 1): [(6, 0.5, reward), (1, 0.5, reward)],
#     (1, 2): [(2, 0.5, reward), (1, 0.5, reward)],
#     (1, 3): [(0, 0.5, reward), (1, 0.5, reward)],
#
#     (2, 0): [(2, 1.0, reward)],
#     (2, 1): [(7, 0.5, reward), (2, 0.5, reward)],
#     (2, 2): [(3, 0.5, reward), (2, 0.5, reward)],
#     (2, 3): [(1, 0.5, reward), (2, 0.5, reward)],
#
#     (3, 0): [(3, 1.0, reward)],
#     (3, 1): [(8, 0.5, reward), (3, 0.5, reward)],
#     (3, 2): [(4, 0.5, reward), (3, 0.5, reward)],
#     (3, 3): [(2, 0.5, reward), (3, 0.5, reward)],
#
#     (4, 0): [(4, 1.0, 0)],
#     (4, 1): [(4, 1.0, 0)],
#     (4, 2): [(4, 1.0, 0)],
#     (4, 3): [(4, 1.0, 0)],
#
#     (5, 0): [(0, 1.0, reward)],
#     (5, 1): [(5, 1.0, reward)],
#     (5, 2): [(6, 1.0, reward)],
#     (5, 3): [(5, 1.0, reward)],
#
#     (6, 0): [(1, 1.0, reward)],
#     (6, 1): [(6, 1.0, reward)],
#     (6, 2): [(7, 1.0, reward)],
#     (6, 3): [(5, 1.0, reward)],
#
#     (7, 0): [(2, 1.0, reward)],
#     (7, 1): [(7, 1.0, reward)],
#     (7, 2): [(8, 1.0, reward)],
#     (7, 3): [(6, 1.0, reward)],
#
#     (8, 0): [(3, 1.0, reward)],
#     (8, 1): [(8, 1.0, reward)],
#     (8, 2): [(9, 1.0, reward)],
#     (8, 3): [(7, 1.0, reward)],
#
#     (9, 0): [(4, 1.0, reward)],
#     (9, 1): [(9, 1.0, reward)],
#     (9, 2): [(9, 1.0, reward)],
#     (9, 3): [(8, 1.0, reward)]
# }
#
# reward = -1
# policy_1 = {
#     (0, 2): [(1, 0.5, reward), (0, 0.5, reward)],
#     (1, 2): [(2, 0.5, reward), (1, 0.5, reward)],
#     (2, 2): [(3, 0.5, reward), (2, 0.5, reward)],
#     (3, 2): [(4, 0.5, reward), (3, 0.5, reward)],
#     (4, 0): [(4, 1.0, 0)],
#     (5, 2): [(6, 1.0, reward)],
#     (6, 2): [(7, 1.0, reward)],
#     (7, 2): [(8, 1.0, reward)],
#     (8, 2): [(9, 1.0, reward)],
#     (9, 0): [(4, 1.0, reward)]
# }

# apply_policy_iteration(states, actions, T, 0.99, 0.01)
