import numpy as np
from iterative_policy_evaluation import evaluate
from value_iteration import apply_value_iteration
from utils import build_new_policy
from utils import build_random_policy


def apply_policy_iteration(states, actions, transitions, gamma=0.999, epsilon=0.001):

    max_values = np.zeros(len(states))
    argmax_values = np.zeros(len(states))

    # evaluate initial policy - evaluate(states, policy_2, 0.9, 0.01, (2, 5))
    policy = build_random_policy(states, actions, transitions)
    policy_actions = np.zeros(len(states))
    count = 0
    while True:
        count += 1
        tmp_max_values = max_values.copy()
        tmp_max_values.fill(float('-inf'))
        tmp_argmax_values = argmax_values.copy()
        tmp_argmax_values.fill(float('-inf'))

        # tries to improve

        max_values, argmax_values = apply_value_iteration(states, actions, transitions, gamma=gamma, epsilon=epsilon)

        policy_values = evaluate(states, policy, gamma=gamma, epsilon=epsilon)
        new_policy = build_new_policy(transitions, argmax_values)
        new_policy_values = evaluate(states, new_policy, gamma=gamma, epsilon=epsilon)

        policy_values_sum = np.sum(policy_values)
        new_policy_values_sum = np.sum(new_policy_values)

        if new_policy_values_sum > policy_values_sum:
            policy = new_policy
            policy_actions = argmax_values

        if np.abs(new_policy_values_sum - policy_values_sum) < epsilon:
            break

    print("convergence in %s iterations" % (count, ))
    return policy, policy_actions


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
    (9, 3): [(8, 1.0, reward)],
}

states = np.array(range(10))
actions = np.array(range(4))
v, p_actions = apply_policy_iteration(states, actions, T)
pvs = evaluate(states, v)
print(v)
print(np.reshape(p_actions, (2, 5)))
print(np.reshape(pvs, (2, 5)))
