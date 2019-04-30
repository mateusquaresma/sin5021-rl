import numpy as np


def compute_q_value(v, transitions, gamma):
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