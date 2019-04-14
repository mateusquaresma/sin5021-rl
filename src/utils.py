

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