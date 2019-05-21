import numpy as np
from queue import PriorityQueue
from utils import compute_q_value


def apply(states, actions, transitions, gamma=0.9, epsilon=0.1):
    q = PriorityQueue()
    q.put(7)
    q.put(5)
    q.put(3)

    while not q.empty():
        print(q.get())

    values = np.zeros(len(states))
    arg_values = np.zeros(len(states))
    count = 0

    for s in states:
        q.put((0, s))

    while not q.empty():
        temp_values = values.copy()
        temp_arg_values = arg_values.copy()

        state = q.get()
        temp = values[state]
        action_star = float('-inf')
        for action in actions:
            try:
                possible_transitions = transitions[(state, action)]
                q_value = compute_q_value(values, possible_transitions, gamma)

                if q_value > temp:
                    temp = q_value
                    action_star = action
            except KeyError:
                pass

        temp_values[state] = temp
        temp_arg_values[state] = action_star

        residuals = np.abs(np.sum(values) - np.sum(temp_values))
        values = temp_values
        arg_values = temp_arg_values
        count += 1

        if residuals < epsilon:
            break
        # else:
        #     print("residual = %s; epsilon=%s" % (residuals, epsilon))

    print('finished after %s iterations; gamma=%s, epsilon=%s' % (count, gamma, epsilon))
    return values, arg_values