import numpy as np
from value_iteration_prev import iterate

states = np.array(range(6))
actions = np.array(range(4))

T = {
    (0, 0): [(0, 1.0, -1)],
    (0, 1): [(3, 1.0, -1)],
    (0, 2): [(1, 1.0, -1)],
    (0, 3): [(0, 1.0, -1)],

    (1, 0): [(1, 1.0, -1)],
    (1, 1): [(4, 1.0, -1)],
    (1, 2): [(2, 1.0, -1)],
    (1, 3): [(0, 1.0, -1)],

    (2, 0): [(2, 1.0, -1)],
    (2, 1): [(5, 1.0, -1)],
    (2, 2): [(2, 1.0, -1)],
    (2, 3): [(1, 1.0, -1)],

    (3, 0): [(0, 1.0, -1)],
    (3, 1): [(3, 1.0, -1)],
    (3, 2): [(4, 1.0, -1)],
    (3, 3): [(3, 1.0, -1)],

    (4, 0): [(1, 0.5, -1), (3, 0.5, -1)],
    (4, 1): [(4, 0.5, -1), (3, 0.5, -1)],
    (4, 2): [(5, 0.5, -1), (3, 0.5, -1)],
    (4, 3): [(0, 1.0, -1)],

    (5, 0): [(5, 1.0, 0)],
    (5, 1): [(5, 1.0, 0)],
    (5, 2): [(5, 1.0, 0)],
    (5, 3): [(5, 1.0, 0)],
}

iterate(states, actions, T, gamma=1, n=15, grid_shape=(2, 3))
