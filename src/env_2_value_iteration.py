import numpy as np
from value_iteration import apply_value_iteration
from env_reader import read_env_file
import policy_drawer as dr


epsilon = 1e-10
gammas = [0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
for gamma in gammas:
    states = np.array(range(1485))
    actions = np.array(range(6))
    T = read_env_file(2, list(range(1, 7)))
    mdp_values, arg_values = apply_value_iteration(states, actions, T, epsilon=epsilon, gamma=gamma)
    values_by_floor = np.reshape(mdp_values, (11, 135))
    actions_by_floor = np.reshape(arg_values, (11, 135))

    # for fl in actions_by_floor:
    #     print(np.reshape(fl, (9, 15)))
    #
    # dr.draw_policy(values_by_floor, actions_by_floor, grid_shape=(9, 15), file_name_prefix='env-2')
