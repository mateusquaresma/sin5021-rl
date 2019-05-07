import numpy as np
from modified_policy_iteration import apply_policy_iteration
from env_reader import read_env_file
import policy_drawer as dr


epsilon = 1e-10
gammas = [0.5, 0.6, 0.7, 0.8, 0.9]
for gamma in gammas:
    states = np.array(range(135))
    actions = np.array(range(6))
    T = read_env_file(1, list(range(1, 7)))
    mdp_values, arg_values = apply_policy_iteration(states, actions, T, gamma=gamma, epsilon=epsilon)

    data = np.reshape(mdp_values, (9, 15))
    a_data = np.reshape(arg_values, (9, 15))
    # print(data)
    # print(a_data)
    # dr.draw_policy([mdp_values], [arg_values], ncols=1)