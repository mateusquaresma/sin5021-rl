import numpy as np
from value_iteration import apply_value_iteration
from env_reader import read_env_file
import policy_drawer as dr

epsilon = 1e-10
gammas = [0.5, 0.6, 0.7, 0.8, 0.9]
# gammas = [0.999]
for gamma in gammas:
    states = np.array(range(135))
    actions = np.array(range(6))
    T = read_env_file(1, list(range(1, 7)))

    mdp_values, arg_values = apply_value_iteration(states, actions, T, epsilon=epsilon, gamma=gamma)
    data = np.reshape(mdp_values, (9, 15))
    a_data = np.reshape(arg_values, (9, 15))
    # print(data)
    # print(a_data)
    # dr.draw_policy([data], [a_data], ncols=1)

# dr.draw_policy([np.reshape(np.random.randn(135), (9, 15))], ncols=1)



