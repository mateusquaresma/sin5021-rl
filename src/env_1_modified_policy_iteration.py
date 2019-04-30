import numpy as np
from modified_policy_iteration import apply_policy_iteration
from env_reader import read_env_file

states = np.array(range(135))
actions = np.array(range(6))
T = read_env_file(1, list(range(1, 7)))
mdp_values, arg_values = apply_policy_iteration(states, actions, T, gamma=0.999, epsilon=0.001)
print(mdp_values)
print(arg_values)