import numpy as np
from modified_policy_iteration import apply_policy_iteration
from env_reader import read_env_file

states = np.array(range(18900))
actions = np.array(range(6))
T = read_env_file(3, list(range(1, 7)))
mdp_values, arg_values = apply_policy_iteration(states, actions, T, gamma=0.99, epsilon=0.01)
print(mdp_values)
actions_by_floor = np.reshape(arg_values, (35, 540))

for fl in actions_by_floor:
    print(np.reshape(fl, (18, 30)))