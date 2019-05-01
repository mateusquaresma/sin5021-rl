import numpy as np
from value_iteration import apply_value_iteration
from env_reader import read_env_file
import policy_drawer as dr

states = np.array(range(1485))
actions = np.array(range(6))
T = read_env_file(2, list(range(1, 7)))
mdp_values, arg_values = apply_value_iteration(states, actions, T, epsilon=0.01, gamma=0.99)
values_by_floor = np.reshape(mdp_values, (11, 135))
actions_by_floor = np.reshape(arg_values, (11, 135))

for fl in actions_by_floor:
    print(np.reshape(fl, (9, 15)))

dr.draw_policy(values_by_floor, grid_shape=(9, 15))
