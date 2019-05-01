import numpy as np
from value_iteration import apply_value_iteration
from env_reader import read_env_file
import policy_drawer as dr

states = np.array(range(135))
actions = np.array(range(6))
T = read_env_file(1, list(range(1, 7)))
mdp_values, arg_values = apply_value_iteration(states, actions, T, epsilon=0.01, gamma=0.99)
data = np.reshape(mdp_values, (9, 15))
a_data = np.reshape(arg_values, (9, 15))
print(data)
print(a_data)

dr.draw_policy([data], ncols=1)



