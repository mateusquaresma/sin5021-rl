import csv
import pprint


def read_env_file(env_number, action_number_list):
    T = {}
    for action_number in action_number_list:
        with open('../resources/Ambiente%s/Action0%s.txt' % (env_number, action_number)) as csv_file:
            mdp_env = csv.reader(csv_file, delimiter=' ')
            for row in mdp_env:
                s, sp, pr = tuple([data for data in row if len(data) > 0])
                s = int(float(s)) - 1
                sp = int(float(sp)) - 1
                pr = float(pr)
                key = (s, action_number - 1)
                if key in T:
                    action_list = T[key]
                    action_list.append((sp, pr, -1))
                else:
                    T[key] = [(sp, pr, -1)]

    return T


transition_matrix = read_env_file(1, list(range(1, 7)))
pp = pprint.PrettyPrinter(indent=3)
pp.pprint(transition_matrix)
