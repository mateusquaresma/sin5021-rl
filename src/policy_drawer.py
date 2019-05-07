import matplotlib.pyplot as plt
import numpy as np


def is_iterable(axes):
    iterable = True
    try:
        iter(axes)
    except TypeError:
        iterable = False
    return iterable


def draw_policy(all_floors_data, all_floors_policy, ncols=3, grid_shape=(9, 15)):
    nrows = max(1, round(len(all_floors_data) / ncols))

    print("%s, %s" % (nrows, ncols))

    fig, axes = plt.subplots(nrows, ncols)

    iterable = is_iterable(axes)

    if iterable:
        f_axes = [ax for sublist in axes for ax in sublist]
    else:
        f_axes = [axes]

    for ax in f_axes:
        ax.tick_params(
            bottom=False, top=False, left=False, right=False,
            labelleft=False, labelbottom=False)

    grid_rows, grid_cols = grid_shape

    for ax, fl_vl, fl_pl in zip(f_axes, all_floors_data, all_floors_policy):
        fl_vl = np.reshape(fl_vl, grid_shape)
        fl_pl = np.reshape(fl_pl, grid_shape)
        ax.imshow(fl_vl, cmap=plt.get_cmap('Blues_r'))
        for row in range(grid_rows):
            for col in range(grid_cols):
                if fl_vl[row][col] == 0:
                    # goal state
                    ax.text(col-0.2, row+0.2, 'G')
                elif fl_pl[row][col] == 0:
                    # north
                    ax.arrow(col, row, 0, -0.15, shape='full', head_width=.20, fc='r')
                elif fl_pl[row][col] == 1:
                    # south
                    ax.arrow(col, row, 0, 0.15, shape='full', head_width=.20, fc='r')
                elif fl_pl[row][col] == 2:
                    # east
                    ax.arrow(col, row, 0.15, 0, shape='full', head_width=.20, fc='r')
                else:
                    # west
                    ax.arrow(col, row, -0.15, 0, shape='full', head_width=.20, fc='r')

    plt.savefig('policy.png')
    plt.show()
