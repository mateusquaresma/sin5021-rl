import matplotlib.pyplot as plt
import numpy as np


def draw_policy(all_floors_data, ncols=3, grid_shape=(9, 15)):
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

    for ax, fl in zip(f_axes, all_floors_data):
        ax.imshow(np.reshape(fl, grid_shape), cmap=plt.get_cmap('Blues_r'))

    plt.show()


def is_iterable(axes):
    iterable = True
    try:
        iter(axes)
    except TypeError:
        iterable = False
    return iterable
