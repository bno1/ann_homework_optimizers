"""
This module contains procedures to display and animate optimizers as they find
the minimum of a function.
"""

import matplotlib.pyplot as plt
import numpy as np

from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation


def plot(f, xs, ys, optimizers, frames=50, levels=50, steps_per_frame=1):
    X, Y = np.meshgrid(xs, ys)

    Z = np.array(
        [f(np.array(v), False) for v in zip(X.flatten(), Y.flatten())]
    ).reshape(X.shape)

    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax2 = fig.add_subplot(1, 2, 2)

    ax1.plot_surface(
        X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=True
    )
    ax1.set_xlim3d((xs.min(), xs.max()))
    ax1.set_ylim3d((ys.min(), ys.max()))
    # ax1.set_zlim3d((0, 100.0))

    ax2.contour(X, Y, Z, levels, cmap=cm.coolwarm)
    ax2.set_xlim((xs.min(), xs.max()))
    ax2.set_ylim((ys.min(), ys.max()))

    data_points = np.zeros((frames * steps_per_frame, len(optimizers), 3))

    for i in range(frames * steps_per_frame):
        for (j, opti) in enumerate(optimizers):
            data_points[i, j, :] = (
                opti.state[0],
                opti.state[1],
                f(opti.state, False)
            )

            opti.step()

    cmap = cm.Dark2

    opti_paths1 = [
        ax1.plot([], [], [], '-o', zorder=5, label=opti.name, color=cmap(c), markevery=[0, -1])[0]
        for (c, opti) in zip(np.linspace(0, 1, len(optimizers)), optimizers)
    ]
    opti_paths2 = [
        ax2.plot([], [], '-o', zorder=5, label=opti.name, color=cmap(c), markevery=[0, -1])[0]
        for (c, opti) in zip(np.linspace(0, 1, len(optimizers)), optimizers)
    ]

    ax1.legend()

    def update(frame, data_points, opti_paths1, opti_paths2):
        n = frame * steps_per_frame

        for i in range(len(opti_paths1)):
            opti_paths1[i].set_data(
                data_points[:n, i, 0].flatten(),
                data_points[:n, i, 1].flatten(),
            )
            opti_paths1[i].set_3d_properties(
                data_points[:n, i, 2].flatten()
            )
            opti_paths2[i].set_data(
                data_points[:n, i, 0].flatten(),
                data_points[:n, i, 1].flatten(),
            )

        return opti_paths1 + opti_paths2

    ani = FuncAnimation(
        fig, update, frames=frames, fargs=(data_points, opti_paths1, opti_paths2),
        blit=False
    )

    return ani
