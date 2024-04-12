import matplotlib.animation as animation
import matplotlib.pyplot as plt
import pandas as pd
import torch
from IPython.display import HTML

X_LIM = (-0.4, 0.4)
Y_LIM = (-0.4, 0.4)
Z_LIM = (-0.4, 0.4)

ELEV = 28
AZIM = 45

OPENPOSE_CONNECTIONS = [
    (0, 1),
    (1, 2),
    (1, 5),
    (1, 8),
    (2, 3),
    (3, 4),
    (5, 6),
    (6, 7),
    (8, 9),
    (9, 10),
    (10, 11),
    (8, 12),
    (12, 13),
    (13, 14),
]

COLORS = [
    "red",
    "blue",
]


def get_3D_animation(data: torch.Tensor, color: str = "red") -> animation.FuncAnimation:
    data = data.reshape(-1, 15, 3)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    def update(i):
        ax.clear()

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

        ax.set_xlim3d(*X_LIM)
        ax.set_ylim3d(*Y_LIM)
        ax.set_zlim3d(*Z_LIM)

        ax.view_init(elev=ELEV, azim=AZIM)

        frame_data = data[i]
        ax.scatter3D(frame_data[:, 0], frame_data[:, 1], frame_data[:, 2], c=color)

        for start, stop in OPENPOSE_CONNECTIONS:
            ax.plot(
                xs=[frame_data[start, 0], frame_data[stop, 0]],
                ys=[frame_data[start, 1], frame_data[stop, 1]],
                zs=[frame_data[start, 2], frame_data[stop, 2]],
                color=color,
            )

    return animation.FuncAnimation(fig, update, frames=data.shape[0], interval=120)


def get_3D_animation_comparison(data_ref, data_query) -> animation.FuncAnimation:
    data_ref = data_ref.reshape(-1, 15, 3)
    data_query = data_query.reshape(-1, 15, 3)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    def update(i):
        ax.clear()

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

        ax.set_xlim3d(*X_LIM)
        ax.set_ylim3d(*Y_LIM)
        ax.set_zlim3d(*Z_LIM)

        ax.view_init(elev=ELEV, azim=AZIM)

        frame_data_query = data_query[i]
        frame_data_ref = data_ref[i]

        ax.scatter3D(
            frame_data_ref[:, 0], frame_data_ref[:, 1], frame_data_ref[:, 2], c="red"
        )
        ax.scatter3D(
            frame_data_query[:, 0],
            frame_data_query[:, 1],
            frame_data_query[:, 2],
            c="green",
        )

        for start, stop in OPENPOSE_CONNECTIONS:
            ax.plot(
                xs=[frame_data_ref[start, 0], frame_data_ref[stop, 0]],
                ys=[frame_data_ref[start, 1], frame_data_ref[stop, 1]],
                zs=[frame_data_ref[start, 2], frame_data_ref[stop, 2]],
                color="red",
            )
            ax.plot(
                xs=[frame_data_query[start, 0], frame_data_query[stop, 0]],
                ys=[frame_data_query[start, 1], frame_data_query[stop, 1]],
                zs=[frame_data_query[start, 2], frame_data_query[stop, 2]],
                color="green",
            )

    return animation.FuncAnimation(
        fig, update, frames=data_query.shape[0], interval=120
    )
