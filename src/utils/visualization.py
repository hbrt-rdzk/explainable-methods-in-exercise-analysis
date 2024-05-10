import os

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.utils.constants import OPENPOSE_CONNECTIONS

X_LIM = (-0.4, 0.4)
Y_LIM = (-0.4, 0.4)
Z_LIM = (-0.4, 0.4)

ELEV = 28
AZIM = 30


def get_3D_animation(
    data: torch.Tensor, color: str = "red", is_plank: bool = False
) -> animation.FuncAnimation:
    """Return animation of the joints representation in time"""
    data = data.reshape(-1, 15, 3)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    if is_plank:
        data = data[..., [0, 2, 1]]
        data[..., 2] = -data[..., 2]

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

    return animation.FuncAnimation(fig, update, frames=data.shape[0], interval=90)


def get_3D_animation_comparison(
    data_ref: np.ndarray, data_query: np.ndarray, label: str, is_plank: bool = False
) -> animation.FuncAnimation:
    """Return animation of incorrect and fixed joints representations in time"""
    data_ref = data_ref.reshape(-1, 15, 3)
    data_query = data_query.reshape(-1, 15, 3)

    if is_plank:
        data_query = data_query[..., [0, 2, 1]]
        data_query[..., 2] = -data_query[..., 2]

        data_ref = data_ref[..., [0, 2, 1]]
        data_ref[..., 2] = -data_ref[..., 2]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    def update(i):
        ax.clear()
        ax.set_title("Error label: " + " ".join(label.split("_")))

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
            frame_data_ref[:, 0],
            frame_data_ref[:, 1],
            frame_data_ref[:, 2],
            c="red",
            label="Original",
        )
        ax.scatter3D(
            frame_data_query[:, 0],
            frame_data_query[:, 1],
            frame_data_query[:, 2],
            c="green",
            label="Corrected",
        )
        ax.legend()

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

    return animation.FuncAnimation(fig, update, frames=data_query.shape[0], interval=90)


def save_anim(anim: animation.FuncAnimation, path: str) -> None:
    os.makedirs("/".join(path.split("/")[:-1]), exist_ok=True)
    anim.save(path, writer="ffmpeg")
