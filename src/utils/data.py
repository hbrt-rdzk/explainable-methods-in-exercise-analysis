import os
import random

import numpy as np
import pandas as pd
import torch
from scipy.fft import dct, idct
from torch import nn
from torch.utils.data import DataLoader
from tslearn.preprocessing import TimeSeriesResampler

from src.dataset import ExerciseDataset
from src.utils.constants import DCT_COEFFICIENTS_SIZE


def get_data(
    dir: str, exercise: str, representation: str, batch_size: int = 8
) -> DataLoader:
    """Get dataloaders from csv file"""
    train_df = pd.read_csv(
        os.path.join(dir, "train", exercise, representation + ".csv")
    )
    test_df = pd.read_csv(os.path.join(dir, "test", exercise, representation + ".csv"))
    train_dataset = ExerciseDataset(train_df, representation=representation)
    test_dataset = ExerciseDataset(test_df, representation=representation)

    train_dl = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    val_dl = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    return train_dl, val_dl


def get_random_sample(
    dl: DataLoader, desired_label: str
) -> tuple[torch.Tensor, int, str]:
    """Return random sample with desired label"""
    label = None
    dl_length = len(dl.dataset)
    while label != desired_label:
        rand_idx = random.randint(0, dl_length - 1)
        label = dl.dataset.labels[rand_idx]

    return (dl.dataset.data[rand_idx], dl.dataset.lengths[rand_idx])


def joints_rep_df_to_numpy(x: pd.DataFrame) -> np.ndarray:
    """Convert joints_df of one repetition to numpy array"""
    return np.array([frame[["x", "y", "z"]] for _, frame in x.groupby("frame")])


def get_angles_from_joints(joints: np.ndarray, angles_formula: dict) -> pd.DataFrame:
    """Convert numpy array with joints to angle features"""
    angles = {}
    for angle_name, angle_joints in angles_formula.items():
        angles[angle_name] = []
        joints_3d_positions = joints[:, angle_joints]
        for frame in joints_3d_positions:
            angles[angle_name].append(calculate_3D_angle(*frame, by_axis="X"))

    return pd.DataFrame(angles)


def calculate_3D_angle(
    A: np.ndarray, B: np.ndarray, C: np.ndarray, by_axis: str = None
) -> float:
    """Calculate angle between 3 points in 3D space"""
    if not (A.shape == B.shape == C.shape == (3,)):
        raise ValueError("Input arrays must all be of shape (3,).")

    if by_axis == "X":
        axis_idx = [1, 2]
    elif by_axis == "Y":
        axis_idx = [0, 2]
    elif by_axis == "Z":
        axis_idx = [0, 1]
    else:
        axis_idx = [0, 1, 2]

    A, B, C = A[axis_idx], B[axis_idx], C[axis_idx]
    ba = A - B
    bc = C - B

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    cosine_angle = np.clip(cosine_angle, -1, 1)
    angle = np.arccos(cosine_angle)

    return np.degrees(angle)


def encode_dct(x: np.ndarray) -> np.ndarray:
    if x.shape[0] < DCT_COEFFICIENTS_SIZE:
        resampler = TimeSeriesResampler(sz=DCT_COEFFICIENTS_SIZE)
        x = np.array(resampler.fit_transform(x)).squeeze()
    return dct(x, norm="ortho")[:DCT_COEFFICIENTS_SIZE]


def decode_dct(x: np.ndarray, length) -> np.ndarray:
    decoded_signal = []
    for feature in x.transpose(1, 0):
        x_dct = np.zeros(length, dtype=float)
        x_dct[:DCT_COEFFICIENTS_SIZE] = feature
        x_idct = idct(x_dct, norm="ortho")
        decoded_signal.append(x_idct)
    return np.array(decoded_signal).transpose(1, 0)


def encode_samples_to_latent(model: nn.Module, data: torch.Tensor) -> torch.Tensor:
    model.eval()
    return model.encoder(data)[0]


def decode_samples_from_latent(model: nn.Module, data: torch.Tensor) -> torch.Tensor:
    model.eval()
    return model.decoder(data)


def segment_signal(
    x: pd.DataFrame, important_angles: list[str], sliding_window_size: int = 5
) -> pd.DataFrame:
    x = x[important_angles]
    rep_signal = x.mean(axis=1)
    # zero_point = np.mean(rep_signal)

    mid_idx = np.argmin(rep_signal)
    finish_idx = len(rep_signal) - 1

    # variances = []
    # for idx in range(0, len(rep_signal) - sliding_window_size + 1, 1):
    #     window = rep_signal[idx : idx + sliding_window_size]
    #     variances.append(np.std(window))
    # variances = np.array(variances)

    # below_mean_indexes = np.where(rep_signal < zero_point)[0]
    # above_mean_indexes = np.where(rep_signal > zero_point)[0]

    # mid_phase_idx = (
    #     below_mean_indexes[np.argmin(variances[below_mean_indexes])]
    #     + sliding_window_size // 2
    # )
    # above_mean_indexes_left = above_mean_indexes[above_mean_indexes < mid_phase_idx]
    # start_phase_idx = (
    #     above_mean_indexes_left[np.argmin(variances[above_mean_indexes_left])]
    #     + sliding_window_size // 2
    # )

    # above_mean_indexes_right = above_mean_indexes[above_mean_indexes > mid_phase_idx]
    # above_mean_indexes_right = above_mean_indexes_right[
    #     above_mean_indexes_right < len(variances)
    # ]
    # finish_phase_idx = (
    #     above_mean_indexes_right[np.argmin(variances[above_mean_indexes_right])]
    #     + sliding_window_size // 2
    # )
    return x.iloc[[0, mid_idx, finish_idx]]
