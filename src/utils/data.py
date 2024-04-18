import os
import random

import numpy as np
import pandas as pd
import torch
from scipy.fft import dct, idct
from torch import nn
from torch.utils.data import DataLoader

from src.dataset import ExerciseDataset
from src.utils.constants import DCT_COEFFICIENTS_SIZE


def get_data(
    dir: str, representation: str, exercise: str, batch_size: int = 8
) -> DataLoader:
    """Get dataloaders from csv file"""
    train_df = pd.read_csv(
        os.path.join(dir, "train", representation, exercise + ".csv")
    )
    test_df = pd.read_csv(os.path.join(dir, "test", representation, exercise + ".csv"))
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


def get_random_sample(dl: DataLoader, desired_label: int) -> tuple[torch.Tensor, int]:
    """Return random sample with desired label"""
    label = None
    dl_length = len(dl)
    while label != desired_label:
        rand_idx = random.randint(0, dl_length - 1)
        label = dl.dataset.labels_encoded[rand_idx]

    return dl.dataset.data[rand_idx], dl.dataset.lengths[rand_idx]


def joints_rep_df_to_numpy(x: pd.DataFrame) -> np.ndarray:
    """Convert joints_df of one repetition to numpy array"""
    joints = []
    for _, frame in x.groupby("frame"):
        joints.append(frame[["x", "y", "z"]])
    return np.array(joints)


def get_angles_from_joints(joints: np.ndarray, angles_formula: dict) -> pd.DataFrame:
    """Convert numpy array with joints to angle features"""
    angles = {}
    for angle_name, angle_joints in angles_formula.items():
        angles[angle_name] = []
        joints_3d_positions = joints[:, angle_joints]
        for frame in joints_3d_positions:
            angles[angle_name].append(calculate_3D_angle(*frame))

    return pd.DataFrame(angles)


def calculate_3D_angle(A: np.ndarray, B: np.ndarray, C: np.ndarray) -> float:
    """Calculate angle between 3 points in 3D space"""
    if not (A.shape == B.shape == C.shape == (3,)):
        raise ValueError("Input arrays must all be of shape (3,).")

    ba = A - B
    bc = C - B

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    cosine_angle = np.clip(cosine_angle, -1, 1)
    angle = np.arccos(cosine_angle)

    return np.degrees(angle)


def encode_dct(x: np.ndarray) -> np.ndarray:
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
