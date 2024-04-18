import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from tslearn.preprocessing import TimeSeriesResampler

from src.utils.constants import (ANGLE_FEATUERES, MEAN_TIME_SERIES_LENGTH,
                                 POSITION_FEATURES)


class ExerciseDataset(Dataset):
    """This class implements custom input for PyTorch's DataLoader"""
    def __init__(self, exercise_data: pd.DataFrame, representation: str = "dct"):
        if representation in ("joints", "dct"):
            feature_names = POSITION_FEATURES
        elif representation == "angles":
            feature_names = ANGLE_FEATUERES
        else:
            raise ValueError("Invalid representation")

        self.labels = []
        self.data = []
        self.lengths = []
        for _, rep in exercise_data.groupby(["rep", "label"]):
            self.lengths.append(rep["length"].values[0])
            self.labels.append(rep["label"].values[0])
            ts = []
            for _, frame in rep.groupby("frame"):
                frame_features = frame[feature_names].values.reshape(-1)
                ts.append(torch.tensor(frame_features, dtype=torch.float32))
            ts = torch.stack(ts)
            self.data.append(ts)

        label_encoder = LabelEncoder()
        self.labels_encoded = torch.tensor(
            label_encoder.fit_transform(self.labels), dtype=torch.long
        )

    def __len__(self):
        """Length of the dataset"""
        return len(self.labels)

    def __getitem__(self, idx):
        """Get data, encoded label and length of the indexed sample"""
        return self.data[idx], self.labels_encoded[idx], self.lengths[idx]

    @staticmethod
    def pad_batch(
        batch: list[list[torch.Tensor], torch.Tensor, int]
    ) -> list[torch.Tensor, torch.Tensor, list[int]]:
        """Pad batch to the longest element in batch with zeros"""
        data, labels = zip(*batch)
        original_lengths = [len(seq) for seq in data]

        data = pad_sequence(data, batch_first=True)
        labels = torch.stack(labels)
        return data, labels, original_lengths

    @staticmethod
    def resample_batch(
        batch: list[list[torch.Tensor], torch.Tensor, int]
    ) -> list[torch.Tensor, torch.Tensor, list[int]]:
        """Use interpolation to resample elements in batch to desired length"""
        resampler = TimeSeriesResampler(sz=MEAN_TIME_SERIES_LENGTH)
        data, labels = zip(*batch)
        original_lengths = [len(seq) for seq in data]

        data = torch.tensor(resampler.fit_transform(data)).float()
        labels = torch.stack(labels)
        return data, labels, original_lengths
