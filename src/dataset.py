import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

POSITION_FEATURES = ["x", "y", "z"]
ANGLE_FEATUERES = [
    "left_knee",
    "right_knee",
    "left_arm",
    "right_arm",
    "left_hip",
    "right_hip",
]


class ExerciseDataset(Dataset):
    def __init__(self, exercise_data: pd.DataFrame, representation: str = "joints"):
        if representation == "joints":
            feature_names = POSITION_FEATURES
        elif representation == "angles":
            feature_names = ANGLE_FEATUERES
        else:
            raise ValueError("Invalid representation")

        self.labels = []
        self.data = []
        for _, rep in exercise_data.groupby(["rep", "label"]):
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
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels_encoded[idx]

    @staticmethod
    def pad_batch(
        batch: list[list[torch.Tensor], torch.Tensor, int]
    ) -> list[torch.Tensor, torch.Tensor, list[int]]:
        data, labels = zip(*batch)
        original_lengths = [len(seq) for seq in data]

        data = pad_sequence(data, batch_first=True)
        labels = torch.stack(labels)
        return data, labels, original_lengths
