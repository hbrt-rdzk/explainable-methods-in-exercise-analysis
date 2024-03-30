import os
from torch.utils.data import Dataset
import pandas as pd


POSITION_FEATURES = ["x", "y", "z"]
ANGLE_FEATUERES = [
    "left_knee",
    "right_knee",
    "left_elbow",
    "right_elbow",
    "left_shoulder",
    "right_shoulder",
    "left_hip",
    "right_hip",
]


class ExerciseDataset(Dataset):
    def __init__(self, exercise_folder: str, representation: str = "joints"):
        if representation == "joints":
            feature_names = POSITION_FEATURES
        elif representation == "angles":
            feature_names = ANGLE_FEATUERES
        else:
            raise ValueError("Invalid representation")

        self.exercise_folder = exercise_folder
        self.labels = []
        self.data = []
        for file in os.listdir(exercise_folder):
            data = pd.read_csv(os.path.join(exercise_folder, file))
            for _, rep in data.groupby("rep"):
                self.labels.append(rep["label"].values[0])
                ts = []
                for _, frame in rep.groupby("frame"):
                    frame_features = frame[feature_names].values.reshape(-1)
                    ts.append(frame_features)
                self.data.append(ts)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
