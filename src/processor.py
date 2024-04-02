import os

import numpy as np
import pandas as pd

LABELS_COLUMNS = ["exercise", "subject", "label", "rep", "frame"]

SQUAT_LABELS = {
    1: "correct",
    2: "feet_too_wide",
    3: "knees_inwards",
    4: "not_low_enough",
    5: "front_bend",
}
LUNGES_LABELS = {
    1: "correct",
    4: "not_low_enough",
    6: "knee_passes_toe",
}
PLANK_LABELS = {
    1: "correct",
    7: "arched_back",
    8: "hunched_back",
}

OPENPOSE_JOINTS = {
    0: "nose",
    1: "upper_spine",
    2: "right_shoulder",
    3: "right_arm",
    4: "right_wrist",
    5: "left_shoulder",
    6: "left_arm",
    7: "left_wrist",
    8: "lower_spine",
    9: "right_hip",
    10: "right_knee",
    11: "right_foot",
    12: "left_hip",
    13: "left_knee",
    14: "left_foot",
}

OPENPOSE_ANGLES = {
    "left_knee": [14, 13, 12],
    "right_knee": [11, 10, 9],
    "right_arm": [4, 3, 2],
    "left_arm": [7, 6, 5],
    "left_hip": [13, 12, 5],
    "right_hip": [10, 9, 2],
}


class Processor:
    def __init__(self, dataset: dict):
        self.dataset = dataset

    def process_data(self, output_dir: str) -> None:
        poses = self.dataset["poses"]
        labels_df = pd.DataFrame(self.dataset["labels"], columns=LABELS_COLUMNS)
        labels_df["frame"] = np.arange(len(labels_df))
        labels_df[LABELS_COLUMNS[2:]] = labels_df[LABELS_COLUMNS[2:]].astype("int")

        squat_df = labels_df[labels_df["exercise"] == "SQUAT"]
        lunges_df = labels_df[labels_df["exercise"] == "Lunges"]
        plank_df = labels_df[labels_df["exercise"] == "Plank"]

        for df, labels in [
            (squat_df, SQUAT_LABELS),
            (lunges_df, LUNGES_LABELS),
            (plank_df, PLANK_LABELS),
        ]:
            joints_data = []
            angles_data = []

            for label, label_name in labels.items():
                joints_data.append(
                    self.__get_df_from_frames(
                        df[df["label"] == label], poses, label_name
                    )
                )
            joints_df = pd.concat(joints_data)
            joints_df.to_csv(
                os.path.join(
                    output_dir, "joints", f"{df['exercise'].values[0].lower()}.csv"
                ),
                index=False,
            )
            for (label, rep, frame), rep_data in joints_df.groupby(
                ["label", "rep", "frame"]
            ):
                rep_data = rep_data.reset_index()
                angles = {}
                for angle_name, angle_joints in OPENPOSE_ANGLES.items():
                    joints_3d_positions = rep_data.loc[angle_joints][
                        ["x", "y", "z"]
                    ].astype("float")
                    angles[angle_name] = self.__calculate_3D_angle(
                        *joints_3d_positions.values
                    )
                angles_data.append(
                    pd.Series(
                        {
                            "rep": rep,
                            "frame": frame,
                            **angles,
                            "label": label,
                        }
                    )
                )
            angles_df = pd.DataFrame(angles_data)
            angles_df.to_csv(
                os.path.join(
                    output_dir, "angles", f"{df['exercise'].values[0].lower()}.csv"
                ),
                index=False,
            )

    def __get_rep_frames_from_df(self, df: pd.DataFrame) -> pd.Grouper:
        groups = df.groupby("subject")

        return [
            rep["frame"].values
            for _, subject_group in groups
            for _, rep in subject_group.groupby("rep")
        ]

    def __get_df_from_frames(
        self, df: pd.DataFrame, exercises: np.ndarray, label: str
    ) -> pd.DataFrame:
        frames = self.__get_rep_frames_from_df(df)
        final_reps_df = pd.DataFrame()
        for rep_num, frames_rep in enumerate(frames, start=1):
            rep_3d_joints = exercises[frames_rep]

            rep_3d_joints_x = rep_3d_joints[:, 0, :].reshape(-1)
            rep_3d_joints_y = rep_3d_joints[:, 1, :].reshape(-1)
            rep_3d_joints_z = rep_3d_joints[:, 2, :].reshape(-1)
            frames_num = len(rep_3d_joints_x) // 25
            final_rep = np.array(
                [
                    np.full_like(rep_3d_joints_x, rep_num, dtype=int),
                    np.repeat(np.arange(frames_num, dtype=int), 25),
                    rep_3d_joints_x,
                    rep_3d_joints_y,
                    rep_3d_joints_z,
                    np.full_like(rep_3d_joints_x, label, dtype="<U15"),
                ]
            ).T
            final_reps_df = pd.concat(
                [
                    final_reps_df,
                    pd.DataFrame(
                        final_rep, columns=["rep", "frame", "x", "y", "z", "label"]
                    ),
                ],
                axis=0,
            )

        final_reps_df["rep"] = final_reps_df["rep"].astype("int")
        final_reps_df["frame"] = final_reps_df["frame"].astype("int")

        return final_reps_df

    @staticmethod
    def __calculate_3D_angle(A: np.ndarray, B: np.ndarray, C: np.ndarray) -> float:
        if not (A.shape == B.shape == C.shape == (3,)):
            raise ValueError("Input arrays must all be of shape (3,).")

        ba = A - B
        bc = C - B

        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        cosine_angle = np.clip(cosine_angle, -1, 1)
        angle = np.arccos(cosine_angle)

        return np.degrees(angle)
