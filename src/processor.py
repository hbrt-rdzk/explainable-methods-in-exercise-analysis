import logging
import os

import numpy as np
import pandas as pd
from scipy.fft import dct
from sklearn.model_selection import train_test_split

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

logger = logging.getLogger(__name__)


class Processor:
    def __init__(self, dataset: dict):
        self.dataset = dataset
        self.poses = self.dataset["poses"]
        self.labels = self.dataset["labels"]

    def process_data(self, output_dir: str) -> None:
        labels_df = pd.DataFrame(self.labels, columns=LABELS_COLUMNS)
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
            exercise_name = df["exercise"].values[0].lower()
            logger.info(f"Processing {exercise_name} exercise...")
            reps = []
            for _, rep in df.groupby(["label", "rep", "subject"]):
                reps.append(rep)
            train, test = train_test_split(
                reps, train_size=0.8, shuffle=True, random_state=42
            )
            train_df, test_df = pd.concat(train), pd.concat(test)

            for dataset_name, df in {"train": train_df, "test": test_df}.items():
                logger.info(f"Processing {dataset_name} joints...")
                joints_data = self.__process_joints(df, labels)
                self.save_data(
                    joints_data,
                    os.path.join(output_dir, dataset_name, "joints"),
                    exercise_name,
                )

                logger.info(f"Processing {dataset_name} angles...")
                angles_data = self.__process_angles(joints_data)
                self.save_data(
                    angles_data,
                    os.path.join(output_dir, dataset_name, "angles"),
                    exercise_name,
                )

                logger.info(f"Processing {dataset_name} dct coefficients...")
                dct_data = self.__process_dct(joints_data)
                self.save_data(
                    dct_data,
                    os.path.join(output_dir, dataset_name, "dct"),
                    exercise_name,
                )

        logger.info(f"Processed succesfully. Data saved in {output_dir}")

    @staticmethod
    def save_data(data: pd.DataFrame, output_dir: str, exercise_name: str) -> None:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        data.to_csv(
            os.path.join(output_dir, f"{exercise_name}.csv"),
            index=False,
        )

    def __process_joints(self, df: pd.DataFrame, labels: dict) -> pd.DataFrame:
        joints_data = []
        for label, label_name in labels.items():
            joints_data.append(
                self.__get_df_from_frames(
                    df[df["label"] == label], self.poses, label_name
                )
            )
        return pd.concat(joints_data)

    def __process_angles(self, joints_df: pd.DataFrame) -> pd.DataFrame:
        angles_data = []
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
        return pd.DataFrame(angles_data)

    def __process_dct(self, joints_df: pd.DataFrame) -> pd.DataFrame:
        dct_data = []
        for (label, rep), rep_data in joints_df.groupby(["label", "rep"]):
            for joint_name in OPENPOSE_JOINTS.values():
                joint_data = rep_data[rep_data["joint_name"] == joint_name]
                joint_data = joint_data.reset_index()
                dct_x = dct(joint_data["x"].values, norm="ortho")[:25]
                dct_y = dct(joint_data["y"].values, norm="ortho")[:25]
                dct_z = dct(joint_data["z"].values, norm="ortho")[:25]

                reps = np.full_like(dct_x, rep, dtype=int)
                dct_coefficients = np.arange(0, dct_x.shape[0], dtype=int)
                labels = np.full_like(dct_x, label, dtype="<U15")
                joint_names = np.full_like(dct_x, joint_name, dtype="<U15")

                dct_data.append(
                    pd.DataFrame(
                        {
                            "rep": reps,
                            "frame": dct_coefficients,
                            "x": dct_x,
                            "y": dct_y,
                            "z": dct_z,
                            "joint_name": joint_names,
                            "label": labels,
                        }
                    )
                )
        return pd.concat(dct_data)

    def __get_df_from_frames(
        self, df: pd.DataFrame, exercises: np.ndarray, label: str
    ) -> pd.DataFrame:
        frames = self.__get_rep_frames_from_df(df)
        final_reps_df = pd.DataFrame()
        for rep_num, frames_rep in enumerate(frames, start=1):
            rep_3d_joints = exercises[frames_rep]

            rep_3d_joints_x = rep_3d_joints[:, 0, :15].reshape(-1)
            rep_3d_joints_y = rep_3d_joints[:, 1, :15].reshape(-1)
            rep_3d_joints_z = rep_3d_joints[:, 2, :15].reshape(-1)
            frames_num = len(rep_3d_joints_x) // 15

            rep = np.full_like(rep_3d_joints_x, rep_num, dtype=int)
            frames = np.repeat(np.arange(frames_num, dtype=int), 15)
            joint_names = np.tile(list(OPENPOSE_JOINTS.values()), frames_num)
            labels = np.full_like(rep_3d_joints_x, label, dtype="<U15")
            final_rep = np.array(
                [
                    rep,
                    frames,
                    rep_3d_joints_x,
                    rep_3d_joints_y,
                    rep_3d_joints_z,
                    joint_names,
                    labels,
                ]
            ).T
            final_reps_df = pd.concat(
                [
                    final_reps_df,
                    pd.DataFrame(
                        final_rep,
                        columns=["rep", "frame", "x", "y", "z", "joint_name", "label"],
                    ),
                ],
                axis=0,
            )
        final_reps_df["rep"] = final_reps_df["rep"].astype("int")
        final_reps_df["frame"] = final_reps_df["frame"].astype("int")

        return final_reps_df

    @staticmethod
    def __get_rep_frames_from_df(df: pd.DataFrame) -> pd.Grouper:
        groups = df.groupby("subject")

        return [
            rep["frame"].values
            for _, subject_group in groups
            for _, rep in subject_group.groupby("rep")
        ]

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
