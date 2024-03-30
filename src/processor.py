import numpy as np
import pandas as pd
import os

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
            exercise_dfs = []
            for label, label_name in labels.items():
                exercise_dfs.append(
                    self.__get_df_from_frames(
                        df[df["label"] == label], poses, label_name
                    )
                )

            pd.concat(exercise_dfs).to_csv(
                os.path.join(output_dir, f"{df['exercise'].values[0].lower()}.csv"),
                index=False,
            )

    def run(self) -> None:
        self.process_data()

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
                    rep_3d_joints_x,
                    rep_3d_joints_y,
                    rep_3d_joints_z,
                    np.repeat(np.arange(frames_num, dtype=int), 25),
                    np.full_like(rep_3d_joints_x, label, dtype="<U15"),
                ]
            ).T
            final_reps_df = pd.concat(
                [
                    final_reps_df,
                    pd.DataFrame(
                        final_rep, columns=["rep", "x", "y", "z", "frame", "label"]
                    ),
                ],
                axis=0,
            )

        final_reps_df["rep"] = final_reps_df["rep"].astype("int")
        final_reps_df["frame"] = final_reps_df["frame"].astype("int")

        return final_reps_df
