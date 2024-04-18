from enum import Enum


class ANGLES_FEATURES(Enum):
    SQUAT_ANGLES = ["left_knee", "right_knee", "left_hip", "right_hip"]
    LUNGES_ANGLES = ["left_knee", "right_knee", "left_hip", "right_hip"]
    PLANK_ANGLES = ["left_knee", "right_knee", "left_hip", "right_hip"]


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

DCT_COEFFICIENTS_SIZE = 25

POSITION_FEATURES = ["x", "y", "z"]
ANGLE_FEATUERES = [
    "left_knee",
    "right_knee",
    "left_arm",
    "right_arm",
    "left_hip",
    "right_hip",
]
MEAN_TIME_SERIES_LENGTH = 75


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
