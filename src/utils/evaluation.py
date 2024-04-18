import numpy as np
import pandas as pd
from tslearn.metrics import dtw, dtw_path

from src.utils.constants import ANGLES_FEATURES


def get_dtw_score(reference: pd.DataFrame, query: pd.DataFrame, exercise: str) -> float:
    """Calculate dtw score between two signals"""
    features = get_features(exercise)
    return dtw(reference[features], query[features])


def get_dtw_indexes(
    reference: pd.DataFrame, query: pd.DataFrame, exercise: str
) -> np.ndarray:
    """Return warped dtw from query to reference"""
    features = get_features(exercise)
    path, _ = dtw_path(reference[features], query[features])
    path = np.array(path)

    reference = path[:, 0]
    query = path[:, 1]
    query_to_reference_warped = filter_repetable_reference_indexes(reference, query)
    return query_to_reference_warped


def filter_repetable_reference_indexes(
    referene_to_query: np.ndarray, query_to_refernce: np.ndarray
) -> np.ndarray:
    """Filter repeateble indexes and return new list"""
    query_to_refernce_cp = query_to_refernce.copy()

    for idx in range(len(referene_to_query) - 1, -1, -1):
        if idx > 0 and referene_to_query[idx] == referene_to_query[idx - 1]:
            query_to_refernce_cp = np.delete(query_to_refernce_cp, idx)

    return query_to_refernce_cp


def get_features(exercise: str) -> list[str]:
    """Return important features for desired exercise"""
    match exercise:
        case "squat":
            return ANGLES_FEATURES.SQUAT_ANGLES.value
        case "lunges":
            return ANGLES_FEATURES.LUNGES_ANGLES.value
        case "plank":
            return ANGLES_FEATURES.PLANK_ANGLES.value
        case _:
            raise ValueError(f"Exercise {exercise} not supported")
