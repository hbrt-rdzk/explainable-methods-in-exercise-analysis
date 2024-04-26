import numpy as np
import pandas as pd
from tslearn.metrics import dtw, dtw_path


def get_dtw_score(reference: pd.DataFrame, query: pd.DataFrame) -> float:
    """Calculate dtw score between two signals"""
    return dtw(reference, query)


def get_dtw_indexes(reference: pd.DataFrame, query: pd.DataFrame) -> np.ndarray:
    """Return warped dtw from query to reference"""
    path, _ = dtw_path(reference, query)
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
