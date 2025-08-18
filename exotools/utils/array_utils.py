import numpy as np


def get_gaps_indices(x: np.ndarray, greater_than_median: float) -> np.ndarray:
    time_diffs = x[1:] - x[:-1]
    exp_time = np.median(time_diffs)
    return np.argwhere(time_diffs > (exp_time * greater_than_median)).ravel()


def get_gaps_interval_indices(x: np.ndarray, greater_than_median: float) -> list[tuple[int, int]]:
    return [(i, i + 1) for i in get_gaps_indices(x=x, greater_than_median=greater_than_median)]


def get_gaps_intervals(x: np.ndarray, greater_than_median: float) -> list[tuple[float, float]]:
    return [(x[i].item(), x[i + 1].item()) for i in get_gaps_indices(x=x, greater_than_median=greater_than_median)]


def get_contiguous_interval_indices(x: np.ndarray, greater_than_median: float) -> list[tuple[int, int]]:
    gaps = get_gaps_indices(x=x, greater_than_median=greater_than_median)
    if len(gaps) == 0:
        return [(0, len(x) - 1)]

    boundaries = [-1] + gaps.tolist() + [len(x) - 1]
    return [(i1 + 1, i2) for i1, i2 in zip(boundaries[:-1], boundaries[1:])]


def get_contiguous_intervals(x: np.ndarray, greater_than_median: float) -> list[tuple[float, float]]:
    return [
        (x[i1].item(), x[i2].item())
        for i1, i2 in get_contiguous_interval_indices(x=x, greater_than_median=greater_than_median)
    ]
