"""Dataset paths, adjacency loading, and missing-mask generation.

The raw and normalized datasets intentionally live outside the ``IHGL`` package.
This module is the only place that knows their on-disk layout.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np


DATASET_SPECS: Mapping[str, dict[str, object]] = {
    "luohutaxi": {
        "data": Path("luohutaxi") / "luohutaxi_norm.csv",
        "adjacency": Path("luohutaxi") / "luohutaxi_adj.csv",
        "num_nodes": 156,
    },
    "metrla": {
        "data": Path("metr_la") / "metr_la_norm.csv",
        "adjacency": Path("metr_la") / "metr_la_adj.csv",
        "num_nodes": 207,
    },
    "pemsbay": {
        "data": Path("pemsbay") / "pems-bay_norm.csv",
        "adjacency": Path("pemsbay") / "pems_bay_adj.csv",
        "num_nodes": 325,
    },
}


def _spec(dataset: str) -> dict[str, object]:
    try:
        return DATASET_SPECS[dataset.lower()]
    except KeyError as exc:
        supported = ", ".join(sorted(DATASET_SPECS))
        raise ValueError(f"Unsupported dataset {dataset!r}; choose one of {supported}") from exc


def data_path(data_root: str | Path, dataset: str) -> Path:
    path = Path(data_root) / _spec(dataset)["data"]
    if not path.is_file():
        raise FileNotFoundError(f"Normalized data file was not found: {path}")
    return path


def adjacency_path(data_root: str | Path, dataset: str) -> Path:
    path = Path(data_root) / _spec(dataset)["adjacency"]
    if not path.is_file():
        raise FileNotFoundError(f"Adjacency file was not found: {path}")
    return path


def load_normalized_data(data_root: str | Path, dataset: str) -> np.ndarray:
    """Load a normalized CSV as a two-dimensional float32 array."""

    data = np.loadtxt(data_path(data_root, dataset), delimiter=",").astype(np.float32)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return data


def load_adjacency(data_root: str | Path, dataset: str) -> np.ndarray:
    """Load the graph adjacency matrix for ``dataset``."""

    adjacency = np.loadtxt(adjacency_path(data_root, dataset), delimiter=",").astype(np.float32)
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError(f"Adjacency matrix must be square, got {adjacency.shape}")
    return adjacency


def _mask_file(data_root: str | Path, dataset: str, pattern: str, rate: float, seed: int) -> Path:
    root = Path(data_root) / _spec(dataset)["data"].parent / "mask"
    exact = root / f"{pattern}_{rate}_{seed}.csv"
    if exact.is_file():
        return exact
    # Accommodate files generated with a trailing zero, e.g. 0.10.
    candidates = sorted(root.glob(f"{pattern}_*_{seed}.csv"))
    for candidate in candidates:
        try:
            if abs(float(candidate.name.split("_")[-2]) - rate) < 1e-12:
                return candidate
        except (ValueError, IndexError):
            continue
    raise FileNotFoundError(
        f"Missing-mask file was not found for {dataset}, pattern={pattern}, "
        f"rate={rate}, seed={seed} under {root}"
    )


def _valid_mask(data: np.ndarray) -> np.ndarray:
    return (data != -200).astype(np.float32)


def _weighted_choice(rng: np.random.Generator, weights: np.ndarray, count: int) -> np.ndarray:
    """Sample distinct indices, tolerating rows that are entirely unavailable."""

    if count <= 0:
        return np.empty(0, dtype=np.int64)
    available = np.flatnonzero(weights > 0)
    if count > available.size:
        raise ValueError("Requested more missing values than observed values")
    probabilities = weights[available].astype(np.float64)
    probabilities /= probabilities.sum()
    return rng.choice(available, size=count, replace=False, p=probabilities)


def generate_missing_mask(data: np.ndarray, pattern: str, rate: float, seed: int) -> np.ndarray:
    """Generate an MCAR, MAR, or MNAR mask without hiding true ``-200`` values."""

    if pattern not in {"mcar", "mar", "mnar"}:
        raise ValueError("missing_pattern must be one of: mcar, mar, mnar")
    if not 0 <= rate < 1:
        raise ValueError("missing_rate must be in the interval [0, 1)")

    rng = np.random.default_rng(seed)
    mask = _valid_mask(data)
    observed = np.argwhere(mask == 1)
    count = int(round(rate * len(observed)))
    if count == 0:
        return mask

    if pattern == "mcar":
        selected = rng.choice(len(observed), size=count, replace=False)
        mask[observed[selected, 0], observed[selected, 1]] = 0
        return mask

    if pattern == "mar":
        # MAR uses the first feature as the sampling signal, matching the old script.
        signal = np.nan_to_num(data[:, 0], nan=0.0)
        ranks = np.argsort(np.argsort(signal)).astype(np.float64) + 1
        weights = ranks * (mask[:, 0] > 0)
        times = _weighted_choice(rng, weights, min(count, np.count_nonzero(weights)))
        columns = rng.integers(0, data.shape[1], size=len(times))
        valid = mask[times, columns] > 0
        for time, column in zip(times[valid], columns[valid]):
            mask[time, column] = 0
        # Fill collisions deterministically from remaining observed cells.
        remaining = count - int(valid.sum())
        if remaining:
            available = np.argwhere(mask == 1)
            selected = rng.choice(
                len(available), size=min(remaining, len(available)), replace=False
            )
            mask[available[selected, 0], available[selected, 1]] = 0
        return mask

    # MNAR samples each sensor's timestamps according to its own value rank.
    # Build all column ranks once; the previous loop sorted a column for every
    # selected value and became prohibitively slow on the larger datasets.
    values = np.nan_to_num(data, nan=0.0)
    ranks = np.argsort(np.argsort(values, axis=0), axis=0).astype(np.float64) + 1
    weights = ranks * mask
    selected = _weighted_choice(rng, weights.ravel(), count)
    rows, columns = np.unravel_index(selected, mask.shape)
    mask[rows, columns] = 0
    return mask


def load_missing_mask(
    data_root: str | Path,
    dataset: str,
    pattern: str,
    rate: float,
    seed: int,
    data: np.ndarray | None = None,
) -> np.ndarray:
    """Load a pre-generated mask or create one for datasets without mask files."""

    dataset = dataset.lower()
    if data is None:
        data = load_normalized_data(data_root, dataset)
    if dataset != "luohutaxi":
        try:
            mask = np.loadtxt(_mask_file(data_root, dataset, pattern, rate, seed), delimiter=",")
            mask = np.asarray(mask, dtype=np.float32)
        except FileNotFoundError:
            # A generated mask is a useful fallback for a new seed or a custom
            # dataset copy; the checked-in masks remain preferred for exact runs.
            mask = generate_missing_mask(data, pattern, rate, seed)
    else:
        mask = generate_missing_mask(data, pattern, rate, seed)

    if data is not None and mask.shape != data.shape:
        raise ValueError(f"Mask shape {mask.shape} does not match data shape {data.shape}")
    return (mask > 0).astype(np.float32)
