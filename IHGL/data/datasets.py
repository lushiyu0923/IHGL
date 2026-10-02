"""PyTorch datasets for fixed-length OOD imputation windows."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .preprocessing import load_missing_mask, load_normalized_data


class OODImputationDataset(Dataset):
    """Windowed view of one chronological split.

    The final incomplete window is dropped, as in the original implementation.
    """

    def __init__(
        self,
        data: np.ndarray,
        condition_mask: np.ndarray,
        seq_len: int,
        start: int,
        end: int,
        num_nodes: int,
        feature: int,
    ) -> None:
        if data.shape != condition_mask.shape:
            raise ValueError("Data and condition mask must have identical shapes")
        data = data[start:end]
        condition_mask = condition_mask[start:end]
        usable = (data.shape[0] // seq_len) * seq_len
        if usable == 0:
            raise ValueError("The selected split does not contain a complete sequence")
        self.data = data[:usable].reshape(usable, num_nodes, feature)
        self.condition_mask = condition_mask[:usable].reshape(usable, num_nodes, feature)
        self.gt_mask = (self.data != -200).astype(np.float32)
        self.seq_len = seq_len

    def __len__(self) -> int:
        return self.data.shape[0] // self.seq_len

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        window = slice(index * self.seq_len, (index + 1) * self.seq_len)
        observed_data = torch.from_numpy(self.data[window]).float()
        observed_tp = torch.arange(self.seq_len, dtype=torch.float32)
        condition_mask = torch.from_numpy(self.condition_mask[window]).float()
        gt_mask = torch.from_numpy(self.gt_mask[window]).float()
        return observed_data, observed_tp, condition_mask, gt_mask


def create_dataloaders(config):
    """Load data, validate dimensions, and create train/test loaders."""

    data = load_normalized_data(config.data_root, config.dataset)
    if data.shape[1] != config.num_nodes * config.feature:
        raise ValueError(
            f"{config.dataset} has {data.shape[1]} columns, expected "
            f"num_nodes * feature = {config.num_nodes * config.feature}"
        )
    config.feature_len = data.shape[0]
    mask = load_missing_mask(
        config.data_root,
        config.dataset,
        config.missing_pattern,
        config.missing_rate,
        config.seed,
        data=data,
    )
    split = int(config.train_rate * data.shape[0])
    train_dataset = OODImputationDataset(
        data, mask, config.seq_len, 0, split, config.num_nodes, config.feature
    )
    test_dataset = OODImputationDataset(
        data, mask, config.seq_len, split, data.shape[0], config.num_nodes, config.feature
    )
    train_loader = DataLoader(train_dataset, batch_size=config.batch, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=config.batch, shuffle=False, num_workers=0)
    return train_loader, test_loader
