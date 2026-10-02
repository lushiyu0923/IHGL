"""Evaluation loop for missing-value imputation."""

from __future__ import annotations

import numpy as np
import torch

from IHGL.utils.metrics import calculate_metrics


@torch.no_grad()
def evaluate(model, test_loader, config) -> dict[str, float]:
    model.eval()
    targets, forecasts, eval_points = [], [], []
    for observed_data, observed_tp, condition_mask, gt_mask in test_loader:
        observed_data = observed_data.to(config.device)
        observed_tp = observed_tp.to(config.device)
        condition_mask = condition_mask.to(config.device)
        imputed = model.model(observed_data, observed_tp, condition_mask)
        imputed = condition_mask * observed_data + (1 - condition_mask) * imputed
        targets.append(observed_data.cpu().numpy())
        forecasts.append(imputed.cpu().numpy())
        eval_points.append((gt_mask - condition_mask).numpy())
    if not targets:
        raise ValueError("The test loader is empty")
    return calculate_metrics(
        np.concatenate(targets), np.concatenate(forecasts), np.concatenate(eval_points)
    )
