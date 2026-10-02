"""Training loop for the OOD wrapper model."""

from __future__ import annotations

import time

import numpy as np
from torch import nn, optim


def train(model: nn.Module, train_loader, config) -> list[float]:
    optimizer = optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    milestones = sorted({max(1, int(0.75 * config.epoch)), max(1, int(0.9 * config.epoch))})
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
    history: list[float] = []
    model.train()
    for epoch in range(config.epoch):
        started = time.perf_counter()
        losses = []
        for observed_data, observed_tp, condition_mask, _ in train_loader:
            observed_data = observed_data.to(config.device)
            observed_tp = observed_tp.to(config.device)
            condition_mask = condition_mask.to(config.device)
            optimizer.zero_grad(set_to_none=True)
            _, loss = model(observed_data, observed_tp, condition_mask)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        scheduler.step()
        epoch_loss = float(np.mean(losses)) if losses else float("nan")
        history.append(epoch_loss)
        print(
            f"Epoch {epoch + 1:03d}/{config.epoch} | loss={epoch_loss:.6f} | time={time.perf_counter() - started:.2f}s"
        )
    return history
