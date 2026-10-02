"""Command-line entry point for IHGL OOD imputation."""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Allow ``python IHGL/main_OOD.py`` as well as ``python -m IHGL.main_OOD``.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from IHGL.config import parse_config


def main(argv: list[str] | None = None) -> dict[str, float]:
    config = parse_config(argv)
    from IHGL.data.datasets import create_dataloaders
    from IHGL.engine.evaluator import evaluate
    from IHGL.engine.trainer import train
    from IHGL.models.ihgl import IHGLModel_OOD
    from IHGL.utils.reproducibility import set_seed

    set_seed(config.seed)
    train_loader, test_loader = create_dataloaders(config)

    model = IHGLModel_OOD(config).to(config.device)
    train(model, train_loader, config)
    metrics = evaluate(model, test_loader, config)

    print(
        f"Dataset: {config.dataset} | missing_rate: {config.missing_rate} | "
        f"missing_pattern: {config.missing_pattern} | seed: {config.seed}"
    )
    for name, value in metrics.items():
        print(f"{name}: {value:.6f}")

    config.output_dir.mkdir(parents=True, exist_ok=True)
    result_file = config.output_dir / f"eval_result_{config.dataset}.jsonl"
    result = {
        "dataset": config.dataset,
        "missing_rate": config.missing_rate,
        "missing_pattern": config.missing_pattern,
        "seed": config.seed,
        "device": str(config.device),
        **metrics,
    }
    with result_file.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(result, ensure_ascii=True) + "\n")
    return metrics


if __name__ == "__main__":
    main()
