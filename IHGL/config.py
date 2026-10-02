"""Command-line configuration for the IHGL OOD experiment."""

from __future__ import annotations

import argparse
from pathlib import Path


DATASET_DEFAULTS = {
    "luohutaxi": {"num_nodes": 156, "feature_len": 2976},
    "metrla": {"num_nodes": 207, "feature_len": 34272},
    "pemsbay": {"num_nodes": 325, "feature_len": 52116},
}


def build_parser() -> argparse.ArgumentParser:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Train and evaluate IHGL for OOD imputation")

    data = parser.add_argument_group("data")
    data.add_argument("--dataset", choices=sorted(DATASET_DEFAULTS), default="luohutaxi")
    data.add_argument(
        "--data_root", "--data-root", dest="data_root", type=Path, default=project_root / "data"
    )
    data.add_argument("--seq_len", "--seq-len", dest="seq_len", type=int, default=16)
    data.add_argument("--num_nodes", "--num-nodes", dest="num_nodes", type=int, default=None)
    data.add_argument("--feature", type=int, default=1)
    data.add_argument("--feature_len", "--feature-len", dest="feature_len", type=int, default=None)
    data.add_argument("--train_rate", "--train-rate", dest="train_rate", type=float, default=0.8)
    data.add_argument(
        "--missing_rate", "--missing-rate", dest="missing_rate", type=float, default=0.1
    )
    data.add_argument(
        "--missing_pattern",
        "--missing-pattern",
        dest="missing_pattern",
        choices=("mcar", "mar", "mnar"),
        default="mnar",
    )

    model = parser.add_argument_group("model")
    model.add_argument("--channels", type=int, default=128)
    model.add_argument("--timeemb", type=int, default=128)
    model.add_argument("--featureemb", type=int, default=16)
    model.add_argument("--layers", type=int, default=3)
    model.add_argument("--nheads", type=int, default=8)
    model.add_argument(
        "--missing_env_num", "--missing-env-num", dest="missing_env_num", type=int, default=3
    )
    model.add_argument("--mask_rate", "--mask-rate", dest="mask_rate", type=float, default=0.2)
    if hasattr(argparse, "BooleanOptionalAction"):
        model.add_argument(
            "--with_var_loss",
            "--with-var-loss",
            dest="with_var_loss",
            action=argparse.BooleanOptionalAction,
            default=True,
        )
    else:  # Python 3.8 compatibility
        model.add_argument(
            "--with_var_loss",
            "--with-var-loss",
            dest="with_var_loss",
            action="store_true",
            default=True,
        )
        model.add_argument("--no-with-var-loss", dest="with_var_loss", action="store_false")
    model.add_argument(
        "--var_loss_weight", "--var-loss-weight", dest="var_loss_weight", type=float, default=0.1
    )

    train = parser.add_argument_group("training")
    train.add_argument("--batch", type=int, default=16)
    train.add_argument(
        "--learning_rate", "--learning-rate", dest="learning_rate", type=float, default=1e-2
    )
    train.add_argument(
        "--weight_decay", "--weight-decay", dest="weight_decay", type=float, default=1e-6
    )
    train.add_argument("--epoch", type=int, default=1)
    train.add_argument("--seed", type=int, default=233)
    train.add_argument(
        "--device", default="auto", help="auto, cpu, or a torch device such as cuda:0"
    )
    train.add_argument(
        "--output_dir",
        "--output-dir",
        dest="output_dir",
        type=Path,
        default=project_root / "IHGL" / "results",
    )
    return parser


def parse_config(argv: list[str] | None = None):
    config = build_parser().parse_args(argv)
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "PyTorch is required to train IHGL. Install dependencies with "
            "`pip install -r IHGL/requirements.txt`."
        ) from exc

    defaults = DATASET_DEFAULTS[config.dataset]
    if config.num_nodes is None:
        config.num_nodes = defaults["num_nodes"]
    if config.feature_len is None:
        config.feature_len = defaults["feature_len"]
    if config.device == "auto":
        config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        config.device = torch.device(config.device)
    if config.device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("A CUDA device was requested, but CUDA is not available")
    if config.epoch < 1 or config.batch < 1 or config.seq_len < 1:
        raise ValueError("epoch, batch, and seq_len must be positive")
    if not 0 < config.train_rate < 1:
        raise ValueError("train_rate must be strictly between 0 and 1")
    if (
        config.channels % config.nheads != 0
        or (config.channels * config.feature) % config.nheads != 0
    ):
        raise ValueError("channels and channels * feature must be divisible by nheads")
    if config.missing_env_num < 1:
        raise ValueError("missing_env_num must be positive")
    if not 0 <= config.mask_rate <= 1:
        raise ValueError("mask_rate must be in the interval [0, 1]")
    return config
