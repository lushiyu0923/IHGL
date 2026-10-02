# IHGL

This directory contains the complete OOD missing-value imputation experiment.
Datasets stay outside the package (the default location is the repository's
`data/` directory).

## Layout

- `main_OOD.py`: command-line entry point.
- `config.py`: validated command-line configuration.
- `data/`: preprocessing, mask generation, and PyTorch data loaders.
- `models/`: IHGL and OOD environment model.
- `engine/`: training and evaluation loops.
- `utils/`: metrics and reproducibility helpers.
- `results/`: generated JSONL evaluation results (created at runtime).

## Run

From the repository root:

```bash
python IHGL/main_OOD.py --dataset luohutaxi --epoch 100 --device auto
```

For a pre-generated mask dataset:

```bash
python -m IHGL.main_OOD --dataset metrla --missing_pattern mnar \
  --missing_rate 0.1 --seed 3407 --epoch 100 --device cuda:0
```

Use `--help` to inspect all data, model, and training options. The normalized
CSV files and adjacency matrices must remain under `--data_root` and are not
copied into `IHGL`.
