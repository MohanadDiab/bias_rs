# bias_rs

Research code for robust object detection in remote-sensing imagery.

## Documentation

- [Experimental action plan](docs/action_plan.md)

## Setup

```bash
pip install -e .
```

## Training

Configs live under `configs/training/`. Shared defaults are in `defaults.yaml`;
each dataset YAML overrides `imgsz` and `batch`.

```bash
python -m src.training --config configs/training/ai_tod_v1.yaml
```

Runs write to `outputs/training/<run_name>/` (gitignored). Training uses every
visible CUDA GPU automatically (Ultralytics DDP when more than one); otherwise CPU.

## Repository layout

```text
bias_rs/
├── configs/    # experiment YAML configs
├── docs/       # project plans and documentation
├── scripts/    # dataset processing and experiment utilities
├── src/        # training and future model code
├── temp/       # local analyses and visualizations
├── datasets/   # local datasets (Git-ignored)
└── outputs/    # training runs (Git-ignored)
```
