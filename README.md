# bias_rs

Research code for bias-aware object detection on remote-sensing imagery. Python 3.12 only. Package manager is [uv](https://docs.astral.sh/uv/).

The experimental writeup is in [docs/action_plan.md](docs/action_plan.md). This file is how to install, test, and run the repo.

## Install uv

On Linux or macOS:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

On Windows (PowerShell):

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Confirm:

```bash
uv --version
```

You need a Python 3.12 interpreter. uv can fetch one:

```bash
uv python install 3.12
```

## Clone and sync

```bash
git clone https://github.com/mohanadDiab/bias_rs.git
cd bias_rs
uv sync
```

`uv sync` creates `.venv`, installs the package in editable mode, and pulls the `dev` group (pytest). Detector backends (Ultralytics, RF-DETR, OpenMIM/MMDet, mmcv-lite) are required; there are no optional extras.

Activate the venv if you want a normal shell instead of `uv run`:

```bash
# Linux / macOS
source .venv/bin/activate

# Windows
.venv\Scripts\Activate.ps1
```

## Tests

From the repo root:

```bash
uv run pytest
```

Quieter:

```bash
uv run pytest -q
```

One file:

```bash
uv run pytest tests/test_pipeline.py tests/test_config_matrix.py -q
```

These tests check config schemas, pipeline YAML expansion, fusion/calibration, and noise math. They do not download datasets or train models.

## Configs

YAML lives under `configs/`. Three layers:

| Directory | Role |
|-----------|------|
| `configs/training/` | Per-dataset train settings (`imgsz`, `batch`, splits). Merged with `configs/training/defaults.yaml`. Expert backends are snippets in `configs/training/models/`. |
| `configs/data/` | Dataset roots and annotation folders. Used by `bias-prepare` and `scripts/data/generate_noise.py`. |
| `configs/pipelines/` | Full detect-and-combine recipes. Merged with `configs/pipelines/defaults.yaml`. |

Training datasets currently wired:

- `configs/training/ai_tod_v1.yaml` / `ai_tod_v2.yaml`
- `configs/training/dota_1024_v1.yaml` / `dota_1024_v15.yaml`
- `configs/training/hit_uav.yaml`
- `configs/training/hrsc2016_ms_640.yaml`
- `configs/training/plant_detection_640.yaml`

`cal_split` is `test` when the dataset has a test split. DOTA 1024 and Plant Detection 640 have train/val only, so they use a carved `cal` split.

Sanity-check a training YAML (loads defaults, resolves paths, fails on missing keys):

```bash
uv run python -c "from src.training.config import load_config; print(load_config('configs/training/hit_uav.yaml')['dataset'])"
```

Expand a pipeline without running it:

```bash
uv run python -c "from src.experiments.pipeline import load_and_expand; print(len(load_and_expand('configs/pipelines/l_jitter.yaml')))"
```

## Datasets

Put COCO datasets under `datasets/` (gitignored). Layout:

```text
datasets/<name>/
  images/{train,val,test}/          # test only when the dataset has it
  <ann_dir>/instances_{split}.json
```

`<ann_dir>` is `annotations`, or `annotations_v1` / `annotations_v2` / `annotations_v1.5` for dual-label sets.

For DOTA 1024 and Plant Detection 640, carve cal from train before generating noise:

```bash
uv run bias-prepare --dataset dota_1024
uv run bias-prepare --dataset plant_detection_640
```

Noisy **train** JSONs (val/test/cal stay clean):

```bash
uv run python scripts/data/generate_noise.py --all
uv run python scripts/data/generate_noise.py --dataset hit_uav
```

Writes `datasets/<name>/annotations_noise/<ann_dir>/<family>_<pct>/instances_train.json`. `bias-run` reads those files; it does not generate them.

## Pipelines

One YAML is the run: dataset, experts, prune, calibration, combiner, eval. Optional `sweep:` expands listed axes (datasets, noise ratios, combine methods, moe).

```bash
uv run bias-run --config configs/pipelines/hit_uav_asym_cal_wbf.yaml
```

Stage gates: `--prepare-only`, `--train`, `--predict`, `--eval`, `--all`. With no stage flags, enabled stages in the YAML run.

Recipe files:

- `configs/pipelines/hit_uav_asym_cal_wbf.yaml` (single dataset, ratio sweep)
- `configs/pipelines/l_jitter.yaml`
- `configs/pipelines/o_objectness.yaml`
- `configs/pipelines/c_classflip_asymmetric.yaml`
- `configs/pipelines/c_classflip_symmetric.yaml`
- `configs/pipelines/mix.yaml`

Edit the YAML to change dataset or noise. Copy a recipe rather than adding CLI flags.

Train a single expert (no fusion):

```bash
uv run bias-train --config configs/training/ai_tod_v1.yaml
```

`--prepare-only` on that command converts COCO and exits. Outputs go under `outputs/` (gitignored). Training uses every visible CUDA GPU, or CPU if none.

The older factorial CLI still works: `uv run bias-experiment --matrix configs/experiments/l_runs.yaml`. Prefer `bias-run` and `configs/pipelines/`.

## Layout

```text
bias_rs/
├── configs/     training, data, and pipeline YAML
├── docs/        experimental notes
├── scripts/     dataset tools (noise, patching)
├── src/         package
├── tests/       pytest
├── datasets/    local images and COCO JSON (gitignored)
└── outputs/     training and experiment runs (gitignored)
```
