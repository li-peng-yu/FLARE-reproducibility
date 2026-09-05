# FLARE reproducibility

Code, compact reference metrics and selected inference checkpoints for training, simulation, evaluation and figure generation. The datasets are too large to upload with the submission and are therefore not included. Upon acceptance, the datasets will be released through public channels, together with checksums and the information needed to regenerate them. Run the commands below from this directory.

## Setup

Git checkouts use Git LFS for checkpoint files. After cloning, run `git lfs install` and `git lfs pull` to download the weights.

```bash
conda env create -f environment.yml
conda activate flare-repro
python -m pip install -e . --no-deps
python -m pip install -r external_baselines/requirements.txt

export FLARE_DATASET_ROOT=/path/to/FLARE_dataset
export FLARE_CHECKPOINT_ROOT="$PWD/checkpoints"
export FLARE_TEMPERATURE_ROOT=/path/to/temperature_data
export PYTHON_BIN="$(command -v python)"
```

Training and prediction require a CUDA GPU. The FLARE training recipe uses two GPUs, a global batch of 128 and 50,000 updates.

Expected dataset layout:

```text
FLARE_dataset/
  primary_x5/
    skx_bt_1000base_x5_20260803/
    skx_bt_1000base_x5_sharedrelax_20260805/
  self_consistency_x30/
    skx_bt_165base_x30_sharedrelax_20260813/
```

The `checkpoints/` directory contains 20 inference checkpoints: seed-78 FLARE, Cartesian CFM, Direct U-Net, tangent-projected and two-coordinate variants, RFM, target-time and FNO ablations, ring-OOD models, three Stage-2 variants, and all seven external baselines. Stage-1 models and external baselines use 50,000 training updates; Stage-2 models use 25,000 additional updates. The released tensors retain their original precision. File names, training steps and SHA-256 checksums are listed in [configs/checkpoint_release.json](configs/checkpoint_release.json).

To reproduce experiments across training seeds, train the additional seeds and place their FLARE checkpoints under `$FLARE_CHECKPOINT_ROOT/flare/` using the names in [configs/flare/checkpoints.json](configs/flare/checkpoints.json). Baselines use `baselines/{poseidon_t,cno_fm,dpot_ti,mpp_avit_ti,pdearena_unet,le_pde,neuralmag_x5}.pt`. Baseline pretraining inputs go under `pretrained/`; their paths are defined in `external_baselines/x5_model_zoo.py` and `run/neuralmag.sh`. The supplied fine-tuned checkpoints can be evaluated without the pretraining inputs.

Build the x5/x30 float16 input caches before training or evaluation:

```bash
python run/reproduce.py cache
```

The supplied normalization and split configurations are in `configs/stats/` and `configs/data/`. Temperature evaluation uses raw OVF fields and the fixed training normalization.

## Training

```bash
NPROC_PER_NODE=2 bash run/train_flare.sh configs/base/x5.yaml configs/flare/stage1_standard_prior50k.yaml

# Direct U-Net: rotation-target MSE, seeds 78–80.
for task in 0 1 2; do
  bash run/experiments/run_x5_rotation_mse_train_2gpu.sh "$task"
done

for method in poseidon_t cno_fm dpot_ti mpp_avit_ti pdearena_unet le_pde; do
  bash run/train_baseline.sh "$method"
done
bash run/neuralmag.sh finetune
python run/reproduce.py ensemble-training
```

For the geometry, target-time, FNO, ring-OOD and Stage-2 models, repeat `train_flare.sh` with the corresponding override in `configs/flare/checkpoints.json`. Train Stage 1 before Stage 2. Register each `checkpoint_final.pt` at its listed checkpoint path; a symlink is sufficient. External baseline training writes `outputs/baseline_training/METHOD/checkpoint_0050000.pt`. Extra ensemble seeds remain at the output paths used by their training and evaluation scripts.

## Main results

Run these workflows in order:

```bash
python run/reproduce.py geometry
python run/reproduce.py external-primary
python run/reproduce.py timing
python run/reproduce.py finalize-primary
bash run/render_primary.sh
python run/reproduce.py ensemble
python run/reproduce.py jitter
```

Main quality evaluation uses 33 base conditions × five exact drive-start anchors, the prescribed driven duration, and predicted handoff into 3.5 ns of relaxation. Distribution Score and angular energy distance use five matched endpoints; fair energy score uses five forecasts per exact anchor. Timing measures a separate 2 ns drive + 3 ns relaxation workload. The timing summary uses the paper's 446,125 ms MuMax reference; `scripts/summarize_x5_fixed_2plus3_timing.py --mumax-ms` accepts a separately measured reference.

All workflows accept `--dry-run` to print commands and `--step INDEX` to run one numbered step. Generated metrics, figures and logs are written under `outputs/` and `reports/`.

## Horizon and auxiliary experiments

```bash
python run/reproduce.py horizons
python run/reproduce.py auxiliary
bash run/evaluate_x30_self_consistency.sh
bash run/render_qualitative.sh
```

**Horizon is the prediction interval Δt from the starting state.** The horizon workflow evaluates 1, 1.5, 2, 2.5 and 3 ns within the post-drive relaxation segment, then aggregates the metrics and renders the figure. It uses 33 base conditions, five anchors per condition and five forecasts per anchor. The 1.5 and 2.5 ns queries test time interpolation.

`auxiliary` runs single-segment ablations, the Heun step-count sweep, Stage-2 rollout, semigroup/handoff, target-time interpolation and ring-OOD. Its five final summaries use 50,000 base-group bootstrap replicates and seed 20260822.

## Temperature experiments

```bash
export MUMAX_BIN=/path/to/mumax3
python run/temperature.py prepare
python run/temperature.py simulate
python run/temperature.py audit
python run/temperature.py evaluate --temperatures 30 90 150 225 300 400
python run/temperature.py summarize
```

The simulator worker uses MuMax 3.12, CUDA 12.0, commit `6e5c98bb` and validates the recorded binary hash. `simulate --temperatures 90 --task 1` runs one of the 36 independent base tasks. Each generated base has fresh target-temperature relaxation and five thermal continuations.

Evaluation uses the fixed seed-78 Stage-1 EMA checkpoint, Heun-10 and the original training statistics. The 90 K reference is the paired per-base midpoint of 30/150 K, and the 225 K reference uses 150/300 K. The 400 K experiment uses the matched 300 K reference. Seeds, sample counts and bootstrap settings are specified in `run/temperature.py`.

## Metric and generative figures

```bash
python run/reproduce.py auxiliary-distribution
python run/reproduce.py metric-stress
bash run/generative_necessity.sh
```

The metric workflow generates the x30 sample pool before evaluating sample-budget and translation sensitivity. The generative figure workflow selects the prescribed cases, generates fresh FLARE and Poseidon predictions, and renders the main and appendix components under `outputs/figures/`.

The dataset generators are in `dataset_generation/`. Third-party implementation revisions are listed in `third_party/revisions.json`; their license notices accompany the source.

```bash
python -m pytest -q
```

## Reference results

These small CSV summaries provide numerical targets for checking a reproduction.

| Reference | Contents | Reproduction workflow |
| --- | --- | --- |
| [main_quality.csv](reference_results/main_quality.csv) | Main-table quality metrics and confidence intervals for ten learned models and controls | [Main results](#main-results) |
| [timing_2plus3ns.csv](reference_results/timing_2plus3ns.csv) | Batch-one and best-batch latency, speedup and peak GPU memory | `python run/reproduce.py timing` |
| [horizon_distribution.csv](reference_results/horizon_distribution.csv) | Distribution metrics at 1, 1.5, 2, 2.5 and 3 ns | `python run/reproduce.py horizons` |
| [temperature_inrange.csv](reference_results/temperature_inrange.csv) | 90/225 K metrics, adjacent-temperature references and degradation | [Temperature experiments](#temperature-experiments) |

The main quality and timing CSVs retain evaluation-output precision; horizon and temperature values use the paper's displayed precision. Columns ending in `_ci_low` and `_ci_high` give 95% confidence bounds. Higher Distribution Score and lower angular errors, angular energy distance and fair energy score are better. For temperature, positive degradation means worse performance, and relative degradation is expressed in percent. Timings depend on the execution hardware.
