# UniMacro (code release)

This repository contains the code to preprocess polymer datasets (CSV → PKL) and train/evaluate models for downstream tasks. The framework uses SMILES strings as the primary molecular representation. It also supports multi-SMILES inputs (e.g., for copolymers or mixtures). Since SMILES are primarily parsed by RDKit to extract molecular features, other molecular formats compatible with RDKit should also be functional, such as .mol files (use ```rdkit.Chem.MolFromMolFiles```)

---

## 0. Environment

We provide a Conda environment file.

```bash
conda env create -f environment.yml
conda activate unimacro
```

Notes:
- Python: 3.9
- PyTorch: 2.5 (CUDA 12.1)
- RDKit is required for SMILES parsing and conformer generation.

---

## 1. Data format (CSV)

Each dataset is a CSV file. All datasets used in this work are provided as CSV files and are located in the ```/datasets``` folder.

### 1.1 Required columns

The preprocessing script reads the following column patterns:

#### SMILES segments
- `SMILES0`, `SMILES1`, ...
- In this codebase, the maximum number of segments is controlled by `MAX_SEGMENTS` in the preprocessing script, so typically you will provide:
  - `SMILES0` (required)
  - `SMILES1` (optional)
  - `SMILES2` (optional)
  - ...
  - `SMILES{MAX_SEGMENTS - 1}` (optional)

#### Segment-level numeric features (per SMILES segment)
- `seg{k}_feat{i}`
- By default, per segment you can provide up to **2** local features (`MAX_LOCAL_FEATS = 2`, feel free to change it to the actual maximum number of features of the segments):
  - `seg0_feat0`, `seg0_feat1`

Examples: degree of polymerization, block fraction, etc.

#### Global numeric features (shared by the full sample)
- `glob_feat{j}`
- In this codebase, you can provide up to **2** global features (`MAX_GLOBAL_FEATS = 2`, feel free to change it to the actual maximum number of features of the segments):
  - `glob_feat0`, `glob_feat1`

Examples: temperature, pressure, etc.

#### Labels
- One or multiple label columns (e.g., `Tg`, `Ei`, ...)
- Specify label column(s) by `--labels`, e.g. `--labels Tg` or `--labels Tg,Ei`

### 1.2 Optional columns

#### `fold` (recommended for fixed CV splits)
If the CSV contains a `fold` column, preprocessing will use it to assign each row to a fold:
- validation fold for `--fold k` in training is `fold{k}`

If `fold` is not provided, folds are assigned randomly using `--seed`.

---

## 2. Preprocess: CSV → PKL

Script: `src/preprocess/preprocessing_polymer.py`

### 2.1 Finetune task (recommended for downstream regression/classification)

This produces a single PKL file that contains all samples + their fold assignments. The preprocessing is typically fast, in our case (64 core, Intel Xeon6458), it takes typically 5-20s to finish the preprocessing process.

Recommended command (also exports split indices and per-fold train/val CSVs):

```bash
python src/preprocess/preprocessing_polymer.py \
  --task finetune \
  --csv /path/to/your_dataset.csv \
  --labels Tg \
  --kfold 5 \
  --seed 42 \
  --outroot data/processed \
  --dataset-name your_dataset_name \
  --export-splits
```

Outputs (under `data/processed/your_dataset_name/`):
- `main/split.pkl`: main file used by `src/main.py`
- `index/split_fold{k}.pkl`: indices in the style of PerioGT (train/val/test)
- `csv/train{k}.csv`, `csv/val{k}.csv`: exported fold CSVs

If you prefer to specify the PKL path directly (without `--outroot`):

```bash
python src/preprocess/preprocessing_polymer.py \
  --task finetune \
  --csv /path/to/your_dataset.csv \
  --labels Tg \
  --kfold 5 \
  --seed 42 \
  --output data/processed/your_dataset_name.pkl \
  --export-splits
```

### 2.2 Pretrain task (optional)

The same script can also shard a large CSV into multiple `shard*.pkl` files:

```bash
python src/preprocess/preprocessing_polymer.py \
  --task pretrain \
  --csv /path/to/pretrain.csv \
  --labels dummy \
  --kfold 5 \
  --seed 42 \
  --outdir data/pretrain_shards
```

(Labels are kept in each sample dict, but for pretraining they may be unused.)

---

## 3. Finetune training

Entry point: `src/main.py`

### 3.1 Single GPU

```bash
python src/main.py \
  --main_task finetune \
  --task_type reg \
  --dataset_name your_dataset_name \
  --fold 0 \
  --pkl_path data/processed/your_dataset_name/main/split.pkl \
  --epochs 100 \
  --batch_size 16 \
  --lr 1e-4 \
  --weight_decay 1e-4
```

Notes:
- By default, the code tries to load `--weight_path` (which is set to an internal absolute path in this repo). If the checkpoint is not found, it will automatically train from scratch.
- For classification, use `--task_type cls`.

### 3.2 Multi-GPU (DDP)

```bash
torchrun --standalone --nproc_per_node 4 src/main.py \
  --distributed \
  --main_task finetune \
  --task_type reg \
  --dataset_name your_dataset_name \
  --fold 0 \
  --pkl_path data/processed/your_dataset_name/main/split.pkl
```

### 3.3 Outputs

By default, results are written to:
- `results/<dataset_name>/fold_<k>/...`

This includes:
- `metrics.json`
- `checkpoints/checkpoint.pt` (best model)
- `predictions/*.npy` (residuals / confusion matrix, etc.)

---

## 4. Troubleshooting

- **RDKit import errors**: make sure you created the conda env from `environment.yml`.
- **No SMILES columns found**: check that your CSV has `SMILES0` (and optionally `SMILES1`).
- **Want fixed CV folds**: add a `fold` column to the CSV before preprocessing.
