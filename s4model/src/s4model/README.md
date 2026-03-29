# S4 Model Training Framework

## 1. Technical Overview
This repository contains a modular framework for training Structured State Space models (S4 family) in PyTorch. Forked from [text](https://github.com/state-spaces/s4)

Supported model variants:
- **HiPPO**: continuous-time operator projection foundation used by S4.
- **S4D**: diagonal S4 for efficient sequence modeling.
- **S4ND**: n-dimensional S4 for image/video-style spatial data.
- **SaShiMi**: specialized waveform model for raw audio tasks.

Key modularity:
- Entry point: `s4model.s4model` (imported as `python -m s4model.s4model`)
- Configuration parser: `s4model.config.get_args()`
- Data handling/adapters: `s4model.dataset` + `s4model.dataloaders`
- Model core: `s4model.model.S4Model`
- Training loops: `s4model.train` (train/eval/checkpoint)
- Output helpers: `s4model.output`

## 2. Modern Installation
### 2.1 pip
```bash
cd /weathermetrics/s4model/src/s4model
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 2.2 Astral `uv`
```bash
python -m pip install uv
cd /weathermetrics/s4model/src/s4model
uv run s4model -- --help
```

> Note: `uv run s4model` invokes the package entrypoint from configured script target,
> and can accept arbitrary arguments forwarded to `s4model.s4model`.

## 3. Execution Guide
### 3.1 Default entrypoint
- Module: `python -m s4model.s4model`
- Package: `uv run s4model -- [args]`

### 3.2 Example (provided by user)
```bash
uv run s4model --modelname 72405013743 \
  --modeltype classification \
  --dataset /weathermetrics/weathermetricsdata/src/weathermetricsdata/output/72405013743_pipeline.parquet \
  --tabulardata \
  --dependent_variable y \
  --epochs 3 \
  --timeseriessplit
```

### 3.3 Tabular training
```bash
python -m s4model.s4model \
  --modelname experiment1 \
  --modeltype regression \
  --dataset /path/to/data.csv \
  --tabulardata \
  --dependent_variable target \
  --independent_variables feat1 feat2 feat3 \
  --epochs 12 \
  --batch_size 32
```

### 3.4 Image train shortcut
`--dataset` accepts built-ins `mnist`, `cifar10`, or full path files.
```bash
python -m s4model.s4model --modelname img01 --modeltype classification --dataset cifar10 --epochs 20
```

## 4. Parameter Reference (from `config.py`)
No `--task` argument exists; model type is defined by `--modeltype` (classification|regression).


## Arguments

### 4.1 Required arguments
- `--modelname`: model checkpoint prefix/name
- `--modeltype`: `classification` or `regression`

### 4.2 Data inputs
- `--dataset`: `mnist`, `cifar10`, or dataset path (CSV/Parquet)
- `--timeseriessplit`: flag, use time-series splitting logic
- `--trainvaltestsplit`: 3 floats for train/val/test drop-in
- `--trainset`, `--valset`, `--testset`: explicit split file paths
- `--tabulardata`: flag for tabular CSV data
- `--dependent_variable`: target field name (required for tabular)
- `--independent_variables`: space-separated feature names

### 4.3 Model & training
- `--n_layers`: network depth (default 4)
- `--d_model`: latent dimension (default 128)
- `--dropout`: dropout probability (default 0.1)
- `--prenorm`: apply prenormalization if set
- `--epochs`: number of epochs (default 20)
- `--patience`: scheduler early stop patience (default 10)
- `--lr`: learning rate (default 0.01)
- `--weight_decay`: L2 weight decay (default 0.01)

### 4.4 Data loaders + resume
- `--num_workers`: DataLoader workers (default 0)
- `--batch_size`: batch size (default 64)
- `--resume`: resume from checkpoint boolean

### 4.5 Output control
- `--output_data`: include input in output dataset
- `--csv`: output results as CSV (otherwise Parquet)

## 5. Architecture Insights
The training workflow is:
1. `s4model.s4model.main()` sets logging and parses args.
2. `dataset.load_data()` builds train/val/test loaders.
3. `S4Model(...)` is created from `model.py`.
4. optimizer/scheduler from `setup_optimizer`.
5. training + validation loop in `train` and `eval` functions.
6. Final `combine_results_to_dataframe` writes output tables.

### 5.1 Core components
- `model.S4Model`: backbone with encoder / S4 blocks / decoder.
- `dataset.load_data`: conditional logic for CIFAR/MNIST versus tabular.
- `output.combine_results_to_dataframe`: persistence to parquet/csv.

## 6. Inference & Results
`infer.py` is available for standalone prediction runs; it uses the same config parser.

### 6.1 inference command pattern
```bash
python -m s4model.infer \
  --modelname 72405013743 \
  --modeltype classification \
  --dataset /path/test.parquet \
  --tabulardata \
  --dependent_variable y \
  --independent_variables f1 f2 f3
```

### 6.2 outputs
- Checkpoint: by name under `checkpoint/`.
- Logs: `logs/training.log` and console output.
- Predictions: saved by `output.combine_results_to_dataframe` in the working directory (CSV or Parquet).

## 7. Quick troubleshooting
- Validate `trainvaltestsplit` sums to 1.0.
- `modeltype` must be `classification` or `regression`.
- For unknown args, they are logged as ignored; fix typos.
- Use `--tabulardata` for non-image tasks.

## 8. Contributors
- **https://github.com/state-spaces/s4**
- **Prajwal Koirala**
- **John Hughes**

## 9. License
Apache 2.0

