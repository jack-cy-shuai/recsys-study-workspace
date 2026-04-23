# RecSys Study Workspace

This repository is a personal research workspace for recommender systems.
It currently includes:

- a minimal `LightGCN` reproduction
- a small implicit-feedback dataset for debugging
- classic recommendation papers
- LLM-based recommendation papers
- space for notes, experiment logs, and future extensions

## Project Layout

```text
run_recmodels/
|-- lightgcn/         # model, data pipeline, evaluation
|-- scripts/          # helper scripts such as dataset generation
|-- data/             # datasets used by experiments
|-- artifacts/        # saved checkpoints and metrics
|-- papers/
|   |-- traditional/  # classic recommendation papers
|   `-- llmrec/       # LLM-based recommendation papers
|-- notes/            # reading notes and personal understanding
|-- docs/             # project docs, summaries, roadmaps
|-- experiments/      # experiment plans and result records
|-- train.py          # main training entry
`-- requirements.txt
```

## Environment

This project is meant to run with the local Conda environment:

```powershell
conda run -n recsys python --version
```

If you use PyCharm, set the interpreter to:

```text
I:\miniconda3\envs\recsys\python.exe
```

## Quick Start

Generate the basic dataset:

```powershell
conda run -n recsys python scripts/make_basic_dataset.py
```

Train and evaluate LightGCN:

```powershell
conda run -n recsys python train.py
```

Example with custom arguments:

```powershell
conda run -n recsys python train.py --epochs 100 --embedding-dim 64 --layers 3 --eval-k 10 20
```

## Outputs

By default, training saves results to:

- `artifacts/basic_run/best_model.pt`
- `artifacts/basic_run/metrics.json`

## Data Format

The sample dataset uses a CSV file with these columns:

- `user_id`
- `item_id`
- `split`
- `timestamp`

The `split` field should be one of:

- `train`
- `val`
- `test`

## Suggested Workflow

For research study, a good workflow is:

1. Put papers into `papers/`
2. Write reading notes into `notes/`
3. Record experiment ideas in `experiments/`
4. Keep reusable explanations and summaries in `docs/`
5. Modify and debug code in PyCharm while using this repository as the shared workspace

