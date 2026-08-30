# InfraGuard AI

InfraGuard AI is an early-stage Computer Vision project for detecting and analyzing building damage from UAV and high-resolution imagery.

The repository is currently in its engineering-foundation phase. No production model, benchmark result, or accuracy claim is available yet.

## Planned scope

- dataset exploration and validation
- defect and damage detection
- multi-dataset evaluation
- reproducible model training
- high-resolution inference
- deployment in a later project phase

## Repository structure

```text
configs/          Dataset and experiment configuration
data/             Local dataset guidance; dataset files are ignored by Git
docs/             Project, dataset, decision, roadmap, and experiment documentation
notebooks/        Exploratory notebooks
scripts/          Command-line entry points
src/infraguard/   Reusable Python package
tests/            Automated tests
```

## Development setup

InfraGuard AI targets Python 3.11.

```bash
python -m venv .venv
pip install -e ".[dev,notebook]"
```

## Quality checks

```bash
pytest
ruff check .
mypy src
```

## Dataset policy

Datasets must be stored locally and must not be committed to this repository. Each dataset is distributed under its own license and usage terms.

The repository's MIT License applies to the InfraGuard AI source code only. It does not automatically apply to MBDD2025, CUBIT-Det, or any other dataset.

See [`data/README.md`](data/README.md) and [`docs/DATASETS.md`](docs/DATASETS.md) for dataset-related guidance and currently documented information.
