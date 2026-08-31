# Local datasets

Datasets are stored locally and must not be committed to Git.

## Directory convention

`data/raw/` contains original downloaded datasets. For MBDD2025, use:

```text
data/raw/mbdd2025/MBDD2025.zip
```

All content under `data/raw/` must remain ignored by Git. Raw files are
read-only source material: do not modify original images or annotations in
place. Future transformations must write derived data to a separate directory,
such as `data/interim/` or `data/processed/`.

## License and redistribution

Each dataset is distributed under its own license. Review and comply with that license before downloading or using the data.

The repository's MIT License applies to the source code only. It does not automatically apply to MBDD2025 or any other dataset used by the project.

Do not upload or redistribute a dataset unless its own license and usage terms
explicitly permit that action.
