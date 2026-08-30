# `📡 🤫`
## Requirements

- Python 3.10 or newer
- [uv](https://docs.astral.sh/uv/getting-started/installation/)

## Initialize the environment

```bash
uv sync
```

## Dataset

Place the required input files in the `dataset/` directory.

## Run the notebooks

```bash
uv run jupyter notebook AOA_Paper.ipynb                          # paper flow on one partial-spoof capture
uv run jupyter notebook AOA_DD_Ephemeris_Graph_Classifier.ipynb
```

Before a new capture goes through `AOA_Paper.ipynb`, check it and the rig from the shell:

```bash
uv run python probe_dataset.py dataset/<capture> --spoof <PRNs>
uv run python calibrate_baseline_azimuth.py dataset/<capture> --spoof <PRNs>
```

## Tests

```bash
uv run python -m unittest
```
