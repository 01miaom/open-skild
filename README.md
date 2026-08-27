# open-skild (WIP)

This project is an open-source reimplementation of skild S1. It is currently a work in progress.

## Project layout

```text
config/{model,dataset,train}/       Configuration files
models/{vision,language,action}/    Model implementations
datasets/                            Data processing
training/                            Training entry points
inference/                           Inference entry points
tools/ and scripts/                  Utilities and scripts
sample_data/                         Example MCAP data
```

## Quick start

```bash
pip install -r requirements.txt
python -m inference.sample_action --config config/model/action.yaml
```

```bash
python -m training.train_action --config config/model/action.yaml --data /path/to/data.npz
```
