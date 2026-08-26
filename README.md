# open-sklid

This project is an open-source reimplementation of SKLID S1, a robotic vision-language-action (VLA) policy. The first phase implements `models/action`: a conditional diffusion action expert that takes a conditioning vector and generates action matrices with shape `[batch, horizon, action_dim]`.

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

The action dimension is configurable in `config/model/action.yaml` rather than hard-coded. The current defaults are `action_dim=14` and `horizon=16`; update them after confirming the target robot's MCAP schema.

## Quick start

```bash
pip install -r requirements.txt
python -m inference.sample_action --config config/model/action.yaml
```

Training data must be an `.npz` file with `actions` shaped `[N, horizon, action_dim]`. An optional `conditions` array can be provided with shape `[N, condition_dim]`.

```bash
python -m training.train_action --config config/model/action.yaml --data /path/to/data.npz
```

To inspect MCAP topics, install `mcap` and run `python -m datasets.mcap_to_npz sample_data/open-source-12.mcap --list-topics`. Since schemas vary across robot versions, action extraction mappings are intentionally kept explicit and configurable.
