# open-skild (WIP)

This project is an open-source reimplementation of skild S1, a robotic vision-language-action (VLA) policy. The first phase implements `models/action`: a conditional diffusion action expert that takes a conditioning vector and generates action matrices with shape `[batch, horizon, action_dim]`.

## About skild S1

This project is inspired by the official [skild S1 announcement](https://skild.ai/blogs/s1). The defining ideas we aim to reproduce are:

- **One-shot capability:** the policy should be able to perform a new task from a single demonstration, without task-specific retraining.
- **Video as the demo interface:** demonstrations are provided as videos, allowing the system to learn from natural human demonstrations rather than rigid task annotations.
- **Implicit context:** the demonstration video carries context that is not necessarily expressed as explicit language or labels, including task intent, object interaction, motion style, timing, and scene-specific constraints.

The action expert in this repository is designed as one component of that broader system. Its conditioning interface is intentionally extensible so visual features, language features, robot state, and implicit context extracted from demonstration videos can be fused later.

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

## Language model: Qwen3.5 0.8B with MLX

The VLA language branch uses Qwen3.5 0.8B through MLX-LM on Apple Silicon. Install the optional dependencies with `pip install mlx mlx-lm`, then run:

```bash
python -m training.finetune_qwen_mlx --config config/train/qwen3_5_0_8b_lora.yaml
```


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
