"""Launch Qwen LoRA fine-tuning through MLX-LM.

Input JSONL records should contain a ``text`` field, or ``prompt`` and
``completion`` fields. MLX-LM's data directory should contain train.jsonl and
valid.jsonl files.
"""

import argparse
import subprocess
import sys
from pathlib import Path

import yaml


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Qwen3.5 0.8B with MLX LoRA")
    parser.add_argument("--config", default="config/train/qwen3_5_0_8b_lora.yaml")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    command = [
        sys.executable,
        "-m",
        "mlx_lm.lora",
        "--model",
        config["model"],
        "--data",
        str(Path(config["data"]).parent),
        "--adapter-path",
        config["adapter_path"],
        "--iters",
        str(config["iters"]),
        "--batch-size",
        str(config["batch_size"]),
        "--num-layers",
        str(config["num_layers"]),
        "--learning-rate",
        str(config["learning_rate"]),
        "--save-every",
        str(config["save_every"]),
        "--val-batches",
        str(config["val_batches"]),
    ]
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()

