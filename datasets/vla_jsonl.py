"""Utilities for converting VLA records into MLX-LM JSONL conversations."""

import json
from pathlib import Path


def write_jsonl(records, path):
    """Write records accepted by MLX-LM's instruction-tuning data loader."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def record(instruction, response, system=None):
    """Build one chat-style record from a robot instruction and target response."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.extend([
        {"role": "user", "content": instruction},
        {"role": "assistant", "content": response},
    ])
    return {"messages": messages}

