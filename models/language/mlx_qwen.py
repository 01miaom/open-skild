"""MLX-LM wrapper for Qwen3.5 0.8B.

The import is intentionally lazy so PyTorch-only action development also works
on machines without Apple Silicon or MLX installed.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class QwenMLXConfig:
    model: str = "Qwen/Qwen3.5-0.8B"
    adapter_path: Optional[str] = None
    max_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    trust_remote_code: bool = True


class QwenMLXModel:
    """Small inference adapter that keeps MLX-specific code out of the VLA stack."""

    def __init__(self, config: QwenMLXConfig):
        try:
            from mlx_lm import generate, load
        except ImportError as exc:
            raise ImportError(
                "MLX support requires Apple Silicon and `pip install mlx mlx-lm`."
            ) from exc
        self._generate = generate
        load_kwargs = {}
        if config.adapter_path:
            load_kwargs["adapter_path"] = config.adapter_path
        self.model, self.tokenizer = load(config.model, **load_kwargs)
        self.config = config

    def generate(self, prompt: str, **kwargs) -> str:
        params = {
            "max_tokens": self.config.max_tokens,
            "temp": self.config.temperature,
            "top_p": self.config.top_p,
        }
        params.update(kwargs)
        return self._generate(self.model, self.tokenizer, prompt=prompt, **params)

    def format_vla_prompt(self, instruction: str, state_summary: str = "") -> str:
        context = f"\nRobot state: {state_summary}" if state_summary else ""
        return (
            "You are the language module of a robot policy. "
            "Convert the instruction and robot state into concise action-relevant context.\n"
            f"Instruction: {instruction}{context}\nAssistant:"
        )

    def encode(self, text: str):
        """Return the penultimate-layer latent feature sequence.

        The returned tensor has shape ``[1, sequence_length, hidden_dim]`` and
        is passed to the action expert as a set of conditioning features.
        """
        import mlx.core as mx

        tokens = self.tokenizer.encode(text)
        input_ids = mx.array([tokens])
        try:
            output = self.model(input_ids, output_hidden_states=True)
        except TypeError as exc:
            raise RuntimeError(
                "This MLX-LM model adapter does not expose hidden states; "
                "the VLA bridge requires the penultimate transformer layer."
            ) from exc
        hidden_states = getattr(output, "hidden_states", None)
        if hidden_states is None and isinstance(output, (tuple, list)):
            hidden_states = output[-1]
        if hidden_states is None or len(hidden_states) < 2:
            raise RuntimeError("Qwen MLX output does not contain penultimate hidden states")
        return hidden_states[-2]
