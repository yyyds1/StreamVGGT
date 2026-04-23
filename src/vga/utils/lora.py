from __future__ import annotations

from math import sqrt
from typing import Iterable, List, Sequence, Tuple

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    def __init__(
        self,
        base_layer: nn.Linear,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
    ):
        super().__init__()
        if not isinstance(base_layer, nn.Linear):
            raise TypeError(f"LoRALinear expects nn.Linear, got {type(base_layer)!r}")
        if rank <= 0:
            raise ValueError(f"rank must be > 0, got {rank}")

        self.base_layer = base_layer
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.lora_A = nn.Linear(base_layer.in_features, self.rank, bias=False)
        self.lora_B = nn.Linear(self.rank, base_layer.out_features, bias=False)

        nn.init.kaiming_uniform_(self.lora_A.weight, a=sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

        self.base_layer.weight.requires_grad_(False)
        if self.base_layer.bias is not None:
            self.base_layer.bias.requires_grad_(False)

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
    ) -> "LoRALinear":
        base_layer = nn.Linear(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
            device=linear.weight.device,
            dtype=linear.weight.dtype,
        )
        base_layer.load_state_dict(linear.state_dict())
        return cls(base_layer=base_layer, rank=rank, alpha=alpha, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base_layer(x) + self.lora_B(self.lora_A(self.dropout(x))) * self.scaling


def _should_replace(name: str, target_names: Sequence[str]) -> bool:
    return any(name == target or name.endswith(f".{target}") or name.endswith(target) for target in target_names)


def apply_lora_to_module(
    module: nn.Module,
    target_names: Sequence[str],
    rank: int,
    alpha: float,
    dropout: float = 0.0,
) -> List[str]:
    """Recursively replace matching nn.Linear layers with LoRALinear.

    Returns the fully qualified module names that were replaced.
    """
    replaced: List[str] = []

    def _recurse(parent: nn.Module, prefix: str = "") -> None:
        for child_name, child in list(parent.named_children()):
            qualified_name = f"{prefix}.{child_name}" if prefix else child_name
            if isinstance(child, nn.Linear) and _should_replace(qualified_name, target_names):
                setattr(parent, child_name, LoRALinear.from_linear(child, rank=rank, alpha=alpha, dropout=dropout))
                replaced.append(qualified_name)
                continue
            _recurse(child, qualified_name)

    _recurse(module)
    return replaced


def extract_lora_state_dict(module: nn.Module) -> dict[str, torch.Tensor]:
    """Collect LoRA parameters and the VGA action-query tokens for resume."""
    state = module.state_dict()
    return {
        key: value.detach().cpu()
        for key, value in state.items()
        if "lora_" in key or key.endswith("action_query_tokens")
    }


def load_lora_state_dict(module: nn.Module, state_dict: dict[str, torch.Tensor], strict: bool = False):
    """Load a previously saved LoRA delta into the module."""
    return module.load_state_dict(state_dict, strict=strict)
