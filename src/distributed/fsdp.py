# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import gc

import torch
from distributed.util import device_empty_cache
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy

from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper as ptd_checkpoint_wrapper,
)


def _iter_block_groups(model):
    """Yield (owner_module, attribute_name) that contains block lists."""
    if hasattr(model, "blocks"):
        yield model, "blocks"

    aggregator = getattr(model, "aggregator", None)
    if aggregator is not None:
        for attr in ("frame_blocks", "global_blocks", "cross_blocks"):
            if hasattr(aggregator, attr):
                yield aggregator, attr


def _iter_blocks(model):
    for owner, attr in _iter_block_groups(model):
        for block in getattr(owner, attr):
            yield block


def _shard_block_children(block, fsdp_config):
    """Shard known child modules when they exist."""
    for child_name in ("attn1", "attn2", "ffn", "attn", "cross_attn", "mlp"):
        child = getattr(block, child_name, None)
        if child is not None:
            fully_shard(child, **fsdp_config)

def apply_ac(model):
    """Apply activation checkpointing to the model."""
    for owner, attr in _iter_block_groups(model):
        blocks = getattr(owner, attr)
        for layer_id, transformer_block in enumerate(blocks):
            transformer_block = ptd_checkpoint_wrapper(transformer_block, preserve_rng_state=False)
            blocks[layer_id] = transformer_block


def shard_model(model,
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.float32):
    mp_policy = MixedPrecisionPolicy(
        param_dtype=param_dtype,
        reduce_dtype=reduce_dtype,
        cast_forward_inputs=False,
    )
    fsdp_config = {"mp_policy": mp_policy, "reshard_after_forward": True}

    for block in _iter_blocks(model):
        _shard_block_children(block, fsdp_config)
        fully_shard(block, **fsdp_config)

    fully_shard(model, **fsdp_config)
    return model


def free_model(model):
    del model
    gc.collect()
    device_empty_cache()
