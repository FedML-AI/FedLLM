from .llm_finetune.src.utils import *
# -----------------------------------------------------------------

from typing import (
    Any,
    Dict,
    List,
    MutableMapping,
    TypeVar,
    Union,
    Optional,
)

import inspect
import logging

import torch
from torch import Tensor
from torch.nn import Module
from transformers.deepspeed import is_deepspeed_zero3_enabled

from .distributed import (
    is_deepspeed_initialized,
    is_deepspeed_module,
    gather_parameter,
    get_rank,
)

T = TypeVar("T")


def to_device(data: T, device: Union[torch.device, str], non_blocking: bool = True) -> T:
    if isinstance(data, list):
        data = [to_device(d, device, non_blocking) for d in data]

    elif isinstance(data, tuple):
        data = tuple(to_device(d, device, non_blocking) for d in data)

    elif isinstance(data, MutableMapping):
        for k in data.keys():
            data[k] = to_device(data[k], device, non_blocking)

    elif isinstance(data, (Tensor, Module)):
        data = data.to(device, non_blocking=non_blocking)

    return data


def log_helper(
        message: str,
        prefix: str = "",
        suffix: str = "",
        stack_prefix: str = "",
        stack_level: int = 1,
        level: int = logging.INFO
):
    logging.log(
        level=level,
        msg=f"{prefix} [{stack_prefix}{inspect.stack()[stack_level][3]}]: {message} {suffix}",
    )


def load_state_dict(
        model: Module,
        state_dict: Dict[str, Any],
        strict: bool = True,
        force_recursive_load: bool = False
) -> None:
    if (is_deepspeed_initialized() and is_deepspeed_zero3_enabled() and is_deepspeed_module(model)) \
            or force_recursive_load:
        metadata = getattr(state_dict, "_metadata", None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        error_msgs = []
        load_state_dict_helper(
            module=model,
            state_dict=state_dict,
            strict=strict,
            metadata=metadata,
            error_msgs=error_msgs
        )
    else:
        model.load_state_dict(state_dict, strict=strict)


def load_state_dict_helper(
        module: Module,
        state_dict: Dict[str, Any],
        prefix: str = "",
        strict: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
        error_msgs: Optional[List[str]] = None
) -> None:
    """
    Recursively load state_dict in to module. This function handles the partitioned cases when using DeepSpeed
        Stage 3 (zero3). This function is adapted from
        `transformers.modeling_utils._load_state_dict_into_model`; see
        https://github.com/huggingface/transformers/blob/539e2281cd97c35ef4122757f26c88f44115fa94/src/transformers/modeling_utils.py#LL493C25-L493C25

    Args:
        module: module (`torch.nn.Module`) to load state_dict.
        state_dict: a dict containing parameters and persistent buffers.
        prefix: the prefix for parameters and buffers used in this module.
        strict: whether to strictly enforce that the keys in `state_dict` with `prefix` match the names of
                parameters and buffers in this module.
        metadata: a dict containing the metadata for this module.
        error_msgs: error messages should be added to this list.

    Returns:

    """
    local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
    if error_msgs is None:
        error_msgs = []

    # Parameters of module and children will start with prefix. We can exit early if there are none in this state_dict
    if any(key.startswith(prefix) for key in state_dict):
        # In sharded models, each shard has only part of the full state_dict, so only gather
        # parameters that are in the current state_dict.
        named_parameters = dict(module.named_parameters(prefix=prefix[:-1], recurse=False))
        params_to_gather = [named_parameters[k] for k in state_dict.keys() if k in named_parameters]
        if len(params_to_gather) > 0:
            # Since Zero3 puts placeholders in model params, this context manager gathers the params of
            # the current layer, then loads from the state dict and then re-partitions them again
            with gather_parameter(params_to_gather, modifier_rank=0):
                if get_rank() == 0 or not is_deepspeed_module(module):
                    module._load_from_state_dict(
                        state_dict=state_dict,
                        prefix=prefix,
                        local_metadata=local_metadata,
                        strict=strict,
                        missing_keys=[],
                        unexpected_keys=[],
                        error_msgs=error_msgs
                    )

    for name, child in module.named_children():
        load_state_dict_helper(
            module=child,
            state_dict=state_dict,
            strict=strict,
            prefix=f"{prefix}{name}.",
            metadata=metadata,
            error_msgs=error_msgs
        )
