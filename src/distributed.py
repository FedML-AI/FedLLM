from .llm_finetune.src.distributed import *
# -----------------------------------------------------------------

from typing import ContextManager, Iterable, Optional, Union

from contextlib import contextmanager, nullcontext

from torch import distributed as dist
from torch.nn import Module, Parameter
from transformers import Trainer
from transformers.deepspeed import is_deepspeed_available, is_deepspeed_zero3_enabled

if is_deepspeed_available():
    import deepspeed
    from deepspeed.runtime.zero import GatheredParameters


def get_rank() -> int:
    if is_deepspeed_initialized():
        return deepspeed.comm.get_rank()
    elif dist.is_initialized():
        return dist.get_rank()
    else:
        return 0


def is_main_process(trainer: Trainer, local: bool = False) -> bool:
    return trainer.is_local_process_zero() if local else trainer.is_world_process_zero()


def should_process_save(trainer: Trainer) -> bool:
    return is_main_process(trainer, trainer.args.save_on_each_node)


@contextmanager
def gather_parameter(
        params: Union[Iterable[Parameter], Parameter],
        modifier_rank: Optional[int] = None,
        fwd_module: Optional[Module] = None,
        enabled: bool = True
) -> ContextManager[None]:
    if enabled and is_deepspeed_initialized() and is_deepspeed_zero3_enabled() and is_deepspeed_module(params):
        context = GatheredParameters(params, modifier_rank, fwd_module, enabled)
    else:
        context = nullcontext()

    with context:
        yield
