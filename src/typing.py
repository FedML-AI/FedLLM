from .llm_finetune.src.typing import *
# -----------------------------------------------------------------

from typing import Union

from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau

LrSchedulerType = Union[LambdaLR, ReduceLROnPlateau]
