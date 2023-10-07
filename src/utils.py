from .llm_finetune.src.utils import *
# -----------------------------------------------------------------

import inspect
import logging


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
