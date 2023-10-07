from .llm_finetune.src.utils import *
# -----------------------------------------------------------------

from typing import Callable

import inspect
import logging

T = TypeVar("T")


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


def dummy_func(*args, **kwargs) -> None:
    return None


def get_dummy_func(default: T) -> Callable[..., T]:
    def _func(*args, **kwargs) -> T:
        return default

    return _func
