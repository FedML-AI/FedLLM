from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from transformers import (
    EvalPrediction,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)
from transformers.trainer_utils import TrainOutput

from .hf_trainer import HFTrainer
from .typing import (
    DataCollatorType,
    DatasetType,
    LrSchedulerType,
    ModelType,
    TokenizerType,
)
from .utils import dummy_func


class FedLLMTrainerCallback(TrainerCallback):
    def __init__(self, reset_list: Optional[List[Tuple[Optional[Any], str, Any]]] = None):
        """

        Args:
            reset_list: a list of (object, attribute_name, value_to_recover) for recovery on_train_begin
        """
        if reset_list is None:
            reset_list = []
        self.reset_list = reset_list

    def on_train_begin(
            self,
            args: TrainingArguments,
            state: TrainerState,
            control: TrainerControl,
            **kwargs
    ) -> TrainerControl:
        for (obj, attr_name, value) in self.reset_list:
            if obj is not None and hasattr(obj, attr_name):
                setattr(obj, attr_name, value)
        self.reset_list.clear()

        return control


class FedLLMTrainer(HFTrainer):
    def __init__(
            self,
            model: Union[ModelType, Module] = None,
            args: TrainingArguments = None,
            data_collator: Optional[DataCollatorType] = None,
            train_dataset: Optional[DatasetType] = None,
            eval_dataset: Optional[Union[DatasetType, Dict[str, DatasetType]]] = None,
            tokenizer: Optional[TokenizerType] = None,
            model_init: Optional[Callable[[], Union[ModelType, Module]]] = None,
            compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
            callbacks: Optional[List[TrainerCallback]] = None,
            optimizers: Tuple[Optimizer, LrSchedulerType] = (None, None),
            preprocess_logits_for_metrics: Optional[Callable[[Tensor, Tensor], Tensor]] = None,
            is_resume_from_interrupt: bool = False,
            resume_train_callback: Optional[FedLLMTrainerCallback] = None
    ):
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics
        )

        # set to `True` if continuing from previous early-stopped train
        self.is_resume_from_interrupt = is_resume_from_interrupt

        if resume_train_callback is None:
            resume_train_callback = FedLLMTrainerCallback()
        self.resume_train_callback = resume_train_callback
        self.add_callback(self.resume_train_callback)

    def create_optimizer_and_scheduler(self, num_training_steps: int) -> None:
        if not self.is_resume_from_interrupt:
            return super().create_optimizer_and_scheduler(num_training_steps)

    def create_optimizer(self) -> Optimizer:
        if not self.is_resume_from_interrupt:
            return super().create_optimizer()
        else:
            return self.optimizer

    def create_scheduler(self, num_training_steps: int, optimizer: Optional[Optimizer] = None) -> LrSchedulerType:
        if not self.is_resume_from_interrupt:
            return super().create_scheduler(num_training_steps, optimizer)
        else:
            return self.lr_scheduler

    def train(
            self,
            resume_from_checkpoint: Optional[Union[str, bool]] = None,
            trial: Union["optuna.Trial", Dict[str, Any]] = None,
            ignore_keys_for_eval: Optional[List[str]] = None,
            **kwargs
    ) -> TrainOutput:
        if self.is_resume_from_interrupt:
            reset_list = self.resume_train_callback.reset_list

            # turn off TrainingArguments.deepspeed to avoid duplicated initializations
            # TODO: verify model, model_wrapped, deepspeed, optimizer, lr_scheduler after reset
            reset_list.append((self.args, "deepspeed", self.args.deepspeed))
            self.args.deepspeed = None

            # TODO: remove the hasattr check once we require transformers>=4.31.0
            if hasattr(self, "_created_lr_scheduler"):
                reset_list.append((self, "_created_lr_scheduler", self._created_lr_scheduler))
                self._created_lr_scheduler = False

            if hasattr(self, "accelerator"):
                # when resuming, should disable the free_memory function call at the beginning
                # of Trainer._inner_training_loop
                reset_list.append((self.accelerator, "free_memory", self.accelerator.free_memory))
                self.accelerator.free_memory = dummy_func

            if hasattr(self, "is_deepspeed_enabled"):
                reset_list.append((self, "is_deepspeed_enabled", self.is_deepspeed_enabled))
                self.is_deepspeed_enabled = False

        train_output = super().train(
            resume_from_checkpoint=resume_from_checkpoint,
            trial=trial,
            ignore_keys_for_eval=ignore_keys_for_eval,
            **kwargs
        )

        self.is_resume_from_interrupt = True
        return train_output
