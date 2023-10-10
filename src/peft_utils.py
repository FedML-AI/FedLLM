from .llm_finetune.src.peft_utils import *
# -----------------------------------------------------------------

from torch.nn.modules.module import _IncompatibleKeys
from peft import PeftModel, PeftType, PromptLearningConfig

from .modeling_utils import load_state_dict


# Adapted from https://github.com/huggingface/peft/blob/c33c42f1582b13e9b14b6b9467eff9275164ea36/src/peft/utils/save_and_load.py#L82
def set_peft_model_state_dict(
        model: PeftModel,
        peft_model_state_dict,
        adapter_name: str = "default"
) -> _IncompatibleKeys:
    """
    Set the state dict of the Peft model.

    Args:
        model ([`PeftModel`]): The Peft model.
        peft_model_state_dict (`dict`): The state dict of the Peft model.
        adapter_name: The adapter name.
    """
    config = model.peft_config[adapter_name]
    state_dict = {}
    if model.modules_to_save is not None:
        for key, value in peft_model_state_dict.items():
            if any(module_name in key for module_name in model.modules_to_save):
                for module_name in model.modules_to_save:
                    if module_name in key:
                        key = key.replace(module_name, f"{module_name}.modules_to_save.{adapter_name}")
                        break
            state_dict[key] = value
    else:
        state_dict = peft_model_state_dict

    if config.peft_type in (PeftType.LORA, PeftType.ADALORA, PeftType.IA3):
        peft_model_state_dict = {}
        parameter_prefix = "ia3_" if config.peft_type == PeftType.IA3 else "lora_"
        for k, v in state_dict.items():
            if parameter_prefix in k:
                suffix = k.split(parameter_prefix)[1]
                if "." in suffix:
                    suffix_to_replace = ".".join(suffix.split(".")[1:])
                    k = k.replace(suffix_to_replace, f"{adapter_name}.{suffix_to_replace}")
                else:
                    k = f"{k}.{adapter_name}"
                peft_model_state_dict[k] = v
            else:
                peft_model_state_dict[k] = v
        if config.peft_type == PeftType.ADALORA:
            rank_pattern = config.rank_pattern
            if rank_pattern is not None:
                model.resize_modules_by_rank_pattern(rank_pattern, adapter_name)
    elif isinstance(config, PromptLearningConfig) or config.peft_type == PeftType.ADAPTION_PROMPT:
        peft_model_state_dict = state_dict
    else:
        raise NotImplementedError

    load_result = load_state_dict(model, peft_model_state_dict, strict=False)
    if isinstance(config, PromptLearningConfig):
        load_state_dict(
            model.prompt_encoder[adapter_name].embedding,
            {"weight": peft_model_state_dict["prompt_embeddings"]},
            strict=True
        )
    return load_result
