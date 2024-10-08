# SPDX-License-Identifier: Apache-2.0
# Standard
from abc import ABC
from importlib import resources
from typing import Any, Optional
import operator
import os

# Third Party
import yaml

# Local
from .filterblock import FilterByValueBlock
from .llmblock import ConditionalLLMBlock, LLMBlock, LLMLogProbBlock
from .utilblocks import (
    CombineColumnsBlock,
    DuplicateColumns,
    FlattenColumnsBlock,
    RenameColumns,
    SamplePopulatorBlock,
    SelectorBlock,
    SetToMajorityValue,
)

MODEL_FAMILY_MIXTRAL = "mixtral"
MODEL_FAMILY_MERLINITE = "merlinite"
MODEL_FAMILY_BLANK = "blank"
MODEL_FAMILY_IBM = "ibm"
MODEL_FAMILY_RHELAI = "rhelai"

_MODEL_PROMPT_MIXTRAL = "<s> [INST] {prompt} [/INST]"
_MODEL_PROMPT_MERLINITE = "<|system|>\nYou are an AI language model developed by IBM Research. You are a cautious assistant. You carefully follow instructions. You are helpful and harmless and you follow ethical guidelines and promote positive behavior.\n<|user|>\n{prompt}\n<|assistant|>\n"
_MODEL_PROMPT_RHELAI = "<|system|>\nI am, Red Hat® Instruct Model based on Granite 7B, an AI language model developed by Red Hat and IBM Research, based on the Granite-7b-base language model. My primary function is to be a chat assistant.\n<|user|>\n{prompt}\n<|assistant|>\n"
_BLANK_PROMPT = "{prompt}"


_MODEL_PROMPTS = {
    MODEL_FAMILY_MIXTRAL: _MODEL_PROMPT_MIXTRAL,
    MODEL_FAMILY_MERLINITE: _MODEL_PROMPT_MERLINITE,
    MODEL_FAMILY_BLANK: _BLANK_PROMPT,
    MODEL_FAMILY_IBM: _MODEL_PROMPT_MERLINITE,
    MODEL_FAMILY_RHELAI: _MODEL_PROMPT_RHELAI,
}


def _get_model_prompt(model_family):
    if model_family not in _MODEL_PROMPTS:
        raise ValueError(f"Unknown model family: {model_family}")
    return _MODEL_PROMPTS[model_family]


BLOCK_TYPE_MAP = {
    "LLMBlock": LLMBlock,
    "FilterByValueBlock": FilterByValueBlock,
    "CombineColumnsBlock": CombineColumnsBlock,
    "SamplePopulatorBlock": SamplePopulatorBlock,
    "SelectorBlock": SelectorBlock,
    "DuplicateColumns": DuplicateColumns,
    "RenameColumns": RenameColumns,
    "FlattenColumnsBlock": FlattenColumnsBlock,
    "ConditionalLLMBlock": ConditionalLLMBlock,
    "SetToMajorityValue": SetToMajorityValue,
    "LLMLogProbBlock": LLMLogProbBlock,
}

MODEL_FAMILY_MAP = {
    "mistralai/Mixtral-8x7B-Instruct-v0.1": MODEL_FAMILY_MIXTRAL,
}

OPERATOR_MAP = {
    "operator.eq": operator.eq,
    "operator.ge": operator.ge,
    "operator.contains": operator.contains,
}

CONVERT_DTYPE_MAP = {
    "float": float,
}


class Flow(ABC):
    def __init__(
        self, client: Any, num_instructions_to_generate: Optional[int] = None
    ) -> None:
        self.client = client
        self.num_instructions_to_generate = num_instructions_to_generate
        self.sdg_base = resources.files(__package__)

    def get_flow_from_file(self, yaml_path: str) -> list:
        yaml_path_relative_to_sdg_base = os.path.join(self.sdg_base, yaml_path)
        if os.path.isfile(yaml_path_relative_to_sdg_base):
            yaml_path = yaml_path_relative_to_sdg_base
        with open(yaml_path, "r", encoding="utf-8") as yaml_file:
            flow = yaml.safe_load(yaml_file)
        for block in flow:
            if "LLM" in block["block_type"]:
                block["block_config"]["client"] = self.client

            block["block_type"] = BLOCK_TYPE_MAP[block["block_type"]]

            if "config_path" in block["block_config"]:
                block_config_path_relative_to_sdg_base = os.path.join(
                    self.sdg_base, block["block_config"]["config_path"]
                )
                if os.path.isfile(block_config_path_relative_to_sdg_base):
                    block["block_config"]["config_path"] = (
                        block_config_path_relative_to_sdg_base
                    )

            if "config_paths" in block["block_config"]:
                if isinstance(block["block_config"]["config_paths"], dict):
                    for key, path in block["block_config"]["config_paths"].items():
                        block_config_path_relative_to_sdg_base = os.path.join(
                            self.sdg_base, path
                        )
                        if os.path.isfile(block_config_path_relative_to_sdg_base):
                            block["block_config"]["config_paths"][key] = (
                                block_config_path_relative_to_sdg_base
                            )

                if isinstance(block["block_config"]["config_paths"], list):
                    for i, path in enumerate(block["block_config"]["config_paths"]):
                        block_config_path_relative_to_sdg_base = os.path.join(
                            self.sdg_base, path
                        )
                        if os.path.isfile(block_config_path_relative_to_sdg_base):
                            block["block_config"]["config_paths"][i] = (
                                block_config_path_relative_to_sdg_base
                            )

            model_family = MODEL_FAMILY_BLANK
            if "model_id" in block["block_config"]:
                model_id = block["block_config"]["model_id"]
                model_family = MODEL_FAMILY_MAP.get(model_id, MODEL_FAMILY_BLANK)

            if "model_family" in block["block_config"]:
                model_family = block["block_config"]["model_family"]

            block["block_config"]["model_prompt"] = _get_model_prompt(model_family)

            if "operation" in block["block_config"]:
                block["block_config"]["operation"] = OPERATOR_MAP[
                    block["block_config"]["operation"]
                ]

            if "convert_dtype" in block["block_config"]:
                block["block_config"]["convert_dtype"] = CONVERT_DTYPE_MAP[
                    block["block_config"]["convert_dtype"]
                ]

            n = self.num_instructions_to_generate
            if n is not None:
                if (
                    "gen_kwargs" in block
                    and block["gen_kwargs"].get("n", None) is not None
                ):
                    block["gen_kwargs"]["n"] = n
        return flow


DEFAULT_FLOW_FILE_MAP = {
    "SimpleKnowledgeFlow": "flows/simple_knowledge.yaml",
    "SimpleFreeformSkillFlow": "flows/simple_freeform_skill.yaml",
    "SimpleGroundedSkillFlow": "flows/simple_grounded_skill.yaml",
    "MMLUBenchFlow": "flows/mmlu_bench.yaml",
    "SynthKnowledgeFlow": "flows/synth_knowledge.yaml",
    "SynthSkillsFlow": "flows/synth_skills.yaml",
    "SynthGroundedSkillsFlow": "flows/synth_grounded_skills.yaml",
    "SynthKnowledgeFlow1.5": "flows/synth_knowledge1.5.yaml",
    "GraniteResponsesFlow": "flows/granite_responses.yaml",
    "AgenticImproveFlow": "flows/agentic_improve_skill.yaml",
    "AnnotationMixtralFlow": "flows/annotate_mixtral.yaml",
    "AnnotationCoTFlow": "flows/annotate_cot.yaml",
    "LogProbsFlow": "flows/logprobs.yaml",
}
