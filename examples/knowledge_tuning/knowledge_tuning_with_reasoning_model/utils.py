"""
utils.py

This module provides utilities to construct, format, and post-process datasets for training 
large language models using structured prompts and outputs. It supports:

- Converting datasets into message-style formats for pretraining or instruction tuning
- Creating summary and QA-based datasets from knowledge sources
- Defining reusable blocks for regex parsing and custom output transformations
- Registering prompt templates tailored to specific LLM architectures like Nemotron

Intended for use within SDG Hub pipelines to modularize data preparation logic.
"""

from datasets import concatenate_datasets, Dataset
from sdg_hub.prompts import PromptRegistry
from sdg_hub.blocks import BlockRegistry, Block
import re
from typing import List
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from knowledge_utils import (
    create_auxiliary_dataset,
    generate_knowledge_qa_dataset
)

def _conv_pretrain(rec, tokenizer):
    """
    Internal utility to convert a record to pretraining format.

    If a tokenizer is provided, it adds an 'unmask' flag to the record. Otherwise,
    it reformats the record into a two-turn conversation using special role tags.

    Args:
        rec (dict): A dataset sample containing a `messages` list.
        tokenizer (Tokenizer or None): Tokenizer used to decide formatting logic.

    Returns:
        dict: Updated record with pretraining-style message formatting.
    """
    if tokenizer is not None:
        rec['unmask'] = True
        return rec
    rec["messages"] = [
        {
            "role": "pretraining",
            "content": f"<|user|>\n{rec['messages'][0]['content']}\n<|assistant|>\n{rec['messages'][1]['content']}",
        }
    ]
    return rec


def create_training_mix(
    ds,
    tokenizer,
    thinking="on",
    create_summary=True,
    nemotron_format=True,
    keep_context_separate=False,
    no_pretrain=False,
    keep_document_outline=False
):
    """
    Creates a training dataset by combining knowledge-based QA pairs with optional summaries.
    Applies model-specific message formatting and pretraining conversion.

    Args:
        ds (Dataset): Original document dataset.
        tokenizer: Tokenizer instance (optional) used to add unmasking for pretraining.
        thinking (str): Flag for system prompt ("on" or "off").
        create_summary (bool): Whether to include auxiliary summaries.
        nemotron_format (bool): Whether to wrap messages in Nemotron system prompts.
        keep_context_separate (bool): Whether to preserve context boundaries in QA generation.
        no_pretrain (bool): Whether to skip pretraining formatting.
        keep_document_outline (bool): Whether to preserve document structure in output.

    Returns:
        Dataset: Concatenated or formatted dataset ready for training.
    """
    knowl_train = generate_knowledge_qa_dataset(ds, keep_context_separate=keep_context_separate, keep_document_outline=keep_document_outline)

    if no_pretrain:
        knowl_train_pretrain = knowl_train
    else:
        knowl_train_pretrain = knowl_train.map(_conv_pretrain, fn_kwargs={"tokenizer": tokenizer}, num_proc=10)

    if nemotron_format:
        knowl_train_pretrain = knowl_train_pretrain.map(lambda x: {
            'messages': [{'content': f'detailed thinking {thinking}', 'role': 'system'}] + x['messages']
        })

    if create_summary:
        summary_ds = create_auxiliary_dataset(ds)
        if no_pretrain and summary_ds:
            summary_ds_pretrain = summary_ds
        else:
            summary_ds_pretrain = summary_ds.map(_conv_pretrain, fn_kwargs={"tokenizer": tokenizer}, num_proc=10)
        if nemotron_format:
            summary_ds_pretrain = summary_ds_pretrain.map(lambda x: {
                'messages': [{'content': 'detailed thinking off', 'role': 'system'}] + x['messages']
            })
        return concatenate_datasets([knowl_train_pretrain, summary_ds_pretrain])
    else:
        return knowl_train_pretrain


@PromptRegistry.register("nvidia/Llama-3_3-Nemotron-Super-49B-v1")
def nemotron_chat_template():
    """
    Returns a Jinja-style prompt template for formatting conversational data
    compatible with Nemotron's Llama-3 chat model. The template wraps system
    and user/assistant messages using header and end-of-turn markers.

    Returns:
        str: A multi-line string representing the template.
    """
    return """{{- bos_token }}
{{- "<|start_header_id|>system<|end_header_id|>\n\n" }}detailed thinking on{{- "<|eot_id|>" }}
{%- for message in messages %}
  {%- if message['role'] == 'assistant' and '</think>' in message['content'] %}
    {%- set content = message['content'].split('</think>')[-1].lstrip() %}
  {%- else %}
    {%- set content = message['content'] %}
  {%- endif %}
  {{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' + content | trim + '<|eot_id|>' }}
{%- endfor %}
{%- if add_generation_prompt %}
  {{- '<|start_header_id|>assistant<|end_header_id|>\n\n' }}
{%- endif %}"""


@BlockRegistry.register("PostProcessThinkingBlock")
class PostProcessThinkingBlock(Block):
    """
    A block that removes reasoning scaffolds from model outputs.

    Specifically, it strips content before `</think>` in a designated column,
    leaving only the final answer or user-facing response.

    Args:
        block_name (str): Unique identifier for the block.
        column_name (str): Name of the column to post-process.

    Returns:
        Dataset: The cleaned dataset with modified column values.
    """
    def __init__(self, block_name: str, column_name: str) -> None:
        super().__init__(block_name=block_name)
        self.column_name = column_name

    def generate(self, samples: Dataset):
        """
        Removes content before the `</think>` tag in each record of the dataset.

        Args:
            samples (Dataset): Dataset to process.

        Returns:
            Dataset: Dataset with cleaned text.
        """
        def post_process_thinking(x):
            if '</think>' in x[self.column_name]:
                x[self.column_name] = x[self.column_name].split('</think>')[-1].lstrip()
            return x
        return samples.map(post_process_thinking)


@BlockRegistry.register("RegexParserBlock")
class RegexParserBlock(Block):
    """
    A block that extracts structured data from a column using a regular expression.

    This is useful for postprocessing generated strings that contain predictable tags
    or fields and converting them into separate dataset columns.

    Args:
        block_name (str): Unique name of the block.
        column_name (str): Column containing the string to parse.
        parsing_pattern (str): Regex pattern with groups for extraction.
        parser_cleanup_tags (List[str]): List of tags to remove after parsing.
        output_cols (List[str]): Column names for parsed output fields.
    """
    def __init__(
        self, 
        block_name: str,
        column_name: str,
        parsing_pattern: str = "",
        parser_cleanup_tags: List[str] = [],
        output_cols: List[str] = []
    ) -> None:
        super().__init__(block_name=block_name)
        self.column_name = column_name
        self.parsing_pattern = parsing_pattern
        self.parser_cleanup_tags = parser_cleanup_tags
        self.output_cols = output_cols

    def generate(self, samples: Dataset):
        """
        Applies the regex parser to each sample and creates new columns 
        from extracted groups. Optionally removes cleanup tags.

        Args:
            samples (Dataset): Input dataset.

        Returns:
            Dataset: Dataset with parsed fields and cleaned content.
        """
        if self.parsing_pattern:
            new_data = []
            for sample in samples:
                parsed_outputs = self._parse(sample[self.column_name])
                max_length = max(len(value) for value in parsed_outputs.values())
                for values in zip(*(lst[:max_length] for lst in parsed_outputs.values())):
                    new_data.append({**sample, **dict(zip(parsed_outputs.keys(), values))})
            samples = Dataset.from_list(new_data)

        if self.parser_cleanup_tags:
            for clean_tag in self.parser_cleanup_tags:
                samples = samples.map(lambda x: {
                    column_name: x[column_name].replace(clean_tag, "") for column_name in self.output_cols
                })

        return samples

    def _parse(self, generated_string):
        """
        Parses the input string using the stored regex pattern and 
        returns matched groups in a dictionary keyed by output columns.

        Args:
            generated_string (str): Text to extract structured fields from.

        Returns:
            dict: Mapping from output column names to lists of matched strings.
        """
        pattern = re.compile(self.parsing_pattern, re.DOTALL)
        all_matches = pattern.findall(generated_string)
        matches = {column_name: [] for column_name in self.output_cols}

        if all_matches and isinstance(all_matches[0], tuple):
            for match in all_matches:
                for column_name, value in zip(self.output_cols, match):
                    matches[column_name].append(value.strip())
        else:
            matches[self.output_cols[0]] = (
                [match.strip() for match in all_matches] if all_matches else []
            )
        return matches
