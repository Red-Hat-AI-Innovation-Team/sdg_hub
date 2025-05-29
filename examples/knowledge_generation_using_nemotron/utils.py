from sdg_hub.utils.parse_and_convert import (
    create_auxiliary_dataset,
    generate_knowledge_qa_dataset,
)
from datasets import concatenate_datasets
from sdg_hub.prompts import PromptRegistry
from sdg_hub.blocks import BlockRegistry, Block
from datasets import Dataset
import re
from typing import List

from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

_DEFAULT_CHUNK_OVERLAP = 100


def _conv_pretrain(rec, tokenizer):
    if tokenizer is not None:
        rec["unmask"] = True
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
    keep_document_outline=False,
):
    knowl_train = generate_knowledge_qa_dataset(
        ds,
        keep_context_separate=keep_context_separate,
        keep_document_outline=keep_document_outline,
    )
    if no_pretrain:
        knowl_train_pretrain = knowl_train
    else:
        knowl_train_pretrain = knowl_train.map(
            _conv_pretrain, fn_kwargs={"tokenizer": tokenizer}, num_proc=10
        )
    if nemotron_format:
        knowl_train_pretrain = knowl_train_pretrain.map(
            lambda x: {
                "messages": [
                    {"content": f"detailed thinking {thinking}", "role": "system"}
                ]
                + x["messages"]
            }
        )
    if create_summary:
        summary_ds = create_auxiliary_dataset(ds)
        if no_pretrain and summary_ds:
            summary_ds_pretrain = summary_ds
        else:
            summary_ds_pretrain = summary_ds.map(
                _conv_pretrain, fn_kwargs={"tokenizer": tokenizer}, num_proc=10
            )
        if nemotron_format:
            summary_ds_pretrain = summary_ds_pretrain.map(
                lambda x: {
                    "messages": [{"content": "detailed thinking off", "role": "system"}]
                    + x["messages"]
                }
            )
        return concatenate_datasets([knowl_train_pretrain, summary_ds_pretrain])
    else:
        return knowl_train_pretrain


@PromptRegistry.register("nvidia/Llama-3_3-Nemotron-Super-49B-v1")
def nemotron_chat_template():
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
    def __init__(self, block_name: str, column_name: str) -> None:
        super().__init__(block_name=block_name)
        self.column_name = column_name

    def generate(self, samples: Dataset):
        def post_process_thinking(x):
            if "</think>" in x[self.column_name]:
                x[self.column_name] = x[self.column_name].split("</think>")[-1].lstrip()
            return x

        samples = samples.map(post_process_thinking)
        return samples


@BlockRegistry.register("RegexParserBlock")
class RegexParserBlock(Block):
    def __init__(
        self,
        block_name: str,
        column_name: str,
        parsing_pattern: str = "",
        parser_cleanup_tags: List[str] = [],
        output_cols: List[str] = [],
    ) -> None:
        super().__init__(block_name=block_name)
        self.column_name = column_name
        self.parsing_pattern = parsing_pattern
        self.parser_cleanup_tags = parser_cleanup_tags
        self.output_cols = output_cols

    def generate(self, samples: Dataset):
        if self.parsing_pattern:
            new_data = []
            for sample in samples:
                parsed_outputs = self._parse(sample[self.column_name])
                max_length = max(len(value) for value in parsed_outputs.values())
                for values in zip(
                    *(lst[:max_length] for lst in parsed_outputs.values())
                ):
                    new_data.append(
                        {**sample, **dict(zip(parsed_outputs.keys(), values))}
                    )
            samples = Dataset.from_list(new_data)
        if self.parser_cleanup_tags:
            for clean_tag in self.parser_cleanup_tags:
                samples = samples.map(
                    lambda x: {
                        column_name: x[column_name].replace(clean_tag, "")
                        for column_name in self.output_cols
                    }
                )
        return samples

    def _parse(self, generated_string):
        pattern = re.compile(self.parsing_pattern, re.DOTALL)
        all_matches = pattern.findall(generated_string)
        matches = {column_name: [] for column_name in self.output_cols}
        if all_matches and isinstance(all_matches[0], tuple):
            for match in all_matches:
                for column_name, value in zip(self.output_cols, match):
                    value = value.strip()
                    # for clean_tag in self.parser_cleanup_tags:
                    #     value = value.replace(clean_tag, "")
                    matches[column_name].append(value)
        else:
            matches[self.output_cols[0]] = (
                [match.strip() for match in all_matches] if all_matches else []
            )
        return matches


def _num_tokens_from_words(num_words) -> int:
    return int(num_words * 1.3)  # 1 word ~ 1.3 token


def _num_chars_from_tokens(num_tokens) -> int:
    return int(num_tokens * 4)  # 1 token ~ 4 English character


def chunk_document(documents: List, server_ctx_size, chunk_word_count) -> List[str]:
    """
    Iterates over the documents and splits them into chunks based on the word count provided by the user.
    Args:
        documents (list): List of documents retrieved from git (can also consist of a single document).
        server_ctx_size (int): Context window size of server.
        chunk_word_count (int): Maximum number of words to chunk a document.
    Returns:
         List[str]: List of chunked documents.
    """

    # Checks for input type error
    if isinstance(documents, str):
        documents = [documents]

    elif not isinstance(documents, list):
        raise TypeError(
            "Expected: documents to be a list, but got {}".format(type(documents))
        )

    no_tokens_per_doc = _num_tokens_from_words(chunk_word_count)
    if no_tokens_per_doc > int(server_ctx_size - 1024):
        raise ValueError(
            "Error: {}".format(
                str(
                    f"Given word count ({chunk_word_count}) per doc will exceed the server context window size ({server_ctx_size})"
                )
            )
        )
    # Placeholder for params
    content = []
    chunk_size = _num_chars_from_tokens(no_tokens_per_doc)
    chunk_overlap = _DEFAULT_CHUNK_OVERLAP

    # Using Markdown as default, document-specific chunking will be implemented in seperate pr.
    text_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.MARKDOWN,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    # Determine file type for heuristics, default with markdown
    for docs in documents:
        # Use regex to remove unnecessary dashes in front of pipe characters in a markdown table.
        docs = re.sub(r"-{2,}\|", "-|", docs)
        # Remove unnecessary spaces in front of pipe characters in a markdown table.
        docs = re.sub(r"\  +\|", " |", docs)
        temp = text_splitter.create_documents([docs])
        content.extend([item.page_content for item in temp])
    return content
