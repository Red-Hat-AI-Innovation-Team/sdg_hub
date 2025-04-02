# SPDX-License-Identifier: Apache-2.0

# Standard
from enum import Enum
from typing import Any
import json
import os
import random
import re
import uuid

# Third Party
from datasets import Dataset
import yaml

# First Party
# pylint: disable=ungrouped-imports
from sdg_hub import utils
from sdg_hub.logger_config import setup_logger
from .datautils import safe_concatenate_datasets

logger = setup_logger(__name__)


class TaxonomyType(Enum):
    KNOWLEDGE = "knowledge"
    SKILL = "skill"


def _unescape(s):
    return bytes(s, "utf-8").decode("utf-8").strip()


# This is a hack because the simple workflow returns a q/a pair as a single output.
# We could possibly try to ask for them separately, but it would cost twice the inference
# API calls. All of this is because the smallest models we use on small environments
# for testing and demos weren't good enough to follow the strict formatting instructions used
# in the full pipeline.
def _get_question(synth_example: dict):
    if "question" in synth_example:
        return synth_example["question"]

    if not synth_example.get("output"):
        raise utils.GenerateException(
            f"Error: output not found in synth_example: {synth_example}"
        )

    parts = synth_example["output"].split("?", 1)
    if len(parts) != 2:
        logger.warning(f"Failed to split generated q&a: {synth_example['output']}")
    return parts[0].strip() + "?" if len(parts) == 2 else ""


# This is also a hack. See the comment above _get_question.
def _get_response(synth_example: dict):
    if "response" in synth_example:
        return synth_example["response"]

    if "output" not in synth_example:
        raise utils.GenerateException(
            f"Error: output not found in synth_example: {synth_example}"
        )

    parts = synth_example["output"].split("?", 1)
    if len(parts) != 2:
        logger.warning(f"Failed to split generated q&a: {synth_example['output']}")
    return parts[1].strip() if len(parts) == 2 else parts[0].strip()


def _convert_to_hack_fmt(sample: dict, sys_prompt: str):
    """
    Convert a sample dictionary to contain 'system', 'user', and 'assistant' columns.

    Note: We should remove this function in the future when we resolve this issue and
    standardize the format to messages.
    """
    # Create user query message
    user_query = _unescape(_get_question(sample))
    response = _unescape(_get_response(sample))
    if "context" in sample:
        user_query = f"{sample['context']}\n\n{user_query}"

    sample["id"] = str(uuid.uuid4())
    sample["system"] = sys_prompt
    sample["user"] = user_query
    sample["assistant"] = response

    return sample


def _convert_to_messages(sample: dict, sys_prompt: str):
    """
    Convert a sample dictionary to contain 'messages'
    and 'metadata' columns required for training.
    """
    # Create user query message
    user_query = _unescape(_get_question(sample))
    response = _unescape(_get_response(sample))

    sample["id"] = str(uuid.uuid4())
    sample["messages"] = [
        {"content": sys_prompt, "role": "system"},
        {"content": user_query, "role": "user"},
        {"content": response, "role": "assistant"},
    ]

    return sample


def create_auxiliary_dataset(generated_dataset: Dataset):
    if "dataset_type" not in generated_dataset.column_names:
        return None

    # get module path of the current file
    module_dir = os.path.dirname(os.path.abspath(__file__))
    aux_inst_path = os.path.join(module_dir, "../configs/knowledge/auxilary_instructions.yaml")
    if os.path.isfile(
            aux_inst_path
    ):
        with open(aux_inst_path, "r", encoding="utf-8") as fp:
            auxiliary_inst = yaml.safe_load(fp)
    else:
        logger.error(f"auxiliary instructions file not found at {aux_inst_path}")
        return None
    auxiliary_ds = generated_dataset.filter(
        lambda x: x["dataset_type"] != "base_document"
    )
    unique_document_auxiliary = auxiliary_ds.to_pandas().drop_duplicates(
        subset=["document"]
    )
    unique_document_auxiliary = Dataset.from_pandas(unique_document_auxiliary)
    unique_document_auxiliary = unique_document_auxiliary.remove_columns(
        [
            col
            for col in unique_document_auxiliary.column_names
            if col
            not in [
                "raw_document",
                "document_outline",
                "domain",
                "dataset_type",
                "document",
            ]
        ]
    )
    unique_document_auxiliary = unique_document_auxiliary.rename_columns(
        {"raw_document": "context", "document": "response"}
    )

    def __create_auxiliary_ds(rec):
        instruction = random.choice(auxiliary_inst[rec["dataset_type"]])
        messages = [
            {"role": "user", "content": f"{rec['context']}\n\n{instruction}"},
            {"role": "assistant", "content": rec["response"]},
        ]
        metadata = json.dumps(
            {
                "dataset_type": rec["dataset_type"],
                "raw_document": rec["context"],
                "dataset": f"document_{rec['dataset_type']}",
                "domain": rec["domain"],
            }
        )
        return {"messages": messages, "metadata": metadata, "id": str(uuid.uuid4())}

    unique_document_auxiliary = unique_document_auxiliary.map(
        __create_auxiliary_ds, remove_columns=unique_document_auxiliary.column_names
    )
    return unique_document_auxiliary


def generate_knowledge_qa_dataset(
    generated_dataset: Dataset, keep_context_separate=False
):
    def __create_qa_row(rec):
        context = rec["document"]
        instruction = rec["question"]
        response = rec["response"]
        metadata = {
            "sdg_document": rec["document"],
            "domain": rec["domain"],
            "dataset": "document_knowledge_qa",
        }
        if "raw_document" in rec and "dataset_type" in rec:
            metadata.update(
                {
                    "raw_document": rec["raw_document"],
                    "dataset_type": rec["dataset_type"],
                }
            )
        metadata = json.dumps(metadata)
        if keep_context_separate:
            messages = [
                {"role": "user", "content": f"{instruction}"},
                {"role": "assistant", "content": response},
            ]
            return {
                "messages": messages,
                "metadata": metadata,
                "id": str(uuid.uuid4()),
                "context": context,
            }
        else:
            messages = [
                {"role": "user", "content": f"{context}\n\n{instruction}"},
                {"role": "assistant", "content": response},
            ]

            return {"messages": messages, "metadata": metadata, "id": str(uuid.uuid4())}

    knowledge_ds = generated_dataset.map(
        __create_qa_row, remove_columns=generated_dataset.column_names
    )
    return knowledge_ds


def build_raft_dataset(ds: Dataset, p, num_doc_in_context=4):
    all_context = list(set(ds["context"]))

    def _pick_documents(rec, p):
        answer_document = rec["context"]
        selected_docs = [e for e in all_context if e != answer_document]
        if len(selected_docs) > 0:
            if len(selected_docs) < num_doc_in_context:
                logger.info(
                    f"Number of unique document is {len(selected_docs)} which is less than {num_doc_in_context}. Using all the documents in the RAFT context"
                )
            if random.uniform(0, 1) < p:
                # golden/answer + distractor documents                                                                                         
                docs = (
                    random.sample(selected_docs, k=num_doc_in_context-1) + [answer_document]
                    if len(selected_docs) >= (num_doc_in_context-1)
                    else selected_docs + [answer_document]
                )
            else:
                # distractor documents                                                                                                         
                docs = (
                    random.sample(selected_docs, k=num_doc_in_context)
                    if len(selected_docs) >= num_doc_in_context
                    else selected_docs
                )
        else:
            logger.info("Only 1 unique document found. Turning off RAFT styling")
            docs = [answer_document]

        random.shuffle(docs)

        docs = "\n".join(([f"Document:\n{e}\n\n" for idx, e in enumerate(docs)]))
        user_idx, user_msg = [
            (idx, rec_msg)
            for idx, rec_msg in enumerate(rec["messages"])
            if rec_msg["role"] == "user"
        ][0]
        user_inst = user_msg["content"]
        rec["messages"][user_idx]["content"] = f"{docs}\n\n{user_inst}"
        rec["messages"] = rec["messages"]
        metadata = json.loads(rec["metadata"])
        metadata["dataset"] += f"_raft_p{p}"
        rec["metadata"] = json.dumps(metadata)
        return rec

    ds = ds.map(_pick_documents, fn_kwargs={"p": p} , remove_columns=["context"])
    return ds



def _conv_pretrain(rec):
    rec["messages"] = [
        {
            "role": "pretraining",
            "content": f"<|user|>\n{rec['messages'][0]['content']}\n<|assistant|>\n{rec['messages'][1]['content']}",
        }
    ]
    return rec


def create_knowledge_regular_ds(generated_dataset: Dataset):
    # Phase 1.0
    knowledge_ds = generate_knowledge_qa_dataset(
        generated_dataset, keep_context_separate=True
    )
    knowledge_ds = build_raft_dataset(knowledge_ds, p=0.4)

    auxiliary_dataset = create_auxiliary_dataset(generated_dataset)
    if auxiliary_dataset is not None:
        transformed_data = safe_concatenate_datasets([knowledge_ds, auxiliary_dataset])
    else:
        transformed_data = knowledge_ds
    return transformed_data


def create_knowledge_pretraining_ds(generated_dataset: Dataset):
    # Phase 0.7
    knowledge_ds = generate_knowledge_qa_dataset(
        generated_dataset, keep_context_separate=False
    )
    knowledge_ds = knowledge_ds.map(_conv_pretrain)

    auxiliary_dataset = create_auxiliary_dataset(generated_dataset)
    if auxiliary_dataset is not None:
        auxiliary_dataset = auxiliary_dataset.map(_conv_pretrain)
        transformed_data = safe_concatenate_datasets([knowledge_ds, auxiliary_dataset])
    else:
        transformed_data = knowledge_ds
    return transformed_data


def post_process_mcq(ds: Dataset, is_mmlu_eval: bool = False) -> Dataset:
    """Filters out badly generated data, adds dataset type column

    Args:
        ds (Dataset): mcq generated dataset from mmmlu pipeline
        is_mmlu_eval (bool, optional): _description_. Defaults to False.

    Returns:
        Dataset: Hf Dataset with new column, filtered dataset
    """
    ds = ds.filter(lambda x: ")" in x["mmlubench_answer"])
    ds = ds.filter(lambda x: "A)" in x["mmlubench_question"])
    ds = ds.add_column("dataset_type", ["mcq_qa"] * ds.num_rows)
    if is_mmlu_eval:
        return format_mmlu_style(ds)
    return ds


def extract_options(text: str) -> list[Any]:
    """regex to extract options from mcq

    Args:
        text (str): question with options/mcq choices

    Returns:
        list[Any]: options under question that match the pattern.
    """
    # Use a regular expression to find patterns and capture the text after the letter and parenthesis
    pattern = r"\b[A-Z]\) (.+)"
    matches = re.findall(pattern, text)
    return matches


def format_mmlu_style(ds: Dataset) -> Dataset:
    """Format the dataset according to lm-harness mmlu requirement.

    Args:
        ds (Dataset): input dataset

    Returns:
        Dataset: formated hf dataset
    """
    ds = ds.map(
        lambda x: {"answer": x["mmlubench_answer"][: x["mmlubench_answer"].index(")")]}
    )
    ds = ds.map(lambda x: {"choices": extract_options(x["mmlubench_question"])})
    ds = ds.map(
        lambda x: {
            "question": x["mmlubench_question"][
                : x["mmlubench_question"].index("A)")
            ].strip()
        }
    )
    ds = ds.rename_columns({"domain": "subject"})
    ds = ds.filter(lambda x: x["choices"])
    ds = ds.filter(lambda x: len(x["choices"]) == 4)
    ds = ds.filter(lambda x: x["answer"] in ["A", "B", "C", "D"])
    ds = ds.class_encode_column("answer")
    return ds


def create_mmlu_evaluation_dataset(generate_mcq_dataset: Dataset) -> Dataset:
    """Filter, format and return mcq dataset that is compatible with lm-harness for doing mmlu-style evaluation

    Args:
        generate_mcq_dataset (Dataset): sdg generated mcq dataset
    Returns:
        Dataset: MMLU MCQ datast
    """
    mmlu_dataset = post_process_mcq(generate_mcq_dataset, is_mmlu_eval=True)
    return mmlu_dataset


def create_mmlu_evaluation_yaml(task_name, eval_data_file_path, yaml_file_path):
    """
    Prepare Task Yaml that will be used in lm_eval_harness to evaluate knowledge using mmlu style metric
    """
    task_yaml = {
        "task": task_name,
        "dataset_kwargs": {"data_files": {"test": eval_data_file_path}},
        "include": "_default_mmlu_pr_template_yaml",
        "group": "mmlu_pr",
    }
    with open(yaml_file_path, "w", encoding="utf-8") as yaml_file:
        yaml.dump(task_yaml, yaml_file, default_flow_style=False)
