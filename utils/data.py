import json
import os
import random
import re
import logging

from datasets import concatenate_datasets, load_dataset
from instructlab.sdg.utils.chunking import chunk_document
from instructlab.sdg.utils.docprocessor import DocProcessor
from instructlab.sdg.utils.parse_and_convert import (
    _conv_pretrain,
    build_raft_dataset,
    create_auxiliary_dataset,
    generate_knowledge_qa_dataset,
)

from rich import print
from rich.logging import RichHandler


logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler()],
)
logger = logging.getLogger(__name__)


def is_string_complete(s):
    # Check if the string ends with common sentence-ending punctuation
    if s.endswith((".", "!", "?", '"', "'", "|")):
        return True

    # Check if the string ends with an incomplete word
    if re.search(r"\b\w+$", s) and not re.search(r"\b\w+\b$", s):
        return False

    return False


def is_multiple_choice_question(s):
    # Define regex patterns to detect multiple-choice options (a), b), c), a., b., c., etc.)
    patterns = [
        re.compile(r"\b[a-z]\)\s", re.IGNORECASE),
        re.compile(r"\b[a-z]\.\s", re.IGNORECASE),
    ]

    # Check if any of the patterns are found in the string
    for pattern in patterns:
        if pattern.search(s):
            return True
    return False


def add_system_message(sample: dict, sys_prompt: str) -> dict:
    # check if the messages have role system
    has_system = False
    for msg in sample["messages"]:
        if msg["role"] == "system":
            has_system = True
            msg["content"] = sys_prompt

    if not has_system:
        sample["messages"].insert(0, {"role": "system", "content": sys_prompt})

    return sample


def postprocess_and_save(
    ds_path,
    dataset_save_path,
    sys_prompt,
    precomputed_skills_path="pre_computed_skills.jsonl",
    dataset_name='custom_generated_knowledge_ds'
):
    ds = load_dataset("json", data_files=ds_path, split="train")
    ds = ds.filter(lambda x: len(x["document"].split(" ")) > 150, num_proc=72)
    ds = ds.map(
        lambda x: {
            "question": x["question"].replace("[END]", "").strip(),
            "response": x["response"].replace("[END]", "").strip(),
        },
        num_proc=72,
    )
    ds = ds.filter(lambda x: is_string_complete(x["question"]), num_proc=72)
    ds = ds.filter(lambda x: is_string_complete(x["response"]), num_proc=72)
    summary_ds = create_auxiliary_dataset(ds)

    summary_ds_07 = summary_ds.map(_conv_pretrain)
    knowl_train = generate_knowledge_qa_dataset(ds, False)
    knowl_train_pretrain = knowl_train.map(_conv_pretrain)
    phase07_train = concatenate_datasets([knowl_train_pretrain, summary_ds_07])
    phase07_train = phase07_train.add_column("dataset", [dataset_name] * phase07_train.num_rows)
    phase07_train = phase07_train.map(
        add_system_message, fn_kwargs={"sys_prompt": sys_prompt}, num_proc=72
    )

    phase07_train.to_json(
        os.path.join(dataset_save_path, "phase07_train.jsonl"),
        orient="records",
        lines=True,
    )

    knowl_train = generate_knowledge_qa_dataset(ds, True)
    knowledge_ds = build_raft_dataset(knowl_train, p=0.4)
    # Concatenate pre-train with phase 1 raft as replay buffer
    knowledge_ds = concatenate_datasets([knowledge_ds, knowl_train_pretrain])
    phase10 = concatenate_datasets([knowledge_ds, summary_ds])
    phase10 = phase10.add_column("group", ["knowledge"] * phase10.num_rows)

    precomputed_skills = load_dataset("json", data_files=precomputed_skills_path, split="train")
    phase10_skills = concatenate_datasets([precomputed_skills, phase10])
    phase10_skills = phase10_skills.map(
        add_system_message, fn_kwargs={"sys_prompt": sys_prompt}, num_proc=72
    )
    phase10 = phase10.add_column("dataset", [dataset_name] * phase10.num_rows)
    
    phase10_skills.to_json(
        os.path.join(dataset_save_path, "phase10_train.jsonl"),
        orient="records",
        lines=True,
    )
    logger.info("Saved datasets to %s", dataset_save_path)


def pretty_print_dict(path):
    data = load_dataset("json", data_files=path, split="train")
    print(json.dumps(data[random.randint(0, len(data))], indent=4))
