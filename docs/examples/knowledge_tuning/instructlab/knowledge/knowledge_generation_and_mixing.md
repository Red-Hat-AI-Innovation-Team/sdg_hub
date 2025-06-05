knowledge_utils.py:
```python
# SPDX-License-Identifier: Apache-2.0

"""
Knowledge Tuning Utilities Module

This module provides utilities for processing and preparing datasets for knowledge tuning tasks.
It includes functionality for:
- Processing docling JSON files and markdown documents
- Creating QA datasets with various formats (regular, pretraining, RAFT)
- Handling document chunking and text processing
- Managing auxiliary datasets and in-context learning examples

Key Components:
- DocProcessor: Main class for processing documents and creating datasets
- Dataset Generation Functions:
  * create_knowledge_qa_dataset: Creates QA pairs from documents
  * create_knowledge_regular_ds: Creates regular knowledge datasets
  * create_knowledge_pretraining_ds: Creates pretraining format datasets
  * build_raft_dataset: Creates RAFT-style datasets with multiple documents
- Text Processing Utilities:
  * chunk_document: Splits documents into manageable chunks
  * fuse_texts: Combines short texts for better context
  * add_heading_formatting: Formats document headings

Usage:
    from sdg_hub.examples.knowledge_tuning.knowledge_utils import DocProcessor

    # Initialize processor
    processor = DocProcessor(
        parsed_doc_dir="path/to/docs",
        tokenizer="instructlab/granite-7b-lab",
        user_config_path="path/to/config.yaml"
    )

    # Process documents
    dataset = processor.get_processed_dataset()

Dependencies:
    - datasets: For dataset manipulation
    - transformers: For tokenization
    - langchain_text_splitters: For text chunking
    - tabulate: For table formatting
    - yaml: For configuration file handling

Note:
    This module is designed to work with the Granite-7b-lab model and follows
    specific formatting requirements for knowledge tuning tasks.
"""

# Standard
import json
import random
import uuid
import os
import yaml
from pathlib import Path
from typing import List
import re

# Third Party
from datasets import Dataset
from tabulate import tabulate
from transformers import AutoTokenizer
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

# Local
import sdg_hub
from sdg_hub.logger_config import setup_logger
from sdg_hub.utils.datautils import safe_concatenate_datasets

logger = setup_logger(__name__)
_DEFAULT_CHUNK_OVERLAP = 100


def create_auxiliary_dataset(generated_dataset: Dataset):
    """
    Creates an auxiliary dataset from the generated dataset by filtering out base documents
    and applying auxiliary instructions.

    Args:
        generated_dataset (Dataset): The input dataset containing various document types.

    Returns:
        Dataset or None: Returns a new dataset with auxiliary instructions applied if auxiliary
        instructions file exists, otherwise returns None.

    Note:
        The function reads auxiliary instructions from a YAML file and applies them to non-base
        documents in the dataset.
    """
    if "dataset_type" not in generated_dataset.column_names:
        return None

    aux_inst_path = os.path.join(
        os.path.dirname(sdg_hub.__file__),
        "configs/knowledge/auxilary_instructions.yaml",
    )
    print(aux_inst_path)

    if os.path.isfile(aux_inst_path):
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


def _conv_pretrain(rec):
    """
    Converts a record into a pretraining format by combining user and assistant messages.

    Args:
        rec (dict): A dictionary containing messages with user and assistant roles.

    Returns:
        dict: The modified record with messages in pretraining format.
    """
    rec["messages"] = [
        {
            "role": "pretraining",
            "content": f"<|user|>\n{rec['messages'][0]['content']}\n<|assistant|>\n{rec['messages'][1]['content']}",
        }
    ]
    return rec


def generate_knowledge_qa_dataset(
    generated_dataset: Dataset, keep_context_separate=False, keep_document_outline=False
):
    """
    Generates a knowledge QA dataset from the input dataset.

    Args:
        generated_dataset (Dataset): The input dataset containing documents and questions.
        keep_context_separate (bool, optional): If True, keeps context separate from the question.
            Defaults to False.
        keep_document_outline (bool, optional): If True, includes document outline in the context.
            Defaults to False.

    Returns:
        Dataset: A new dataset containing QA pairs in the specified format.
    """
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
            if keep_document_outline:
                messages = [
                    {
                        "role": "user",
                        "content": f"{rec['document_outline']}\n{context}\n\n{instruction}",
                    },
                    {"role": "assistant", "content": response},
                ]
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
    """
    Builds a RAFT (Retrieval-Augmented Fine-Tuning) dataset by combining multiple documents
    in the context.

    Args:
        ds (Dataset): Input dataset containing documents and questions.
        p (float): Probability threshold for including the answer document in the context.
        num_doc_in_context (int, optional): Number of documents to include in the context.
            Defaults to 4.

    Returns:
        Dataset: A new dataset with multiple documents in the context for each question.
    """
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
                    random.sample(selected_docs, k=num_doc_in_context - 1)
                    + [answer_document]
                    if len(selected_docs) >= (num_doc_in_context - 1)
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

    ds = ds.map(_pick_documents, fn_kwargs={"p": p}, remove_columns=["context"])
    return ds


def create_knowledge_regular_ds(generated_dataset: Dataset):
    """
    Creates a regular knowledge dataset by combining QA pairs and auxiliary data.

    Args:
        generated_dataset (Dataset): The input dataset containing documents and questions.

    Returns:
        Dataset: A combined dataset containing both QA pairs and auxiliary data.
    """
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
    """
    Creates a pretraining dataset by converting regular QA pairs into pretraining format.

    Args:
        generated_dataset (Dataset): The input dataset containing documents and questions.

    Returns:
        Dataset: A dataset formatted for pretraining with combined user and assistant messages.
    """
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


def fuse_texts(text_list, short_length_threshold=100):
    """
    Fuses short texts with previous longer texts to create more meaningful chunks.

    Args:
        text_list (list): List of text strings to be fused.
        short_length_threshold (int, optional): Maximum word count for a text to be considered short.
            Defaults to 100.

    Returns:
        list: List of fused text strings.
    """
    fused_texts = []
    previous_long_text = ""

    for text in text_list:
        word_count = len(text.split())

        if word_count <= short_length_threshold and previous_long_text:
            # Append the short text to the last long text
            fused_texts[-1] += "\n\n" + text
        else:
            # This is a long text, so add it to the list and remember it
            fused_texts.append(text)
            previous_long_text = text

    return fused_texts


def handle_footnote(book_element):
    """
    Handles footnote elements in the document. Currently a placeholder function.

    Args:
        book_element (dict): The footnote element to be processed.
    """
    pass


def create_tokenizer():
    """
    Creates and returns a tokenizer instance for the Granite-7b-lab model.

    Returns:
        AutoTokenizer: A tokenizer instance configured for the Granite-7b-lab model.
    """
    return AutoTokenizer.from_pretrained("instructlab/granite-7b-lab")


def get_token_count(text, tokenizer):
    """
    Calculates the number of tokens in a given text using the provided tokenizer.

    Args:
        text (str): The input text to tokenize.
        tokenizer: The tokenizer instance to use.

    Returns:
        int: The number of tokens in the text.
    """
    return len(tokenizer.tokenize(text))


def add_heading_formatting(text):
    """
    Adds markdown formatting to headings in the text.

    Args:
        text (str): The input text containing potential headings.

    Returns:
        str: The text with formatted headings.
    """
    text = text.split(".")
    # TODO: Change this from hardcoded to something that makes sense
    if len(text) > 1 and len(text[0].split(" ")) < 3:
        text = f"**{text[0]}**" + ".".join(text[1:])
    else:
        text = ".".join(text)
    return text


def generate_table_from_parsed_rep(item):
    """
    Generates a markdown table from a parsed representation.

    Args:
        item (dict): Dictionary containing table data and optional caption.

    Returns:
        str: A markdown-formatted table string with optional caption.
    """
    caption = ""
    if "text" in item:
        # print("caption: ", item["text"])
        caption = item["text"]

    data = item["data"]

    if len(data) <= 1 or len(data[0]) <= 1:
        return ""

    table = []
    for i, row in enumerate(data):
        trow = []
        for j, cell in enumerate(row):
            trow.append(cell["text"])
        table.append(trow)

    table_text = tabulate(table, tablefmt="github")
    if caption:
        table_text += f"\nCaption: {caption}\n"
    return table_text


def get_table(json_book, table_ref):
    """
    Retrieves a table from the JSON book using a table reference.

    Args:
        json_book (dict): The JSON book containing tables.
        table_ref (str): Reference to the table in the format "type/index".

    Returns:
        str: The generated table in markdown format.
    """
    parts = table_ref.split("/")
    table_text = generate_table_from_parsed_rep(json_book[parts[1]][int(parts[2])])
    return table_text


def get_table_page_number(json_book, idx):
    """
    Gets the page number for a table by looking at surrounding elements.

    Args:
        json_book (dict): The JSON book containing the table.
        idx (int): Index of the table in the book.

    Returns:
        int or None: The page number of the table, or None if not found.
    """
    # Get previous page number
    prev_page_num, next_page_num = None, None
    for book_element in json_book["main-text"][idx - 1 :: -1]:
        if "prov" in book_element:
            prev_page_num = book_element["prov"][0]["page"]
            break
    for book_element in json_book["main-text"][idx:]:
        if "prov" in book_element:
            next_page_num = book_element["prov"][0]["page"]
            break
    if prev_page_num is not None and next_page_num is not None:
        if prev_page_num == next_page_num:
            return prev_page_num
        else:
            return next_page_num
    elif prev_page_num is not None:
        return prev_page_num
    elif next_page_num is not None:
        return next_page_num


def build_chunks_from_docling_json(
    json_book,
    max_token_per_chunk,
    tokenizer,
    keep_same_page_thing_together=False,
    chunking_criteria=None,
):
    """
    Builds document chunks from a docling JSON file.

    Args:
        json_book (dict): The JSON book to be chunked.
        max_token_per_chunk (int): Maximum number of tokens per chunk.
        tokenizer: The tokenizer to use for counting tokens.
        keep_same_page_thing_together (bool, optional): If True, keeps elements from the same page together.
            Defaults to False.
        chunking_criteria (callable, optional): Custom function to determine chunk boundaries.
            Defaults to None.

    Returns:
        list: List of document chunks.
    """
    current_buffer = []
    document_chunks = []
    prev_page_number = None
    book_title = None

    for idx, book_element in enumerate(json_book["main-text"]):
        if book_element["type"] in [
            "page-footer",
            "picture",
            "reference",
            "meta-data",
            "figure",
            "page-header",
        ]:
            continue
        elif book_element["type"] == "footnote":
            handle_footnote(book_element)
            current_book_page_number = book_element["prov"][0]["page"]
        elif book_element["type"] in [
            "subtitle-level-1",
            "paragraph",
            "table",
            "title",
            "equation",
        ]:  # 'page-header',
            if book_element["type"] == "table":
                current_book_page_number = get_table_page_number(json_book, idx)
            else:
                current_book_page_number = book_element["prov"][0]["page"]
                book_text = book_element["text"]

            if book_element["type"] == "subtitle-level-1":
                if book_title is None:
                    book_title = book_text
                    book_text = f"# Title: **{book_text}**"
                else:
                    book_text = f"## **{book_text}**"

            if book_element["type"] == "title":
                book_text = f"# **{book_text}**"
            if book_element["type"] == "page-header":
                book_text = f"Page Header: **{book_text}**\n\n"

            if chunking_criteria is not None:
                # custom break function that can be used to chunk document
                if chunking_criteria(book_text):
                    document_chunks.append("\n\n".join(current_buffer))
                    current_buffer = []
            elif (
                prev_page_number is not None
                and prev_page_number != current_book_page_number
            ) and keep_same_page_thing_together:
                document_chunks.append("\n\n".join(current_buffer))
                current_buffer = []
            else:
                if (
                    get_token_count("\n\n".join(current_buffer), tokenizer)
                    >= max_token_per_chunk
                    and len(current_buffer) > 1
                ):
                    # chunk_text = '\n\n'.join(current_buffer[:-1])
                    # print(f"Current chunk size {get_token_count(chunk_text, tokenizer)} and max is {max_token_per_chunk}")
                    document_chunks.append("\n\n".join(current_buffer[:-1]))

                    if (
                        get_token_count(current_buffer[-1], tokenizer)
                        >= max_token_per_chunk
                    ):
                        # print(f"This is too big document to be left in the current buffer { get_token_count(current_buffer[-1], tokenizer)}")
                        document_chunks.append(current_buffer[-1])
                        current_buffer = []
                    else:
                        current_buffer = current_buffer[-1:]

            if book_element["type"] == "paragraph":
                book_text = add_heading_formatting(book_text)
            elif book_element["type"] == "table":
                book_text = get_table(json_book, book_element["$ref"])
            if "## References" in book_text or "## Acknowledgements" in book_text:
                # For reasearch papers we ignore everything after this sections
                break
            current_buffer.append(book_text)

        try:
            prev_page_number = current_book_page_number
        except:
            logger.error(book_element)
    if "\n\n".join(current_buffer) not in document_chunks:
        document_chunks.append("\n\n".join(current_buffer))
    return document_chunks


def _num_tokens_from_words(num_words) -> int:
    """
    Estimates the number of tokens from a given number of words.

    Args:
        num_words (int): Number of words.

    Returns:
        int: Estimated number of tokens (words * 1.3).
    """
    return int(num_words * 1.3)  # 1 word ~ 1.3 token


def _num_chars_from_tokens(num_tokens) -> int:
    """
    Estimates the number of characters from a given number of tokens.

    Args:
        num_tokens (int): Number of tokens.

    Returns:
        int: Estimated number of characters (tokens * 4).
    """
    return int(num_tokens * 4)  # 1 token ~ 4 English character


def chunk_document(documents: List, server_ctx_size, chunk_word_count) -> List[str]:
    """
    Chunks documents into smaller pieces based on word count and server context size.

    Args:
        documents (List): List of documents to be chunked.
        server_ctx_size (int): Maximum context size of the server.
        chunk_word_count (int): Maximum number of words per chunk.

    Returns:
        List[str]: List of chunked documents.

    Raises:
        TypeError: If documents is not a list or string.
        ValueError: If chunk_word_count would exceed server context size.
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


class DocProcessor:
    """
    A class for processing documents and creating datasets for knowledge tuning.

    This class handles the processing of parsed docling JSON files and markdown files,
    creating datasets suitable for knowledge tuning tasks.

    Attributes:
        parsed_doc_dir (Path): Directory containing parsed docling JSON files.
        user_config (dict): User configuration loaded from YAML file.
        docling_jsons (list): List of JSON file paths.
        tokenizer: Tokenizer instance for text processing.
    """

    def __init__(
        self,
        parsed_doc_dir: Path,
        tokenizer: str = "instructlab/granite-7b-lab",
        user_config_path: Path = None,
    ):
        """
        Initialize the DocProcessor.

        Args:
            parsed_doc_dir (Path): Directory containing parsed docling JSON files.
            tokenizer (str, optional): Name of the tokenizer to use. Defaults to "instructlab/granite-7b-lab".
            user_config_path (Path, optional): Path to user configuration file. Defaults to None.

        Raises:
            FileNotFoundError: If parsed_doc_dir or user_config_path does not exist.
        """
        self.parsed_doc_dir = self._path_validator(parsed_doc_dir)
        self.user_config = self._load_user_config(
            self._path_validator(user_config_path)
        )
        self.docling_jsons = list(self.parsed_doc_dir.glob("*.json"))
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)

    def _path_validator(self, path) -> Path:
        """
        Validates and converts a path string to a Path object.

        Args:
            path (str or Path): Path to validate.

        Returns:
            Path: Validated Path object.

        Raises:
            FileNotFoundError: If path does not exist.
        """
        if isinstance(path, str):
            path = Path(path)
            if not path.exists():
                raise FileNotFoundError(f"{path} does not exist.")
        return path

    def _load_user_config(self, user_config_path: Path) -> dict:
        """
        Loads user configuration from a YAML file.

        Args:
            user_config_path (Path): Path to the user configuration file.

        Returns:
            dict: Loaded configuration dictionary.
        """
        # load user config as yaml
        with open(user_config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _process_parsed_docling_json(self, json_fp: Path) -> Dataset:
        """
        Processes a parsed docling JSON file into a dataset.

        Args:
            json_fp (Path): Path to the JSON file.

        Returns:
            Dataset: Processed dataset containing document chunks and metadata.
        """
        logger.info(f"Processing parsed docling json file: {json_fp}")
        with open(json_fp, "r", encoding="utf-8") as f:
            data = json.load(f)

        file_name = json_fp.name.split(".")[0]
        chunks = build_chunks_from_docling_json(
            data,
            max_token_per_chunk=500,
            tokenizer=self.tokenizer,
        )
        chunks = fuse_texts(chunks, 200)
        return Dataset.from_dict(
            {
                "document": chunks,
                "document_outline": [self.user_config["document_outline"]]
                * len(chunks),
                "document_title": [file_name] * len(chunks),
                "domain": [self.user_config["domain"]] * len(chunks),
            }
        )

    def _add_icls(self, chunked_document: Dataset) -> Dataset:
        """
        Adds in-context learning examples to the dataset.

        Args:
            chunked_document (Dataset): Input dataset to add ICLs to.

        Returns:
            Dataset: Dataset with added in-context learning examples.
        """
        icl = self.user_config["seed_examples"]
        chunked_document_all_icl = []
        for icl_ in icl:
            chunked_document_all_icl.append(
                chunked_document.map(
                    lambda x: {
                        "icl_document": icl_["context"],
                        "icl_query_1": icl_["questions_and_answers"][0]["question"],
                        "icl_response_1": icl_["questions_and_answers"][0]["answer"],
                        "icl_query_2": icl_["questions_and_answers"][1]["question"],
                        "icl_response_2": icl_["questions_and_answers"][1]["answer"],
                        "icl_query_3": icl_["questions_and_answers"][2]["question"],
                        "icl_response_3": icl_["questions_and_answers"][2]["answer"],
                    }
                )
            )
        chunked_document_all_icl = safe_concatenate_datasets(chunked_document_all_icl)
        chunked_document_all_icl = chunked_document_all_icl.map(
            lambda x: {
                "chunks": chunk_document(
                    [x["document"]], server_ctx_size=4096, chunk_word_count=1024
                )
                if get_token_count(x["document"], self.tokenizer) > 1024
                else [x["document"]]
            }
        )
        df = chunked_document_all_icl.to_pandas()
        df_exploded = df.explode("chunks").reset_index(drop=True)
        new_ds = Dataset.from_pandas(df_exploded)
        new_ds = new_ds.remove_columns("document").rename_columns(
            {"chunks": "document"}
        )

        # Only keep document greater than 100 tokens
        new_ds = new_ds.filter(
            lambda x: get_token_count(x["document"], self.tokenizer) > 100
        )
        return new_ds

    def get_processed_dataset(self) -> Dataset:
        """
        Processes all parsed docling JSON files into a combined dataset.

        Returns:
            Dataset: Combined dataset containing all processed documents.
        """
        datasets = []
        for json_fp in self.docling_jsons:
            chunk_ds = self._process_parsed_docling_json(json_fp)
            chunk_ds_with_icls = self._add_icls(chunk_ds)
            datasets.append(chunk_ds_with_icls)
        return safe_concatenate_datasets(datasets)

    def get_processed_markdown_dataset(self, list_md_files: list[Path]) -> Dataset:
        """
        Processes markdown files into a dataset.

        Args:
            list_md_files (list[Path]): List of markdown file paths.

        Returns:
            Dataset: Processed dataset containing markdown content and metadata.
        """
        chunks_mds = []
        for md_file in list_md_files:
            with open(md_file, "r", encoding="utf-8") as f:
                text = f.read()
                chunks_mds.append(
                    {
                        "document": text,
                        "document_outline": self.user_config["document_outline"],
                        "document_title": md_file,
                        "domain": self.user_config["domain"],
                    }
                )
        chunk_ds = Dataset.from_list(chunks_mds)
        chunk_ds_with_icls = self._add_icls(chunk_ds)
        return chunk_ds_with_icls
```

```python
%load_ext autoreload
%autoreload 2
```

### Install SDG
```bash 
pip install sdg-hub==0.1.0a2
pip install rich datasets tabulate transformers
```
 - If you haven't already, run the document pre-processing notebook to create the seed data


```python
# Third Party
from datasets import load_dataset
from openai import OpenAI

# First Party
from sdg_hub.flow import Flow
from sdg_hub.pipeline import Pipeline
from sdg_hub.sdg import SDG
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '..', '..')))
from knowledge_utils import DocProcessor
import sys
```

### Setup OpenAI Client for interacting with the model


```python
endpoint = f"http://localhost:8000/v1"
openai_api_key = "EMPTY"
openai_api_base = endpoint

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)
teacher_model = client.models.list().data[0].id
print(teacher_model)
```

### Run SDG
- This will create knowledge flow from provided yaml file
- We will run this on small dataset for demo purposes
- For large scale generation, please use the python command provided in the next cell
- You can analyze the generated data to ensure the quality is similar to proivded QnA pairs


```python
knowledge_agentic_pipeline = "../../../src/instructlab/sdg/flows/generation/knowledge/synth_knowledge1.5.yaml"
flow_cfg = Flow(client).get_flow_from_file(knowledge_agentic_pipeline)
sdg = SDG(
    [Pipeline(flow_cfg)],
    num_workers=1,
    batch_size=1,
    save_freq=1000,
)
```


```python
number_of_samples = 5
seed_data_dir = f"sdg_demo_output/"
ds = load_dataset('json', data_files=f'{seed_data_dir}/seed_data.jsonl', split='train')
ds = ds.shuffle(seed=42).select(range(number_of_samples))
```


```python
# Checkpoint directory is used to save the intermediate datasets
generated_data = sdg.generate(ds, checkpoint_dir="Tmp")
```

### Run SDG through python command (For large scale generation)

```python
python /home/lab/sdg/scripts/generate.py --ds_path {output_dir}/seed_data.jsonl --bs 8 --num_workers 8 --save_path {output_dir}/gen.jsonl --flow ../src/instructlab/sdg/flows/generation/knowledge/synth_knowledge1.5.yaml --endpoint {teacher_endpoint_url} --checkpoint_dir {output_dir}/data_checkpoints --save_freq 2
```

### Save the generated data into training format


```python
from knowledge_utils import create_knowledge_regular_ds, create_knowledge_pretraining_ds

from datasets import concatenate_datasets

output_dir = f"sdg_demo_output/"

# Add the system prompt to final dataset if needed. For 
#  we use system prompt similar to below
system_prompt_lab = (
    "I am a LAB Instruct Model, an AI language model developed by Red Hat and IBM Research based on the granite-3.1-8b-base model. My primary role is to serve as a chat assistant."
)

# This is a general instruction tuning dataset that is mixed with generated knowledge to train LLM simultaneously on your knowledge and general instructions.
precomputed_skills_path = "<LAB precomputed skills path>"
precomputed_skills = load_dataset('json', data_files=precomputed_skills_path, split='train')

generated_ds = load_dataset('json', data_files=f'{output_dir}/gen.jsonl', split='train')

# Create Pretraining Knowledge Dataset (Also known as Phase 0.7/Phase 7)
phase_0_7_ds = create_knowledge_pretraining_ds(generated_ds)
phase_0_7_ds.to_json(f'{output_dir}/phase_0_7_ds.jsonl', orient='records', lines=True)

# Create Regular Knowledge Dataset (Also known as Phase 1.0/Phase 10)
phase_1_ds = create_knowledge_regular_ds(generated_ds)

# Mix the pre-computed skills with the regular knowledge dataset. If more than one dataset were generated simply add those in this concatenation stage.
# If you have any generated instruction data, that can be also mixed in this stage. If you only have generated skills phase 07 generation and training can be skipped.
phase_1_ds = concatenate_datasets([phase_1_ds, precomputed_skills])
phase_1_ds.to_json(f'{output_dir}/phase_1_ds.jsonl', orient='records', lines=True)
```
