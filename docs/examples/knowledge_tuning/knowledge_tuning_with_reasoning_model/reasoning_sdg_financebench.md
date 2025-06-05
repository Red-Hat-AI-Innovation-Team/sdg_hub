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

utils.py:
```yaml
# =============================================================================
# Reasoning Model Knowledge Tuning Utilities
# =============================================================================
# This module provides essential utilities for knowledge-based question answering
# systems with reasoning capabilities. It includes:
#   - Dataset creation and mixing for training
#   - Pretraining format conversion
#   - Thinking step processing
#   - Structured content parsing
# 
# Key components:
#   - create_training_mix: Creates combined training datasets
#   - PostProcessThinkingBlock: Handles thinking step removal
#   - RegexParserBlock: Extracts structured information
#   - nemotron_chat_template: Provides chat formatting for Nemotron model
# =============================================================================

"""
Utility functions and classes for knowledge tuning with reasoning models.

This module provides tools for creating and processing training datasets for knowledge-based
question answering systems, with support for reasoning and thinking steps. It includes
functions for dataset creation, conversion, and processing blocks for handling thinking
steps and regex-based parsing.

The module integrates with the SDG Hub framework and provides specialized blocks for
post-processing thinking steps and parsing structured outputs using regular expressions.
"""

from datasets import concatenate_datasets
from sdg_hub.prompts import PromptRegistry
from sdg_hub.blocks import BlockRegistry, Block
from datasets import Dataset
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
    Convert a record to pretraining format.

    Args:
        rec (dict): The input record containing messages
        tokenizer: Optional tokenizer for processing

    Returns:
        dict: The converted record with pretraining format
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

def create_training_mix(ds, tokenizer, thinking="on", create_summary=True, nemotron_format=True, 
                       keep_context_separate=False, no_pretrain=False, keep_document_outline=False):
    """
    Create a mixed training dataset combining knowledge QA and summary data.

    Args:
        ds (Dataset): Input dataset
        tokenizer: Tokenizer for processing
        thinking (str, optional): Thinking mode setting. Defaults to "on"
        create_summary (bool, optional): Whether to include summary data. Defaults to True
        nemotron_format (bool, optional): Whether to use Nemotron format. Defaults to True
        keep_context_separate (bool, optional): Whether to keep context separate. Defaults to False
        no_pretrain (bool, optional): Whether to skip pretraining conversion. Defaults to False
        keep_document_outline (bool, optional): Whether to keep document outline. Defaults to False

    Returns:
        Dataset: Combined training dataset
    """
    knowl_train = generate_knowledge_qa_dataset(ds, keep_context_separate=keep_context_separate, keep_document_outline=keep_document_outline)
    if no_pretrain:
        knowl_train_pretrain = knowl_train
    else:
        knowl_train_pretrain = knowl_train.map(_conv_pretrain, fn_kwargs={"tokenizer": tokenizer}, num_proc=10)
    if nemotron_format:
        knowl_train_pretrain = knowl_train_pretrain.map(lambda x: {'messages': [{'content': f'detailed thinking {thinking}', 'role': 'system'}] + x['messages']})
    if create_summary:
        summary_ds = create_auxiliary_dataset(ds)
        if no_pretrain and summary_ds:
            summary_ds_pretrain = summary_ds
        else:
            summary_ds_pretrain = summary_ds.map(_conv_pretrain, fn_kwargs={"tokenizer": tokenizer}, num_proc=10)
        if nemotron_format:
            summary_ds_pretrain = summary_ds_pretrain.map(lambda x: {'messages': [{'content': 'detailed thinking off', 'role': 'system'}] + x['messages']})
        return concatenate_datasets([knowl_train_pretrain, summary_ds_pretrain])
    else:
        return knowl_train_pretrain

@PromptRegistry.register("nvidia/Llama-3_3-Nemotron-Super-49B-v1")
def nemotron_chat_template():
    """
    Generate the chat template for Nemotron model.

    Returns:
        str: The formatted chat template string
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
    Block for post-processing thinking steps in generated content.

    This block removes thinking steps (content between <think> tags) from the specified column
    in the dataset, keeping only the final response.

    Args:
        block_name (str): Name of the block
        column_name (str): Name of the column to process
    """
    def __init__(self, block_name: str, column_name: str) -> None:
        super().__init__(block_name=block_name)  
        self.column_name = column_name
    
    def generate(self, samples: Dataset):
        """
        Process the samples to remove thinking steps.

        Args:
            samples (Dataset): Input dataset

        Returns:
            Dataset: Processed dataset with thinking steps removed
        """
        def post_process_thinking(x):
            if '</think>' in x[self.column_name]:
                x[self.column_name] = x[self.column_name].split('</think>')[-1].lstrip()
            return x
        samples = samples.map(post_process_thinking)
        return samples

@BlockRegistry.register("RegexParserBlock")
class RegexParserBlock(Block):
    """
    Block for parsing structured content using regular expressions.

    This block extracts structured information from text using regex patterns and
    can optionally clean up specific tags from the extracted content.

    Args:
        block_name (str): Name of the block
        column_name (str): Name of the column to process
        parsing_pattern (str, optional): Regex pattern for parsing. Defaults to ""
        parser_cleanup_tags (List[str], optional): Tags to clean up from parsed content. Defaults to []
        output_cols (List[str], optional): Names of output columns for parsed content. Defaults to []
    """
    def __init__(self, block_name: str, column_name: str, parsing_pattern: str="", 
                 parser_cleanup_tags: List[str]=[], output_cols: List[str]=[]) -> None:
        super().__init__(block_name=block_name)
        self.column_name = column_name
        self.parsing_pattern = parsing_pattern
        self.parser_cleanup_tags = parser_cleanup_tags
        self.output_cols = output_cols

    def generate(self, samples: Dataset):
        """
        Process the samples using regex parsing.

        Args:
            samples (Dataset): Input dataset

        Returns:
            Dataset: Processed dataset with parsed content
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
               samples = samples.map(lambda x: {column_name: x[column_name].replace(clean_tag, "") for column_name in self.output_cols})
        return samples

    def _parse(self, generated_string):
        """
        Parse the input string using the configured regex pattern.

        Args:
            generated_string (str): Input string to parse

        Returns:
            dict: Dictionary mapping output column names to parsed values
        """
        pattern = re.compile(self.parsing_pattern, re.DOTALL)
        all_matches = pattern.findall(generated_string)
        matches = {column_name: [] for column_name in self.output_cols}
        if all_matches and isinstance(all_matches[0], tuple):
            for match in all_matches:
                for column_name, value in zip(self.output_cols, match):
                    value = value.strip()
                    matches[column_name].append(value)
        else:
            matches[self.output_cols[0]] = (
                [match.strip() for match in all_matches] if all_matches else []
            )
        return matches
```


```python
%load_ext autoreload
%autoreload 2
```


```python
import os

data_base_path = os.getenv("DATA_BASE_PATH", "./data")  # default or raise with message
if not data_base_path:
    raise RuntimeError(
        "DATA_BASE_PATH env-var is required – point it to your dataset root."
    )
```

### Intro
- For this reasoning demo we will use the financeBench documents to customize a model on SEC fillings


```python
from datasets import load_dataset, concatenate_datasets
from openai import OpenAI

endpoint = f"http://localhost:8000/v1"
openai_api_key = "EMPTY"
openai_api_base = endpoint

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)
teacher_model = client.models.list().data[0].id
```


<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7fbfbf; text-decoration-color: #7fbfbf">[14:57:13] </span><span style="color: #000080; text-decoration-color: #000080">INFO    </span> HTTP Request: <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">GET</span> <span style="color: #0000ff; text-decoration-color: #0000ff; text-decoration: underline">http://localhost:8000/v1/models</span> <span style="color: #008000; text-decoration-color: #008000">"HTTP/1.1 200 OK"</span>             <a href="file:///home/lab/.conda/envs/research_sdg/lib/python3.10/site-packages/httpx/_client.py" target="_blank"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">_client.py</span></a><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">:</span><a href="file:///home/lab/.conda/envs/research_sdg/lib/python3.10/site-packages/httpx/_client.py#1025" target="_blank"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">1025</span></a>
</pre>



### Do Document Processing on Finance Bench Documents
- git clone https://github.com/patronus-ai/financebench.git to get pdfs
- We will use docling and its hybrid chunker to parse and convert these pdfs into chunks


```python
data_dir = 'document_collection/ibm-annual-report'
!OMP_NUM_THREADS=32 mamba run -n docling python ../scripts/docparser_v2.py --input-dir financebench/finance_documents/ --output-dir financebench/finance_documents/ -c docling_v2_config.yaml
```


```python
import yaml
config = {
    "pipeline": {
        "ocr": {
            "enabled": True,      # Enable/disable OCR processing
            "languages": ["es"],  # List of language codes (e.g., eng, fra, deu)
        },
        "tables": {
            "enabled": True,        # Enable/disable table detection
            "cell_matching": True,  # Enable/disable cell matching in tables
        },
        "performance": {
            "threads": 32,          # Number of processing threads
            "device": "auto"       # Device selection (auto, cpu, gpu)
        }
    },
    "export": {
        "formats": {
            "json": False,      # Deep Search JSON format
            "text": False,      # Plain text
            "markdown": True,  # Markdown with structure
            "html": True,     # HTML with styling
            "doctags": False   # Document tokens
        }
    }
}

with open('docling_v2_config.yaml', 'w') as f:
    yaml.dump(config, f)

```


```python
import os
from datasets import load_dataset

ds = load_dataset("PatronusAI/financebench", split="train")
ds.filter(lambda x: '3M' in x['doc_name'])
all_docs = ds['doc_name']
all_pdfs = [e.replace(".pdf", "") for e in os.listdir("financebench/pdfs/")]
all_pdfs = [e for e in all_pdfs if e in all_docs]
import shutil
# Copy the pdfs to the financebench/pdfs_filtered/ directory
os.makedirs("financebench/pdfs_filtered/", exist_ok=True)
for pdf in all_pdfs:
    shutil.copy(f"financebench/pdfs/{pdf}.pdf", f"financebench/pdfs_filtered/{pdf}.pdf")
```


```python
# We can reuse ICLs from the previous quality data
icls = [{
  "icl_document": """**Investor Briefing Summary: 3M 2022 Financial Highlights**  

**Total Company Performance (YoY Change)**  
- **Revenue**: $34.229B (↓3.2% from $35.355B in 2021)  
- **Operating Income**: $6.539B (↓11.3% from $7.369B in 2021)  
- **Operating Margin**: 19.1% (↓1.7% from 20.8% in 2021)  

**Segment Performance (YoY Changes)**  
1. **Safety and Industrial**  
   - Revenue: $11.604B (↓3.2%)  
   - Operating Income: $1.199B (↓51.4%)  
   *Sharp decline due to litigation costs (PFAS-related) and productivity headwinds.*  

2. **Transportation and Electronics**  
   - Revenue: $8.902B (↓3.9%)  
   - Operating Income: $1.012B (↓46.2%)  
   *Impacted by divestitures and currency translation.*  

3. **Health Care**  
   - Revenue: $8.421B (↓2.0%)  
   - Operating Income: $1.815B (↓10.9%)  
   *Most resilient segment, with organic growth offset by currency.*  

4. **Consumer**  
   - Revenue: $5.298B (↓3.9%)  
   - Operating Income: $994M (↓14.4%)  
   *Organic sales declined 0.9% amid pricing and logistics challenges.*  

5. **Corporate and Unallocated**  
   - Operating Income: $1.519B (vs. $(176)M in 2021)  
   *Driven by divestiture gains and reduced unallocated costs.*""",

  "icl_query_1": "How did 3M’s overall financial performance change in 2022 compared to 2021?",
  "icl_response_1": "In 2022, 3M’s total revenue declined by 3.2% year-over-year, falling from $35.355 billion in 2021 to $34.229 billion. Operating income also decreased by 11.3%, dropping from $7.369 billion to $6.539 billion, while the operating margin declined from 20.8% to 19.1%.",
  "icl_query_2": "What factors contributed to the significant decline in operating income in the Safety and Industrial segment?",
  "icl_response_2": "The Safety and Industrial segment saw a 51.4% drop in operating income, down to $1.199 billion. This sharp decline was primarily attributed to litigation costs related to PFAS and productivity headwinds.",

  "icl_query_3": "Which segment showed the most resilience in 3M’s 2022 performance, and what challenges did it face?",
  "icl_response_3": "The Health Care segment was the most resilient in 2022, experiencing the smallest year-over-year revenue decline of 2.0% and a 10.9% drop in operating income. Its organic growth helped offset currency-related headwinds.",
}
]

```


```python
import glob
from utils import chunk_document
from datasets import Dataset

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("nvidia/Llama-3.1-Nemotron-Nano-8B-v1")


def break_large_chunks(text, max_chunk_size=7000):
    chunks = chunk_document([text], server_ctx_size=50000, chunk_word_count=max_chunk_size)
    return chunks

all_mds = glob.glob("financebench/finance_documents/*.md")
finance_bench_ds = []
for md in all_mds:
    print(md)
    with open(md, "r") as f:
        text = f.read()
    chunks = chunk_document([text], server_ctx_size=40000, chunk_word_count=7000)
    new_chunks = []
    for chunk in chunks:
        if len(tokenizer.encode(chunk)) > 7000:
            new_chunks.extend(break_large_chunks(chunk, 5000))
        else:
            new_chunks.append(chunk)
    document_outline = md.split("/")[-1].replace(".md", "").replace("_", " ")
    finance_bench_ds.extend([{'document_outline': document_outline, 'document': chunk, **icls[0]} for chunk in new_chunks])

finance_bench_ds = Dataset.from_list(finance_bench_ds)
finance_bench_ds.to_json("3M_SEC_filings_seed_data.jsonl", orient="records", lines=True)
```

### Analyze Chunks

- Our aim is to get similar chunk size distribution (in terms of tokens) as Quality dataset. 
- We are using the markdown chunker that uses words to approximate chunk size in tokens


```python
finance_bench_ds = finance_bench_ds.map(lambda x: {"token_size": len(tokenizer.encode(x['document']))}, batched=False)
# Plot the distribution of SEC filings data for 3M
import matplotlib.pyplot as plt
plt.hist(finance_bench_ds['token_size'], bins=100)
plt.show()
```


    
![png](output_11_0.png)
    



```python
quality_seed = load_dataset("json", data_files="seed_data.jsonl", split="train")
quality_seed = quality_seed.map(lambda x: {"token_size": len(tokenizer.encode(x['document']))}, batched=False)
# Plot the distribution of the token size of Quality Data
import matplotlib.pyplot as plt
plt.hist(quality_seed['token_size'], bins=100)
plt.show()
```


    
![png](output_12_0.png)
    


## SDG Generation

### Changes to Prompt
- Once our teacher model is hosted and visible through the Openai client we can start setting up the flow
- Coming from previous notebooks we will make 1 small change to the summarization prompt. We will add this:

    `- Include as much of the document as possible to create a comprehensive summary`

    `- If there are tables include all the data of the table in the summary`

- This is required to ensure finance documents, that have densely packed information, does not get removed when summarizing

### flows/synth_knowledge_reasoning_nemotron_super_49b_summary_diversity.yaml:
```yaml
- block_type: LLMBlock
  block_config:
    block_name: gen_summary_instructions
    config_path: prompts/generate_summary_inst.yaml
    model_id: nvidia/Llama-3_3-Nemotron-Super-49B-v1
    output_cols:
      - summary_instruction
  gen_kwargs:
    max_tokens: 4096
    temperature: 0.6
    top_p: 0.95
    n: 2
    seed: 43146

- block_type: PostProcessThinkingBlock
  block_config:
    block_name: post_process_thinking_instruction
    column_name: summary_instruction

- block_type: RegexParserBlock
  block_config:
    block_name: regex_parser
    column_name: summary_instruction
    parsing_pattern: "(?:^|\\n)\\s*\\d+[\\.\\)]\\s*([^\\n]+)"
    parser_cleanup_tags:
      - "[END]"
    output_cols:
      - summary_instruction

- block_type: LLMBlock
  block_config:
    block_name: gen_detailed_summary
    config_path: prompts/generate_summary.yaml
    model_id: nvidia/Llama-3_3-Nemotron-Super-49B-v1
    output_cols:
      - document_summary
  gen_kwargs:
    max_tokens: 4096
    temperature: 0.6
    top_p: 0.95
    n: 1

- block_type: PostProcessThinkingBlock
  block_config:
    block_name: post_process_thinking_summary
    column_name: document_summary

- block_type: RenameColumns
  block_config:
    block_name: rename_to_document_column
    columns_map:
      document: raw_document
      document_summary: document

- block_type: LLMBlock
  block_config:
    block_name: knowledge question generation
    config_path: prompts/generate_questions.yaml
    model_id: nvidia/Llama-3_3-Nemotron-Super-49B-v1
    output_cols:
      - question
  gen_kwargs:
    temperature: 0.6
    max_tokens: 1024
    top_p: 0.95
    n: 1

- block_type: PostProcessThinkingBlock
  block_config:
    block_name: post_process_thinking
    column_name: question


- block_type: RegexParserBlock
  block_config:
    block_name: regex_parser
    column_name: question
    parsing_pattern: "\\[(?:Question|QUESTION)\\]\\s*(.*?)\\s*(?=\\[(?:Question|QUESTION)\\]|$)"
    parser_cleanup_tags:
      - "[END]"
    output_cols:
      - question


- block_type: LLMBlock
  block_config:
    block_name: knowledge answer generation
    config_path: prompts/generate_answers.yaml
    model_id: nvidia/Llama-3_3-Nemotron-Super-49B-v1
    output_cols:
      - response
  gen_kwargs:
    temperature: 0.6
    max_tokens: 4096
    top_p: 0.95
    n: 1

- block_type: RegexParserBlock
  block_config:
    block_name: regex_parser
    column_name: response
    parser_cleanup_tags:
      - "[END]"
      - "[ANSWER]"
      - "assistant"
    output_cols:
      - response

- block_type: LLMBlock
  block_config:
    block_name: eval_faithfulness_qa_pair
    config_path: configs/knowledge/evaluate_faithfulness.yaml
    model_id: nvidia/Llama-3_3-Nemotron-Super-49B-v1
    output_cols:
      - explanation
      - judgment
  gen_kwargs:
    max_tokens: 2048
```
generate_summary_inst.yaml:
```yaml
system: You are an AI assistant that is expert at summarizing text.

introduction: |
  Given below document, analyze it, and generate a list of 10 diverse instructions for summarizing it. 
  Each instruction should vary in perspective, tone, or purpose and should be relevant to the document. Keep them short and distinct.

principles: ""

examples: |
  Example:
  1. Summarize the article in simple terms for a 10-year-old.
  2. Highlight the major arguments and counterarguments.
  3. Provide a summary focusing on implications for future research.

generation: |
  Document:
  {{document_outline}}

  {{document}}
  
  Now given above document, generate 10 more diverse instructions for summarizing it.

start_tags: [""]
end_tags: [""]
```

generate_summary.yaml:
```yaml
system: You are an AI assistant that is expert at summarizing text.

introduction: |
  Given below document, summarize it using the following instructions:
  {{summary_instruction}}

principles: 
  - Include as much of the document as possible to create a comprehensive summary
  - If there are tables include all the data of the table in the summary

examples: ""

generation: |
  Document:
  {{document_outline}}
  {{document}}
  

start_tags: [""]
end_tags: [""]
```

generate_questions.yaml:
```yaml
system: You are a very knowledgeable AI Assistant that will faithfully assist the user with their task.

introduction: Develop a series of educational questions from a chapter in a {{domain}} textbook. 

principles: |
  The questions should:
  * Self-contained – understandable without needing to reference tables, figures, or specific text sections.
  * Focus on the provided example and follow the format and style of the provided examples.
  * Relevant to the subject – based on the textbook’s domain (e.g., legal, scientific, etc.).
  * Independently answerable – avoid direct references to theorems, figures, or text numbers.
  * Varied in difficulty - Make difficult same as the provided examples.
  * Use same format as the provided examples.

  Strictly follow this format for each question your generate while responding

  [QUESTION]
  <Insert question here>
  [END]


examples: |
  Here are some examples of questions:

  [Document]
  {{icl_document}}

  [QUESTION]
  {{icl_query_1}}
  [END]

  [QUESTION]
  {{icl_query_2}}
  [END]

  [QUESTION]
  {{icl_query_3}}
  [END]

generation: |
  Here is the document:
  
  [DOCUMENT]
  {{document_outline}}
  {{document}}

start_tags: [""]
end_tags: [""]
```

generate_answers.yaml:
```yaml
system: You are a very knowledgeable AI Assistant that will faithfully assist the user with their task.

introduction: Answer the question based on the provided document.  

principles: |
  The answers should:
  * The answer is grounded in the provided document.
  * Follows the format and style of the provided examples.
  * Directly answers the question.
  Strictly follow this format for each question your generate while responding

  [ANSWER]
  <Insert answer here>
  [END]


examples: |
  Here are some examples of answers for given questions for a document:

  [Document]
  {{icl_document}}

  [QUESTION]
  {{icl_query_1}}

  [ANSWER]
  {{icl_response_1}}
  [END]

  [QUESTION]
  {{icl_query_2}}

  [ANSWER]
  {{icl_response_2}}
  [END]


generation: |
  Here is the document:
  
  [DOCUMENT]
  {{document_outline}}
  {{document}}

  [QUESTION]
  {{question}}

  [ANSWER]

start_tags: [""]
end_tags: [""]
```

```python
# =============================================================================
# Reasoning Model Processing Blocks
# =============================================================================
# This module provides specialized processing blocks for handling reasoning model outputs:
#   - PostProcessThinkingBlock: Removes thinking steps from model outputs
#   - RegexParserBlock: Extracts structured information using regex patterns
#
# These blocks are designed to work with the SDG Hub framework and process
# datasets containing model-generated content with reasoning steps.
# =============================================================================

"""
Processing blocks for reasoning model outputs.

This module contains specialized blocks for processing and transforming outputs from
reasoning models. It provides functionality for handling thinking steps and extracting
structured information from model-generated text.
"""

from sdg_hub.blocks import BlockRegistry, Block
import re
from typing import List
from datasets import Dataset


@BlockRegistry.register("PostProcessThinkingBlock")
class PostProcessThinkingBlock(Block):
    """
    Block for post-processing thinking steps in model outputs.

    This block removes thinking steps (content between <think> tags) from the specified
    column in the dataset, keeping only the final response. It's useful for cleaning up
    model outputs that include intermediate reasoning steps.

    Args:
        block_name (str): Name of the block
        column_name (str): Name of the column containing the text to process
    """
    def __init__(self, block_name: str, column_name: str) -> None:
        super().__init__(block_name=block_name)  
        self.column_name = column_name
    
    def generate(self, samples: Dataset):
        """
        Process the samples to remove thinking steps.

        Args:
            samples (Dataset): Input dataset containing the text to process

        Returns:
            Dataset: Processed dataset with thinking steps removed from the specified column
        """
        def post_process_thinking(x):
            if '</think>' in x[self.column_name]:
                x[self.column_name] = x[self.column_name].split('</think>')[-1].lstrip()
            return x
        samples = samples.map(post_process_thinking)
        return samples

@BlockRegistry.register("RegexParserBlock")
class RegexParserBlock(Block):
    """
    Block for parsing structured content using regular expressions.

    This block extracts structured information from text using regex patterns and
    can optionally clean up specific tags from the extracted content. It's useful for
    converting unstructured model outputs into structured data.

    Args:
        block_name (str): Name of the block
        column_name (str): Name of the column containing the text to parse
        parsing_pattern (str, optional): Regex pattern for parsing. Defaults to ""
        parser_cleanup_tags (List[str], optional): Tags to clean up from parsed content. Defaults to []
        output_cols (List[str], optional): Names of output columns for parsed content. Defaults to []
    """
    def __init__(self, block_name: str, column_name: str, parsing_pattern: str="", 
                 parser_cleanup_tags: List[str]=[], output_cols: List[str]=[]) -> None:
        super().__init__(block_name=block_name)
        self.column_name = column_name
        self.parsing_pattern = parsing_pattern
        self.parser_cleanup_tags = parser_cleanup_tags
        self.output_cols = output_cols

    def generate(self, samples: Dataset):
        """
        Process the samples using regex parsing.

        Args:
            samples (Dataset): Input dataset containing the text to parse

        Returns:
            Dataset: Processed dataset with parsed content added as new columns
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
               samples = samples.map(lambda x: {column_name: x[column_name].replace(clean_tag, "") for column_name in self.output_cols})
        return samples

    def _parse(self, generated_string):
        """
        Parse the input string using the configured regex pattern.

        Args:
            generated_string (str): Input string to parse

        Returns:
            dict: Dictionary mapping output column names to lists of parsed values
        """
        pattern = re.compile(self.parsing_pattern, re.DOTALL)
        all_matches = pattern.findall(generated_string)
        matches = {column_name: [] for column_name in self.output_cols}
        if all_matches and isinstance(all_matches[0], tuple):
            for match in all_matches:
                for column_name, value in zip(self.output_cols, match):
                    value = value.strip()
                    matches[column_name].append(value)
        else:
            matches[self.output_cols[0]] = (
                [match.strip() for match in all_matches] if all_matches else []
            )
        return matches
```

```python
from sdg_hub.flow import Flow
from sdg_hub.pipeline import Pipeline
from sdg_hub.sdg import SDG
from transformers import AutoTokenizer
import utils
from datasets import load_dataset
import prompts.prompts
import blocks.blocks

ds = load_dataset('json', data_files="3M_SEC_filings_seed_data.jsonl", split="train")
knowledge_agentic_pipeline = "flows/synth_knowledge_reasoning_nemotron_super_49b_summary_diversity.yaml"
flow_cfg = Flow(client).get_flow_from_file(knowledge_agentic_pipeline)
sdg = SDG(
    [Pipeline(flow_cfg)],
    num_workers=1,
    batch_size=1,
    save_freq=1000,
)
```


```python
from datasets import Dataset
ds_new = Dataset.from_list([{
    "document": """## PERFORMANCE BY BUSINESS SEG MENT

For a detailed discussion of the markets served and types of products offered by 3M's business segments, see Item 1, Business Segments. Financial information and other disclosures are provided in the Notes to the Consolidated Financial Statements. Effective in the first quarter of 2018, as part of 3M's continuing effort to improve the alignment of its businesses around markets and customers, the Company made the following changes:

## Consolidation of customer account activity within international countries - expanding dual credit reporting

The Company consolidated its customer account activity in each country into centralized sales districts for certain countries that make up approximately 70 percent of 3M's 2017 international net sales. Expansion of these initiatives, which previously had been deployed only in the U.S., reduces the complexity for customers when interacting with multiple 3M businesses. 3M business segment reporting measures include dual credit to business segments for certain sales and related operating income. This dual credit is based on which business segment provides customer account activity with respect to a particular product sold in a specific country. The expansion of alignment of customer accounts within additional countries increased the attribution of dual credit across 3M's business segments. Additionally, certain sales and operating income results for electronic bonding product lines that were previously equally divided between the Electronics and Energy business segment and the Industrial business segment are now reported similarly to dual credit.""",
    "document_outline": "3M SEC Filings Form 10-K 2018",
    "icl_document": ds[0]['icl_document'],
    "icl_query_1": ds[0]['icl_query_1'],
    "icl_response_1": ds[0]['icl_response_1'],
    "icl_query_2": ds[0]['icl_query_2'],
    "icl_response_2": ds[0]['icl_response_2'],
    "icl_query_3": ds[0]['icl_query_3'],
    "icl_response_3": ds[0]['icl_response_3']
}])
```


```python
generated_data = sdg.generate(ds_new, checkpoint_dir="Tmp")
```


<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7fbfbf; text-decoration-color: #7fbfbf">[15:12:57] </span><span style="color: #000080; text-decoration-color: #000080">INFO    </span> No existing checkpoints found in Tmp, generating from scratch                        <a href="file:///workspace/home/lab/abhi/sdg_hub/src/sdg_hub/sdg.py" target="_blank"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">sdg.py</span></a><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">:</span><a href="file:///workspace/home/lab/abhi/sdg_hub/src/sdg_hub/sdg.py#107" target="_blank"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">107</span></a>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7fbfbf; text-decoration-color: #7fbfbf">           </span><span style="color: #000080; text-decoration-color: #000080">INFO    </span> Splitting the dataset into smaller batches                                           <a href="file:///workspace/home/lab/abhi/sdg_hub/src/sdg_hub/sdg.py" target="_blank"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">sdg.py</span></a><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">:</span><a href="file:///workspace/home/lab/abhi/sdg_hub/src/sdg_hub/sdg.py#123" target="_blank"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">123</span></a>
</pre>



    100%|██████████| 1/1 [00:00<00:00, 27413.75it/s]



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7fbfbf; text-decoration-color: #7fbfbf">           </span><span style="color: #000080; text-decoration-color: #000080">INFO    </span> Generating dataset with <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1</span> splits, batch size <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1</span>, and <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1</span> workers                        <a href="file:///workspace/home/lab/abhi/sdg_hub/src/sdg_hub/sdg.py" target="_blank"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">sdg.py</span></a><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">:</span><a href="file:///workspace/home/lab/abhi/sdg_hub/src/sdg_hub/sdg.py#129" target="_blank"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">129</span></a>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7fbfbf; text-decoration-color: #7fbfbf">           </span><span style="color: #000080; text-decoration-color: #000080">INFO    </span> Processing split <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span>                                                                    <a href="file:///workspace/home/lab/abhi/sdg_hub/src/sdg_hub/sdg.py" target="_blank"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">sdg.py</span></a><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">:</span><a href="file:///workspace/home/lab/abhi/sdg_hub/src/sdg_hub/sdg.py#75" target="_blank"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">75</span></a>
</pre>



      0%|          | 0/1 [00:00<?, ?it/s]


<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7fbfbf; text-decoration-color: #7fbfbf">[15:13:16] </span><span style="color: #000080; text-decoration-color: #000080">INFO    </span> HTTP Request: <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">POST</span> <span style="color: #0000ff; text-decoration-color: #0000ff; text-decoration: underline">http://localhost:8000/v1/completions</span> <span style="color: #008000; text-decoration-color: #008000">"HTTP/1.1 200 OK"</span>       <a href="file:///home/lab/.conda/envs/research_sdg/lib/python3.10/site-packages/httpx/_client.py" target="_blank"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">_client.py</span></a><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">:</span><a href="file:///home/lab/.conda/envs/research_sdg/lib/python3.10/site-packages/httpx/_client.py#1025" target="_blank"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">1025</span></a>
</pre>



    Map: 100%|██████████| 2/2 [00:00<00:00, 580.04 examples/s]
    Map: 100%|██████████| 20/20 [00:00<00:00, 6003.44 examples/s]



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7fbfbf; text-decoration-color: #7fbfbf">[15:13:45] </span><span style="color: #000080; text-decoration-color: #000080">INFO    </span> HTTP Request: <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">POST</span> <span style="color: #0000ff; text-decoration-color: #0000ff; text-decoration: underline">http://localhost:8000/v1/completions</span> <span style="color: #008000; text-decoration-color: #008000">"HTTP/1.1 200 OK"</span>       <a href="file:///home/lab/.conda/envs/research_sdg/lib/python3.10/site-packages/httpx/_client.py" target="_blank"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">_client.py</span></a><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">:</span><a href="file:///home/lab/.conda/envs/research_sdg/lib/python3.10/site-packages/httpx/_client.py#1025" target="_blank"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">1025</span></a>
</pre>



    Map: 100%|██████████| 20/20 [00:00<00:00, 4306.27 examples/s]



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7fbfbf; text-decoration-color: #7fbfbf">[15:14:11] </span><span style="color: #000080; text-decoration-color: #000080">INFO    </span> HTTP Request: <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">POST</span> <span style="color: #0000ff; text-decoration-color: #0000ff; text-decoration: underline">http://localhost:8000/v1/completions</span> <span style="color: #008000; text-decoration-color: #008000">"HTTP/1.1 200 OK"</span>       <a href="file:///home/lab/.conda/envs/research_sdg/lib/python3.10/site-packages/httpx/_client.py" target="_blank"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">_client.py</span></a><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">:</span><a href="file:///home/lab/.conda/envs/research_sdg/lib/python3.10/site-packages/httpx/_client.py#1025" target="_blank"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">1025</span></a>
</pre>



    Map: 100%|██████████| 20/20 [00:00<00:00, 3926.52 examples/s]
    Map: 100%|██████████| 80/80 [00:00<00:00, 10023.13 examples/s]



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7fbfbf; text-decoration-color: #7fbfbf">[15:14:33] </span><span style="color: #000080; text-decoration-color: #000080">INFO    </span> HTTP Request: <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">POST</span> <span style="color: #0000ff; text-decoration-color: #0000ff; text-decoration: underline">http://localhost:8000/v1/completions</span> <span style="color: #008000; text-decoration-color: #008000">"HTTP/1.1 200 OK"</span>       <a href="file:///home/lab/.conda/envs/research_sdg/lib/python3.10/site-packages/httpx/_client.py" target="_blank"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">_client.py</span></a><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">:</span><a href="file:///home/lab/.conda/envs/research_sdg/lib/python3.10/site-packages/httpx/_client.py#1025" target="_blank"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">1025</span></a>
</pre>



    Map: 100%|██████████| 80/80 [00:00<00:00, 8456.68 examples/s]
    Map: 100%|██████████| 80/80 [00:00<00:00, 9271.74 examples/s]
    Map: 100%|██████████| 80/80 [00:00<00:00, 9205.10 examples/s]



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7fbfbf; text-decoration-color: #7fbfbf">[15:15:05] </span><span style="color: #000080; text-decoration-color: #000080">INFO    </span> HTTP Request: <span style="color: #808000; text-decoration-color: #808000; font-weight: bold">POST</span> <span style="color: #0000ff; text-decoration-color: #0000ff; text-decoration: underline">http://localhost:8000/v1/completions</span> <span style="color: #008000; text-decoration-color: #008000">"HTTP/1.1 200 OK"</span>       <a href="file:///home/lab/.conda/envs/research_sdg/lib/python3.10/site-packages/httpx/_client.py" target="_blank"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">_client.py</span></a><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">:</span><a href="file:///home/lab/.conda/envs/research_sdg/lib/python3.10/site-packages/httpx/_client.py#1025" target="_blank"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">1025</span></a>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #7fbfbf; text-decoration-color: #7fbfbf">           </span><span style="color: #000080; text-decoration-color: #000080">INFO    </span> Finished future processing split <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span>                                                   <a href="file:///workspace/home/lab/abhi/sdg_hub/src/sdg_hub/sdg.py" target="_blank"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">sdg.py</span></a><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">:</span><a href="file:///workspace/home/lab/abhi/sdg_hub/src/sdg_hub/sdg.py#149" target="_blank"><span style="color: #7f7f7f; text-decoration-color: #7f7f7f">149</span></a>
<span style="color: #7fbfbf; text-decoration-color: #7fbfbf">           </span>                                                                                              <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">          </span>
<span style="color: #7fbfbf; text-decoration-color: #7fbfbf">           </span>                                                                                              <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">          </span>
</pre>



    100%|██████████| 1/1 [02:07<00:00, 127.53s/it]



```python
idx = 3
generated_data_ = generated_data.filter(lambda x: len(x['document']) < 1800)
print("### Document")
print(generated_data_['raw_document'][idx])
print("### Generated Summary Instruction")
print(generated_data_['summary_instruction'][idx])
print("### Generated Document Summary")
print(generated_data_['document'][idx])
```

    Filter:   0%|          | 0/80 [00:00<?, ? examples/s]

    Filter: 100%|██████████| 80/80 [00:00<00:00, 21751.87 examples/s]

    ### Document
    ## PERFORMANCE BY BUSINESS SEG MENT
    
    For a detailed discussion of the markets served and types of products offered by 3M's business segments, see Item 1, Business Segments. Financial information and other disclosures are provided in the Notes to the Consolidated Financial Statements. Effective in the first quarter of 2018, as part of 3M's continuing effort to improve the alignment of its businesses around markets and customers, the Company made the following changes:
    
    ## Consolidation of customer account activity within international countries - expanding dual credit reporting
    
    The Company consolidated its customer account activity in each country into centralized sales districts for certain countries that make up approximately 70 percent of 3M's 2017 international net sales. Expansion of these initiatives, which previously had been deployed only in the U.S., reduces the complexity for customers when interacting with multiple 3M businesses. 3M business segment reporting measures include dual credit to business segments for certain sales and related operating income. This dual credit is based on which business segment provides customer account activity with respect to a particular product sold in a specific country. The expansion of alignment of customer accounts within additional countries increased the attribution of dual credit across 3M's business segments. Additionally, certain sales and operating income results for electronic bonding product lines that were previously equally divided between the Electronics and Energy business segment and the Industrial business segment are now reported similarly to dual credit.
    ### Generated Summary Instruction
    **Employee Update**: Describe organizational changes in customer account management for internal stakeholders.
    ### Generated Document Summary
    **Employee Update: Organizational Changes in Customer Account Management**  
    
    As part of 3M’s efforts to align businesses around markets and customers, effective Q1 2018, the company implemented two key changes in customer account management for international operations:  
    
    1. **Consolidation of Customer Account Activity**:  
       - **Scope**: Centralized sales districts were established in countries accounting for **~70% of 3M’s 2017 international net sales**.  
       - **Purpose**: Reduce complexity for customers interacting with multiple 3M businesses by streamlining account management within each country (previously only deployed in the U.S.).  
    
    2. **Expansion of Dual Credit Reporting**:  
       - **Mechanism**: Dual credit is assigned to business segments for sales and operating income based on which segment manages the **customer account activity** for a product in a specific country.  
       - **Impact**: The international consolidation increased dual credit attribution across segments.  
       - **Electronic Bonding Product Lines**: Previously split equally between the **Electronics and Energy** and **Industrial** segments, these are now reported using the dual credit methodology (aligned with country-based account responsibility).  
    
    **Key Data Points**:  
    - **Effective Date**: Q1 2018.  
    - **International Sales Coverage**: ~70% of 2017 international net sales impacted.  
    - **Product Adjustment**: Electronic bonding lines shifted from equal division to dual credit alignment.  
    
    These changes enhance market alignment, simplify customer interactions, and refine segment reporting accuracy.  
    
    ---  
    *Note: The document contains no tables, so all relevant textual data is summarized above.*


    


### Generated Summary Instruction
```text
**Highlight Tax Strategy Efficiency**: Analyze how 3M’s tax planning (e.g., Food Safety divestiture, GILTI/FDII) reduced its effective tax rate, with numerical examples.
```
### Generated Document Summary
```text
**Comprehensive Summary of 3M’s 2022 Form 10-K Filing with Emphasis on Tax Strategy Efficiency**

---

### **1. Effective Tax Rate Reconciliation & Tax Strategy Efficiency**

| **Factor**               | **2022** | **2021** | **2020** |  
|--------------------------|---------|---------|---------|  
| Statutory U.S. Tax Rate  | 21.0%   | 21.0%   | 21.0%   |  
| **Food Safety Divestiture** | **-8.4%** | -       | -       | *Key driver of 2022 tax rate reduction*  
| State Income Taxes       | -       | 0.9%   | 1.2%   |  
| International Income Taxes| -0.4%  | -1.2%  | -1.2%  |  
| **GILTI**                | 0.7%   | 0.7%   | 0.8%   |  
| **FDII**                 | -2.3%  | -3.1%  | -1.8%  |  
| U.S. R&D Credit          | -1.0%  | -0.7%  | -1.0%  |  
| Reserves for Tax Contingencies| -    | 0.6%   | 0.5%   |  
| Employee Share-Based Payments| -0.2% | -0.6%  | -0.5%  |  
| All Other                 | 0.2%   | 0.2%   | 0.7%   |  
| **Effective Tax Rate**   | **9.6%** | **17.8%**| **19.7%**|  

- **Tax Rate Reduction**:  
  - **2022 vs. 2021**: 8.2% decrease primarily due to **Food Safety divestiture** (saved 8.4% of tax rate).  
  - **2021 vs. 2020**: 1.9% decrease attributed to geographical income mix and favorable TCJA adjustments.  

- **GILTI/FDII Impact**:  
  - GILTI increased tax rate by 0.7% in 2022 (vs. 0.8% in 2020).  
  - FDII reduced tax rate by 2.3% in 2022 (vs. 3.1% in 2021), reflecting efficient intangible income planning.  

---

### **2. TCJA Provisions & Transition Tax**

- **Transition Tax (TCJA)**: $380M (2022) and $508M (2021) in long-term income taxes payable over 8 years.  
- **Short-Term Transition Tax**: $126M (2022) and $68M (2021).  

---

### **3. Uncertain Tax Positions (UTB)**

| **(Millions)**         | **2022** | **2021** | **2020** |  
|-------------------------|---------|---------|---------|  
| Gross UTB Balance (Jan 1)| 1,071   | 1,113   | 1,167   |  
| Additions (Current Year)| 115    | 91      | 74      |  
| Additions (Prior Years) | 36    | 22      | 106     |  
| Reductions (Prior Years)| -138   | -60     | -173    |  
| Settlements              | -118   | -57     | -8      |  
| Statute of Limitations   | -39    | -38     | -53     |  
| **Gross UTB (Dec 31)**  | **927** | **1,071**| **1,113**|  
| **Net UTB (Impact on ETR)**| **965** | **1,112**| **1,145**|  

- **Net UTB**: $965M (2022) could affect ETR if recognized.  
- **Interest/Penalties**: $116M accrued in 2022 (vs. $140M in 2021).  

---

### **4. Foreign Tax Benefits & Undistributed Earnings**

- **Tax-Exempt/Subsidized Income**:  
  - China (2022), Switzerland (2026), Brazil (2029), Singapore (2032) generated $170M (30¢/share) in 2022 tax benefits.  
- **Undistributed Foreign Earnings**: $16.0B (2022), with $8.0B no longer permanently reinvested (immaterial repatriation tax cost).  

---

### **5. Marketable Securities (2022/2021)**

| **(Millions)**         | **2022** | **2021** |  
|-------------------------|---------|---------|  
| Commercial Paper         | 213    | 109     |  
| Certificates of Deposit  | 21     | 14     |  
| U.S. Treasury Securities | -      | 75     |  
| **Current Securities**   | **238** | **201** |  
| Non-Current Securities   | 23     | 27     |  
| **Total**                | **261** | **228** |  

---

### **6. Long-Term Debt (Select Entries, 2022/2021)**

| **Description**                  | **2022 Carrying Value** | **2021 Carrying Value** | **Effective Interest Rate** |  
|-----------------------------------|------------------------|------------------------|---------------------------|  
| USD Fixed (2023, $500M)          | $500                  | $499                  | 1.86%                    |  
| EUR Fixed (2023, €600M)          | $679                  | $679                  | 1.14%                    |  
| USD Floating (2024, $300M)      | $300                  | $300                  | 4.81%                    |  
| **Total Long-Term Debt**         | **$15,939**            | **$17,347**            |                           |  

---

### **Key Tax Strategy Takeaways**

1. **Food Safety Divestiture**: Saved 8.4% of tax rate in 2022 via tax-efficient structuring.  
2. **GILTI/FDII Optimization**: FDII provided greater tax savings than GILTI costs (e.g., 2022: -2.3% vs. +0.7%).  
3. **Foreign Subsidiary Planning**: Leveraged low-tax jurisdictions (e.g., China, Switzerland) for $170M in annual benefits.  
4. **Repatriation Strategy**: $8.0B in non-permanently reinvested earnings with minimal tax liability.  

---

### **Full Tables for Reference**

#### **Reconciliation of Effective Tax Rate (Detailed)**

| Factor                                  | 2022 | 2021 | 2020 |  
|-----------------------------------------|------|------|------|  
| Statutory U.S. Tax Rate                 |21.0%|21.0%|21.0%|  
| **Food Safety Divestiture**             |-8.4%| -   | -   |  
| State Income Taxes                      |-    |0.9%|1.2% |  
| International Income Taxes              |-0.4%|-1.2%|-1.2%|  
| **GILTI**                              |0.7%|0.7%|0.8% |  
| **FDII**                               |-2.3%|-3.1%|-1.8%|  
| U.S. R&D Credit                         |-1.0%|-0.7%|-1.0%|  
| Reserves for Tax Contingencies          |-    |0.6%|0.5% |  
| Employee Share-Based Payments           |-0.2%|-0.6%|-0.5%|  
| All Other                               |0.2%|0.2%|0.7% |  
| **Effective Tax Rate**                  |9.6%|17.8%|19.7%|  

#### **Gross Unrecognized Tax Benefits**

| **(Millions)**         | **2022** | **2021** | **2020** |  
|-------------------------|---------|---------|---------|  
| Gross UTB Balance (Jan 1)| 1,071   | 1,113   | 1,167   |  
| Additions (Current Year)| 115    | 91      | 74      |  
| Additions (Prior Years) | 36    | 22      | 106     |  
| Reductions (Prior Years)| -138   | -60     | -173    |  
| Settlements              | -118   | -57     | -8      |  
| Statute of Limitations   | -39    | -38     | -53     |  
| **Gross UTB (Dec 31)**  | **927** | **1,071**| **1,113**|  

#### **Marketable Securities Maturity**

| **(Millions)**         | **2022** |  
|-------------------------|---------|  
| Due in one year or less | 238    |  
| Due after one year      | 15     |  
| Due after five years    | 8      |  
| **Total**               | **261** |  

#### **Long-Term Debt (Full Table Omitted for Brevity)**

*Refer to original filing for detailed debt schedule.*  

---

### **Numerical Examples of Tax Efficiency**

1. **Food Safety Divestiture Impact**: Reduced 2022 ETR by 8.4% (from 21.0% to 9.6%).  
2. **FDII Savings**: Saved 2.3% of tax rate in 2022 (vs. 3.1% in 2021).  
3. **GILTI Cost**: Added 0.7% to tax rate in 2022 (consistent across years).  
4. **Foreign Tax Credits**: $170M benefit in 2022 (30¢/share).  

--- 

**Summary Conclusion**:  
3M’s tax strategy leveraged divestitures, TCJA provisions (GILTI/FDII), and foreign jurisdiction planning to achieve a record-low 9.6% ETR in 2022. The Food Safety divestiture alone saved 8.4%, while FDII outperformed GILTI in tax savings. Foreign subsidiaries contributed $170M in annual benefits, and undistributed earnings pose minimal repatriation risk.  

---

**All Tables Included as Requested**  

Let me know if further clarification is needed!
```


```python
idx = 2
print("### Question")
print(generated_data_[idx]['question'])
print("### Generated Answer")
print(generated_data_[idx]['response'])
```

    ### Question
    Explain the dual credit reporting mechanism introduced by 3M in 2018 and how it affected the Electronic Bonding Product Lines.
    
    ### Generated Answer
    <think>
    Okay, let's tackle this question. The user is asking about the dual credit reporting mechanism introduced by 3M in 2018 and its impact on the Electronic Bonding Product Lines.
    
    First, I need to recall the document provided. The key points from the document mention that 3M made organizational changes in customer account management. Specifically, they expanded the dual credit reporting mechanism. 
    
    The dual credit mechanism assigns credit (for sales and operating income) to business segments based on which segment manages the customer account activity for a product in a specific country. Before this change, Electronic Bonding Product Lines were split equally between the Electronics and Energy and Industrial segments. Now, with the dual credit method, the reporting is aligned with the country-based account responsibility. 
    
    So, the mechanism itself is about attributing credits based on account management responsibility. The effect on Electronic Bonding lines is that they moved from a 50-50 split to being reported under whichever segment handles the account in that country. This would mean that instead of fixed allocation, it's now dynamic based on who manages the account. 
    
    I should make sure to mention both parts: the mechanism's definition and the specific change for Electronic Bonding. Also, check if there's any mention of the impact beyond the reporting method, like benefits or challenges, but the document only states the alignment with account responsibility. 
    
    Double-check the document to ensure accuracy. The answer should directly reference the dual credit explanation and the product line adjustment. Avoid adding anything not in the document. The answer needs to be concise, in the specified format, and only based on the provided text.
    </think>
    
    
    The dual credit reporting mechanism assigns sales and operating income credits to business segments based on which segment manages the customer account activity for a product in a specific country. For the Electronic Bonding Product Lines, this change shifted their reporting from an equal split between the **Electronics and Energy** and **Industrial** segments to alignment with country-based account responsibility using the dual credit methodology.
    


### Question
```text
What was the primary factor contributing to the 8.2% decrease in 3M’s effective tax rate from 2021 to 2022?
```
### Generated Answer
```text
<think>
Okay, let's tackle this question. The user is asking for the primary factor that caused the 8.2% decrease in 3M's effective tax rate from 2021 to 2022.

First, I need to look through the provided document to find the relevant information. The document is a comprehensive summary of 3M's 2022 Form 10-K filing, focusing on tax strategy efficiency. 

Scanning through the sections, the first part under "1. Effective Tax Rate Reconciliation & Tax Strategy Efficiency" has a table showing the effective tax rates for 2022, 2021, and 2020. The effective tax rate dropped from 17.8% in 2021 to 9.6% in 2022, which is an 8.2% decrease. 

Below the table, there's a bullet point under "Tax Rate Reduction" that explains the 2022 vs. 2021 decrease. It states: "- **2022 vs. 2021**: 8.2% decrease primarily due to **Food Safety divestiture** (saved 8.4% of tax rate)." 

Additionally, in the "Key Tax Strategy Takeaways" section, point 1 mentions the Food Safety Divestiture saving 8.4% of the tax rate in 2022 via tax-efficient structuring. The numerical examples also highlight that the Food Safety Divestiture reduced the ETR by 8.4%, which aligns with the primary factor.

Other factors in the reconciliation table, like FDII and GILTI, have smaller impacts. For instance, FDII reduced the tax rate by 2.3% in 2022, but that's not the primary factor here. The main driver is clearly the Food Safety divestiture as stated in multiple parts of the document.

So the answer should directly reference the Food Safety divestiture as the primary factor causing the 8.2% decrease. Making sure to mention the 8.4% savings from the divestiture and that it's the primary reason as per the document.
</think>


The primary factor contributing to the 8.2% decrease in 3M’s effective tax rate from 2021 to 2022 was the **Food Safety divestiture**, which saved 8.4% of the tax rate through tax-efficient structuring. This divestiture was explicitly cited as the key driver of the reduced effective tax rate from 17.8% in 2021 to 9.6% in 2022.
```

### Analyze Generated data


```python
# Data generated from previous notebook
from transformers import AutoTokenizer
from datasets import load_dataset

# Load tokenizer for pre-training formatting
tokenizer = AutoTokenizer.from_pretrained("nvidia/Llama-3.1-Nemotron-Nano-8B-v1")
threeM_reasoning_ds = load_dataset("json", data_dir=f"{data_base_path}/3m_sec_filings/synth_knowledge_reasoning_nemotron_super_49b_summary_diversity/", split="train")
threeM_reasoning_ds = threeM_reasoning_ds.map(lambda x: {'token_size': len(tokenizer.encode(x['document']))}, batched=False, num_proc=16)
# Plot the distribution of SEC filings data for 3M
import matplotlib.pyplot as plt
plt.hist(threeM_reasoning_ds['token_size'], bins=100)
plt.show()
```

    Map (num_proc=16): 100%|██████████| 8573/8573 [00:01<00:00, 7131.43 examples/s] 



    
![png](output_23_1.png)
    



```python
# With changes to prompt
from transformers import AutoTokenizer
from datasets import load_dataset
data_base_path = os.environ["DATA_BASE_PATH"]
# Load tokenizer for pre-training formatting
tokenizer = AutoTokenizer.from_pretrained("nvidia/Llama-3.1-Nemotron-Nano-8B-v1")
threeM_reasoning_ds = load_dataset("json", data_dir=f"{data_base_path}/3m_sec_filings/synth_knowledge_reasoning_nemotron_super_49b_summary_diversity_detailed/", split="train")
threeM_reasoning_ds = threeM_reasoning_ds.map(lambda x: {'token_size': len(tokenizer.encode(x['document']))}, batched=False, num_proc=16)
# Plot the distribution of SEC filings data for 3M
import matplotlib.pyplot as plt
plt.hist(threeM_reasoning_ds['token_size'], bins=100)
plt.show()
```

    Generating train split: 8110 examples [00:01, 6133.65 examples/s]
    Map (num_proc=16): 100%|██████████| 8110/8110 [00:01<00:00, 4576.26 examples/s]



    
![png](output_24_1.png)
    


## Create training mix


```python
from utils import create_training_mix
from transformers import AutoTokenizer
from datasets import load_dataset, concatenate_datasets
 
def extract_think(x):
    if 'messages' in x:
        if '<think>' in x['messages'][2]['content'] and '</think>' in x['messages'][2]['content']:
            return {'think' : x['messages'][2]['content'][x['messages'][2]['content'].find('<think>')+len('<think>'):x['messages'][2]['content'].find('</think>')]}
    elif '<think>' in x['response'] and '</think>' in x['response']:
        return {'think' : x['response'][x['response'].find('<think>')+len('<think>'):x['response'].find('</think>')]}
    return {'think' : ''}


data_base_path = os.environ["DATA_BASE_PATH"]
# Load tokenizer for pre-training formatting
tokenizer = AutoTokenizer.from_pretrained("nvidia/Llama-3.1-Nemotron-Nano-8B-v1")


# Load reasoning dataset from nemotron super 49b
threeM_reasoning_ds = load_dataset("json", data_dir=f"{data_base_path}/3m_sec_filings/synth_knowledge_reasoning_nemotron_super_49b_rewrite_with_diversity/", split="train")
threeM_reasoning_ds = threeM_reasoning_ds.filter(lambda x: '[QUESTION]' not in x['response'])
# Filter out responses that don't have a single think block
threeM_reasoning_ds = threeM_reasoning_ds.filter(lambda x: x['response'].count('<think>') == 1 and x['response'].count('</think>') == 1)
threeM_reasoning_ds = threeM_reasoning_ds.filter(lambda x: x['question'].count('<think>') == 0 and x['question'].count('</think>') == 0)
# Filter out summaries that are too short
threeM_reasoning_ds = threeM_reasoning_ds.map(lambda x: {'token_size': len(tokenizer.encode(x['document']))}, batched=False, num_proc=16)
threeM_reasoning_ds = threeM_reasoning_ds.filter(lambda x: x['token_size'] > 300).remove_columns(['token_size'])
print(threeM_reasoning_ds)
# Create reasoning training mix
threeM_reasoning_ds = threeM_reasoning_ds.add_column('domain', ['SEC Filings'] * len(threeM_reasoning_ds))
threeM_reasoning_ds = create_training_mix(threeM_reasoning_ds, tokenizer, 'on', create_summary=False)
threeM_reasoning_ds = threeM_reasoning_ds.filter(lambda x: '<think>' in x['messages'][2]['content'] and '</think>' in x['messages'][2]['content'])
threeM_reasoning_ds = threeM_reasoning_ds.map(extract_think, batched=False, num_proc=16)
threeM_reasoning_ds = threeM_reasoning_ds.filter(lambda x: x['think'].strip()).remove_columns(['think'])


# Load non-reasoning dataset from nemotron super 49b
threeM_non_reasoning_ds = load_dataset("json", data_dir=f"{data_base_path}/3m_sec_filings/synth_knowledge_reasoning_nemotron_super_49b/", split="train")
threeM_non_reasoning_ds = threeM_non_reasoning_ds.add_column('domain', ['SEC Filings'] * len(threeM_non_reasoning_ds))
threeM_non_reasoning_ds = threeM_non_reasoning_ds.filter(lambda x: x['score'] == '2' and x['judgment'] == 'YES')
print(threeM_non_reasoning_ds)

# Create non-reasoning training mix
threeM_non_reasoning_ds = create_training_mix(threeM_non_reasoning_ds, tokenizer, 'off', create_summary=False).shuffle(seed=894375)



# Load nemotron replay buffer
nemotron_ds_replay_buffer = load_dataset("json", data_files=f"{data_base_path}/quality/training_mix/nemotron_replay_buffer_data.jsonl", split="train")
nemotron_ds_replay_buffer = nemotron_ds_replay_buffer.filter(lambda x: '<think>' in x['messages'][2]['content'] and '</think>' in x['messages'][2]['content'])
nemotron_ds_replay_buffer = nemotron_ds_replay_buffer.map(extract_think, batched=False, num_proc=16)
nemotron_ds_replay_buffer = nemotron_ds_replay_buffer.filter(lambda x: x['think'].strip()).remove_columns(['think'])

# Concatenate training mix with replay buffer
training_mix = concatenate_datasets([threeM_reasoning_ds, threeM_non_reasoning_ds, nemotron_ds_replay_buffer.shuffle(seed=894375).select(range(len(threeM_reasoning_ds)))])

training_mix_only_reasoning = concatenate_datasets([threeM_reasoning_ds, nemotron_ds_replay_buffer.shuffle(seed=894375).select(range(len(threeM_reasoning_ds)))])

print(training_mix)
training_mix.to_json(f"{data_base_path}/3m_sec_filings/training_mix/3m_sec_filings_reasoning_mix_with_doc_rewrite_diversity_detailed_non_reasoning.jsonl", orient='records', lines=True)
```

### Wrapping Up
- We have generated reasoning data on small subset of Finance Bench document. This can be easily extended to run on all documents
- You can also checkout out two more pipelines
    - `synth_knowledge_reasoning_nemotron_super_49b_rewrite_with_diversity.yaml`: Re-write the document instead of summarizing it
    - `synth_knowledge_reasoning_nemotron_super_49b_summary_diversity_cot.yaml`: Modified version of the main reasoning pipeline where we use COT Question/Answer generation prompt

generate.py:
```python
# Third Party
from datasets import load_dataset
from openai import OpenAI
import click

# First Party
from sdg_hub.flow import  Flow
from sdg_hub.logger_config import setup_logger
from sdg_hub.pipeline import Pipeline
from sdg_hub.sdg import SDG
from sdg_hub.prompts import PromptRegistry
from sdg_hub.blocks import BlockRegistry, Block
from transformers import AutoTokenizer
import re
from typing import List
from datasets import Dataset

logger = setup_logger(__name__)

### Nemotron Chat Template with detailed thinking on
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
            if '</think>' in x[self.column_name]:
                x[self.column_name] = x[self.column_name].split('</think>')[-1].lstrip()
            return x
        samples = samples.map(post_process_thinking)
        return samples

@BlockRegistry.register("RegexParserBlock")
class RegexParserBlock(Block):
    def __init__(self, block_name: str, column_name: str, parsing_pattern: str="", parser_cleanup_tags: List[str]=[], output_cols: List[str]=[]) -> None:
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
                for values in zip(*(lst[:max_length] for lst in parsed_outputs.values())):
                    new_data.append({**sample, **dict(zip(parsed_outputs.keys(), values))})
            samples = Dataset.from_list(new_data)
        if self.parser_cleanup_tags:
            for clean_tag in self.parser_cleanup_tags:
               samples = samples.map(lambda x: {column_name: x[column_name].replace(clean_tag, "") for column_name in self.output_cols})
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

@click.command()
@click.option(
    "--ds_path",
    type=click.Path(exists=True),
    required=True,
    help="Path to the dataset.",
)
@click.option("--bs", type=int, default=8, show_default=True, help="Batch size.")
@click.option(
    "--num_workers", type=int, default=32, show_default=True, help="Number of workers."
)
@click.option(
    "--save_path", type=click.Path(), required=True, help="Path to save the output."
)
@click.option(
    "--endpoint", type=str, required=True, help="Endpoint for data processing."
)
@click.option(
    "--flow", type=str, required=True, help="Flow configuration for the process."
)
@click.option(
    "--checkpoint_dir",
    type=click.Path(),
    required=True,
    help="Path to save checkpoints.",
)
@click.option(
    "--save_freq",
    type=int,
    default=2,
    show_default=True,
    help="Frequency to save checkpoints.",
)
@click.option("--debug", is_flag=True, help="Enable debug mode.")
@click.option("--dataset_start_index", type=int, default=0, help="Start index of the dataset.")
@click.option("--dataset_end_index", type=int, default=None, help="End index of the dataset.")
def main(
    ds_path,
    bs,
    num_workers,
    save_path,
    endpoint,
    flow,
    checkpoint_dir,
    save_freq,
    debug,
    dataset_start_index,
    dataset_end_index,
):
    """
    Main function to process the dataset.

    Parameters:
    ds_path (str): Path to the dataset.
    bs (int): Batch size.
    num_workers (int): Number of workers.
    save_path (str): Path to save the output.
    endpoint (str): Endpoint for data processing.
    flow (str): Flow configuration for the process.
    checkpoint_dir (str): Path to save checkpoints.
    save_freq (int): Frequency to save checkpoints.
    debug (bool): Enable debug mode.
    """
    logger.info(f"Generation configuration: {locals()}\n\n")
    ds = load_dataset("json", data_files=ds_path, split="train")
    if dataset_start_index is not None and dataset_end_index is not None:
        if dataset_end_index > len(ds):
            dataset_end_index = len(ds)
        ds = ds.select(range(dataset_start_index, dataset_end_index))
        logger.info(f"Dataset sliced from {dataset_start_index} to {dataset_end_index}")

    if debug:
        # For debugging, use a smaller subset of the dataset
        ds = ds.shuffle(seed=42).select(range(30))

    openai_api_key = "EMPTY"
    openai_api_base = endpoint

    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )

    flow_cfg = Flow(client).get_flow_from_file(flow)
    sdg = SDG(
        [Pipeline(flow_cfg)],
        num_workers=num_workers,
        batch_size=bs,
        save_freq=save_freq,
    )
    generated_data = sdg.generate(ds, checkpoint_dir=checkpoint_dir)
    
    save_path = save_path.replace(".jsonl", f"_{dataset_start_index}_{dataset_end_index}.jsonl")
    generated_data.to_json(save_path, orient="records", lines=True)
    logger.info(f"Data saved to {save_path}")


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    main()
```