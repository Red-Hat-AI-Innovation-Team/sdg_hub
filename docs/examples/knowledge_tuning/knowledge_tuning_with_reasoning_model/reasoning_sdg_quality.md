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

# Knowledge generation using a resoning model

- In this notebook, we will use the Nemotron Super 49B model to generate knowledge from a corpus of articles.
- Specifically, we will use the Quality Corpus for this purpose.
- The Nemotron Super 49B model will also be used not only for synthesizing training data but also for generation ICL examples for SDG.
- Since Nemotron Super is a thinking model, we will leverage it to generate synthetic "thinking data" for customizing a smaller thinking model - in this case, the Nemotron Nano model.

## Install sdg-hub

```bash 
pip install sdg-hub==0.1.0a2
```


## Installing Vllm

- Clone vllm repo
- Checkout PR #15008
- Build from source
    ```bash
    VLLM_USE_PRECOMPILED=1 pip install --editable .
    ```


```python
from datasets import load_dataset, concatenate_datasets
from openai import OpenAI

endpoint = f"http://localhost:8008/v1"
openai_api_key = "EMPTY"
openai_api_base = endpoint

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)
teacher_model = client.models.list().data[0].id
```

    /home/lab/.conda/envs/research_sdg/lib/python3.10/site-packages/tqdm/auto.py:21: TqdmWarning: IProgress not found. Please update jupyter and ipywidgets. See https://ipywidgets.readthedocs.io/en/stable/user_install.html
      from .autonotebook import tqdm as notebook_tqdm


### Construct ICL examples
- We will use nemotron super 49B model to generated some ICL examples
- We will use the generated ICLs to generate more examples from the quality corpus
- You can customize these prompts according to your downstream task
- Here we aim to create QA aimed at understanding not only specific details about articles but also to create a comprehensive understanding of the article


```python
def get_response_from_nemotron(msg):
    response = client.chat.completions.create(
        model=teacher_model,
        messages=msg,
        temperature=0.6,
        top_p=0.95,
        max_tokens=4096,
    )
    return response.choices[0].message.content
```


```python
ds = load_dataset("zitongyang/entigraph-quality-corpus", split='train').remove_columns(['entity', 'entigraph']).rename_columns({'raw': 'document', 'uid': 'document_outline'})
```


```python
prompt = f"Give me a short 3-4 line document and 3 detailed GRE-style thematic questions and answers based and grounded on the document."
prompt = f"Give me a short 8-9 line passage that has 2 paragraphs. Then, provide 3 detailed open-ended questions and their answers. Each question should require synthesizing or connecting ideas, entities, or concepts that appear in different parts of the passage"
prompt = f"Give me a short 8-9 line article. Then provide 3 Question and Answer based on the article such that the answers are long, comprehensive, detailed, and are grounded in the article."
msg = [{"role": "system", "content": f"detailed thinking on"}, {'role': 'user', 'content': prompt}]
generated_qa_pairs = get_response_from_nemotron(msg)
msg += [{"role": "assistant", "content": generated_qa_pairs}] + [{"role": "user", "content": "You refine the answers from previous response such that you distill your thinking into the answer"}]
edited_qa_pairs = get_response_from_nemotron(msg)
msg = msg + [{"role": "assistant", "content": edited_qa_pairs}] + [{"role": "user", "content": "Can you parse the generated question and answer pairs into a json format? If there are any steps include them in the json's answer's field"}]
json_verified_qa_pairs = get_response_from_nemotron(msg)
msg = msg + [{"role": "assistant", "content": json_verified_qa_pairs}] + [{"role": "user", "content": "Verify the generated question and answer pairs from previous response are correct, complete and grounded in the document. Fix all the issues in the answer if any."}]
verified_qa_pairs = get_response_from_nemotron(msg)
print(generated_qa_pairs)
print("-"*3)
print(edited_qa_pairs)
print("-"*3)
print(json_verified_qa_pairs)
print("-"*3)
print(verified_qa_pairs)
```

## Example of ICL samples generated chaining above prompts from Nemotron Super

### **Response 1** - Generate Initial Draft of Question and Answer Pairs
```text
<think>
Okay, the user wants a short 3-4 line document and three detailed GRE-style thematic questions with answers based on it. Let me start by crafting the document. It needs to be concise but rich enough for questions. Maybe a philosophical or historical topic since those lend themselves to thematic analysis.

I'll go with a quote from Nietzsche about morality. That's dense and can lead to questions on interpretation, implications, and comparisons. Now, the document should be 3-4 lines. Let me check the length. The Nietzsche passage I have in mind is four sentences. Perfect.

Next, the questions. GRE thematic questions often ask for interpretation, analysis of assumptions, and application to another context. First question: Identify the main thesis and explain. Second: Analyze the assumption underlying the claim. Third: Apply the idea to a different field, like science.

For the answers, they need to be detailed but not too long. Make sure each answer addresses all parts of the question. Check that the questions are indeed GRE-style—focusing on themes, structure, and application. Ensure the answers ground their points in the document. Avoid jargon, keep it clear. Let me review each part to ensure alignment with the user's request. Document length, question types, answer structure all seem to fit. Okay, that should work.
</think>

**Document (4 lines)**  
Nietzsche argued that traditional morality is a "slave morality" created by the weak to subjugate the strong, claiming, "The noble type of man experiences himself as the goal and measure of things; he does not require approval or justification from outsiders." In contrast, slave morality, rooted in resentment, posits equality and humility as virtues to constrain natural superiority. Nietzsche saw this as a perversion of life’s true will to power. Only the Übermensch could transcend such moral dichotomies.

---

**Thematic Questions & Answers**

### **1. Interpretation of Thesis**  
**Question:** Identify the central thesis of the passage and explain how the contrast between "noble" and "slave" morality supports Nietzsche’s broader philosophical claim.  
**Answer:**  
The thesis is that traditional morality is a subversive construct of the weak ("slave morality") to undermine natural hierarchy, contrasting with the self-affirming "noble morality" of the strong. The passage supports this by framing noble morality as intrinsic (e.g., "the goal and measure of things") and slave morality as reactive ("rooted in resentment"). This dichotomy underscores Nietzsche’s claim that conventional ethics are not objective but a political tool to dilute the "will to power" of exceptional individuals.

### **2. Assumption Analysis**  
**Question:** What unexamined assumption underlies Nietzsche’s critique of slave morality as a "perversion of life"? Analyze the implications if this assumption is challenged.  
**Answer:**  
The assumption is that "life’s true will to power" is a universal, instinctual drive that should not be morally constrained. If challenged (e.g., if cooperation, not domination, is life’s primary drive), Nietzsche’s critique collapses, as slave morality’s emphasis on equality could be seen as adaptive rather than perverse. This would invert his hierarchy, legitimizing collective ethics over individual supremacy.

### **3. Application to Another Context**  
**Question:** Apply Nietzsche’s framework to scientific progress. How might a "noble" vs. "slave" dynamic manifest in the pursuit of knowledge?  
**Answer:**  
A "noble" scientific ethos might prioritize unencumbered innovation (e.g., unfunded maverick researchers pursuing risky, visionary projects). Conversely, "slave morality" could manifest in bureaucratic, consensus-driven science (e.g., funding prioritizing collaborative, "socially responsible" projects over disruptive ideas). Nietzsche would argue the latter stifles genius, while defenders might see it as ensuring ethical accountability—a parallel to his moral critique.

```

---

### **Response 2** - Verify Question and Answer Pairs
```text
<think>
Okay, let me try to work through this. The user wants me to verify if the question and answer pairs generated earlier are correct, complete, and grounded in the document provided. First, I need to look at each question and answer carefully.

Starting with the first question: Interpretation of Thesis. The question asks to identify the central thesis and explain how the contrast between noble and slave morality supports Nietzsche’s claim. The answer given states the thesis correctly as traditional morality being a slave morality created by the weak. It also explains the contrast by saying noble morality is intrinsic and slave is reactive. The document does mention "noble type of man" experiencing himself as the goal and slave morality rooted in resentment. So the answer seems accurate here. The support from the document is there, so this pair is correct.

Next, the second question: Assumption Analysis. The question asks for an unexamined assumption in Nietzsche’s critique. The answer points out the assumption that the will to power is a universal drive that shouldn’t be constrained. The document mentions "life’s true will to power" as being perverted by slave morality. If this assumption is challenged, the critique would collapse. The implications analysis seems valid. The answer correctly identifies the assumption and discusses the consequences if it's challenged. This pair checks out too.

Third question: Application to Another Context. Applying Nietzsche’s framework to science. The answer differentiates between a noble ethos (unencumbered innovation) and slave (bureaucratic consensus). The document doesn’t mention science, but the framework of noble vs. slave morality is applied conceptually. Nietzsche’s ideas are being extended appropriately here. The answer stays true to his philosophy even though the context is new. So this is correct as well.

Now, checking completeness. Each question has a clear answer that addresses all parts. The answers reference the document specifically, like quoting "goal and measure of things" and "rooted in resentment." They don’t introduce information not in the document. The assumptions are correctly identified based on the text. 

Grounded in the document: All key points in the answers are backed by the document. For example, the Übermensch is mentioned in the document and used in the first answer to support transcending moral dichotomies. The will to power is directly from the document. 

Are there any inaccuracies? Let me re-read the document. The document says slave morality posits equality and humility to constrain natural superiority. The answers reflect that. The second answer’s assumption about will to power being universal is in the document as "life’s true will to power." The third question's application doesn't contradict the original ideas. 

I think all pairs are correct. The user might have been concerned if the answers added external info or misinterpreted the text, but they seem to align closely. Maybe the third question's answer could have a stronger link, but since it's an application, it's acceptable as long as the framework is applied correctly, which it is.
</think>

**Verification of Question & Answer Pairs**

All three question-answer pairs are **correct, complete, and grounded in the document**. Below is a detailed breakdown:

---

### **1. Interpretation of Thesis**  
- **Correctness**: ✅ The central thesis is accurately identified, and the contrast between "noble" and "slave" morality is explained using direct evidence from the document (e.g., "goal and measure of things," "rooted in resentment").  
- **Completeness**: ✅ Fully addresses both parts of the question (thesis + supporting contrast).  
- **Grounding**: ✅ Quotes and concepts (e.g., "intrinsic" vs. "reactive") align with the passage.  

---

### **2. Assumption Analysis**  
- **Correctness**: ✅ The assumption about "will to power" is correctly inferred from the phrase "perversion of life’s true will to power."  
- **Completeness**: ✅ Clearly states the assumption and analyzes implications if challenged.  
- **Grounding**: ✅ Directly tied to the document’s language about "life’s true will to power" and its contrast with slave morality.  

---

### **3. Application to Another Context**  
- **Correctness**: ✅ Faithfully applies Nietzsche’s framework to science without distorting his ideas.  
- **Completeness**: ✅ Distinguishes "noble" (unfettered innovation) and "slave" (bureaucratic) dynamics, engaging with potential counterarguments.  
- **Grounding**: ✅ While the *context* (science) is new, the moral dichotomy ("unencumbered pursuit" vs. "constrained collective ethics") is rooted in the document’s themes.  

---

### **Overall Strengths**  
1. **Document Alignment**: Answers cite specific phrases (e.g., "goal and measure of things," "rooted in resentment") to support claims.  
2. **Conceptual Accuracy**: No misinterpretation of Nietzsche’s terms (e.g., "slave morality" as reactive, "Übermensch" as transcendent).  
3. **Depth of Analysis**: Questions 2 and 3 require critical thinking (assumption-testing, application) beyond mere recall.  

### **Minor Suggestions for Enhancement (Optional)**  
- For **Question 3**, adding a brief phrase from the document to the answer (e.g., "mirroring how slave morality ‘constrains natural superiority’") could strengthen explicit grounding.  
- For **Question 2**, noting that the assumption also relies on Nietzsche’s value judgment that "domination" is more "natural" than cooperation could enrich the analysis.  

---

**Final Conclusion**: The pairs are well-designed, rigorous, and tightly anchored to the document. They effectively test interpretation, critical analysis, and thematic application—all hallmarks of GRE-style questions.  

Let me know if you’d like revisions or additional pairs!
```

---

### **Response 3** - Parse Question and Answer Pairs into JSON Format
```text
<think>
Okay, the user wants the question and answer pairs parsed into JSON format. Let me start by recalling the previous interaction. They had a document and three GRE-style questions with answers. Then they asked for verification, which I did, and now they want JSON.

First, I need to structure each question and answer properly. JSON requires key-value pairs. Each question should have a "Question" key, "Answer", and maybe "Type" to categorize them. The original questions were labeled as 1, 2, 3, so I can use those as IDs.

Looking at the first question: Interpretation of Thesis. The answer has bullet points for Correctness, Completeness, Grounding in the verification, but the actual answer is the paragraph before that. Wait, no. Wait, in the previous response, the answer for Question 1 was the paragraph starting with "The thesis is...", and then the verification had correctness etc. But the user now wants the original Q&A pairs in JSON, not the verification details.

So I need to go back to the initial Q&A provided. Let me check:

Original Q1:
**1. Interpretation of Thesis**  
**Question:** Identify the central thesis...  
**Answer:** The thesis is... 

Yes. So each entry in JSON should have the question text and the answer text. The verification part was separate, but the user didn't ask for that in JSON, just the pairs.

So for each of the three questions, structure as:

{
  "id": 1,
  "type": "Interpretation of Thesis",
  "question": "Identify the central thesis...",
  "answer": "The thesis is..."
},

And similarly for the others. Also, include the document in JSON if needed, but the user specified "question and answer pairs", so maybe just the Q&A.

Wait, the user said "parse the generated question and answer pairs into a json format". So the document is separate, but the pairs are the three questions with their answers.

So the JSON should be an array of objects, each containing id, type, question, answer.

Checking the original answer:

The three questions are:

1. Interpretation of Thesis
2. Assumption Analysis
3. Application to Another Context

Each has a Question and Answer.

So the JSON would look like:

{
  "questions": [
    {
      "id": 1,
      "type": "Interpretation of Thesis",
      "question": "Identify the central thesis of the passage and explain how the contrast between 'noble' and 'slave' morality supports Nietzsche’s broader philosophical claim.",
      "answer": "The thesis is that traditional morality is a subversive construct of the weak... This dichotomy underscores Nietzsche’s claim..."
    },
    {
      "id": 2,
      "type": "Assumption Analysis",
      "question": "What unexamined assumption underlies Nietzsche’s critique of slave morality as a 'perversion of life'? Analyze the implications if this assumption is challenged.",
      "answer": "The assumption is that 'life’s true will to power' is a universal, instinctual drive... This would invert his hierarchy..."
    },
    {
      "id": 3,
      "type": "Application to Another Context",
      "question": "Apply Nietzsche’s framework to scientific progress. How might a 'noble' vs. 'slave' dynamic manifest in the pursuit of knowledge?",
      "answer": "A 'noble' scientific ethos might prioritize unencumbered innovation... while defenders might see it as ensuring ethical accountability—a parallel to his moral critique."
    }
  ]
}

I need to make sure the quotes are properly escaped in JSON. Also, check if the user wants the document included. The instruction says "question and answer pairs", so probably not. But maybe confirm. The original document was separate, so likely just the Q&A.

Yes, the user's request was specifically about the generated pairs, so the above structure should be correct.

Now, formatting the JSON correctly with escaped quotes and proper syntax. Also, ensuring that each key is properly named. The types are the question types as labeled. The IDs are 1,2,3. Questions and answers are copied exactly from the original.

Double-checking for any markdown formatting in the answers. The original answers had line breaks but in JSON, line breaks can be preserved with \n or just as continuous text. Since the user didn't specify, keeping as continuous is safer.

So the final JSON should have the questions array with three entries as above.
</think>

Here’s the parsed JSON format for the question-answer pairs, including the document for context:

```json
{
  "document": [
    "Nietzsche argued that traditional morality is a \"slave morality\" created by the weak to subjugate the strong, claiming, \"The noble type of man experiences himself as the goal and measure of things; he does not require approval or justification from outsiders.\"",
    "In contrast, slave morality, rooted in resentment, posits equality and humility as virtues to constrain natural superiority.",
    "Nietzsche saw this as a perversion of life’s true will to power.",
    "Only the Übermensch could transcend such moral dichotomies."
  ],
  "questions": [
    {
      "id": 1,
      "type": "Interpretation of Thesis",
      "question": "Identify the central thesis of the passage and explain how the contrast between \"noble\" and \"slave\" morality supports Nietzsche’s broader philosophical claim.",
      "answer": "The thesis is that traditional morality is a subversive construct of the weak (\"slave morality\") to undermine natural hierarchy, contrasting with the self-affirming \"noble morality\" of the strong. The passage supports this by framing noble morality as intrinsic (e.g., \"the goal and measure of things\") and slave morality as reactive (\"rooted in resentment\"). This dichotomy underscores Nietzsche’s claim that conventional ethics are not objective but a political tool to dilute the \"will to power\" of exceptional individuals."
    },
    {
      "id": 2,
      "type": "Assumption Analysis",
      "question": "What unexamined assumption underlies Nietzsche’s critique of slave morality as a \"perversion of life\"? Analyze the implications if this assumption is challenged.",
      "answer": "The assumption is that \"life’s true will to power\" is a universal, instinctual drive that should not be morally constrained. If challenged (e.g., if cooperation, not domination, is life’s primary drive), Nietzsche’s critique collapses, as slave morality’s emphasis on equality could be seen as adaptive rather than perverse. This would invert his hierarchy, legitimizing collective ethics over individual supremacy."
    },
    {
      "id": 3,
      "type": "Application to Another Context",
      "question": "Apply Nietzsche’s framework to scientific progress. How might a \"noble\" vs. \"slave\" dynamic manifest in the pursuit of knowledge?",
      "answer": "A \"noble\" scientific ethos might prioritize unencumbered innovation (e.g., unfunded maverick researchers pursuing risky, visionary projects). Conversely, \"slave morality\" could manifest in bureaucratic, consensus-driven science (e.g., funding prioritizing collaborative, \"socially responsible\" projects over disruptive ideas). Nietzsche would argue the latter stifles genius, while defenders might see it as ensuring ethical accountability—a parallel to his moral critique."
    }
  ]
}

### **Key Features of the JSON Structure**
1. **Document**: Included as an array of sentences for clarity.
2. **Questions**: Array with `id`, `type`, `question`, and `answer` fields.
3. **Formatting**: Quotes are properly escaped (`\"`), and line breaks are omitted for compactness (add `\n` if line breaks are preferred).

### **Example Use Case (GRE Study Tool)**
```json
// Extracting a single question for display
{
  "question": questions[0].question,
  "answer":   questions[0].answer
}

Let me know if you’d like further refinements (e.g., adding metadata like "difficulty" or "keywords")!
```



```python
from datasets import load_dataset
icls = [{
  "icl_document": (
    "Nietzsche argued that traditional morality is a \"slave morality\" created by the weak to subjugate the strong, claiming, \"The noble type of man experiences himself as the goal and measure of things; he does not require approval or justification from outsiders.\"",
    "In contrast, slave morality, rooted in resentment, posits equality and humility as virtues to constrain natural superiority.",
    "Nietzsche saw this as a perversion of life’s true will to power.",
    "Only the Übermensch could transcend such moral dichotomies."
  ),
  "icl_query_1": "Identify the central thesis of the passage and explain how the contrast between \"noble\" and \"slave\" morality supports Nietzsche’s broader philosophical claim.",
  "icl_response_1": "The thesis is that traditional morality is a subversive construct of the weak (\"slave morality\") to undermine natural hierarchy, contrasting with the self-affirming \"noble morality\" of the strong. The passage supports this by framing noble morality as intrinsic (e.g., \"the goal and measure of things\") and slave morality as reactive (\"rooted in resentment\"). This dichotomy underscores Nietzsche’s claim that conventional ethics are not objective but a political tool to dilute the \"will to power\" of exceptional individuals.",
  
  "icl_query_2": "What unexamined assumption underlies Nietzsche’s critique of slave morality as a \"perversion of life\"? Analyze the implications if this assumption is challenged.",
  "icl_response_2": "The assumption is that \"life’s true will to power\" is a universal, instinctual drive that should not be morally constrained. If challenged (e.g., if cooperation, not domination, is life’s primary drive), Nietzsche’s critique collapses, as slave morality’s emphasis on equality could be seen as adaptive rather than perverse. This would invert his hierarchy, legitimizing collective ethics over individual supremacy.",

  "icl_query_3": "Apply Nietzsche’s framework to scientific progress. How might a \"noble\" vs. \"slave\" dynamic manifest in the pursuit of knowledge?",
  "icl_response_3": "A \"noble\" scientific ethos might prioritize unencumbered innovation (e.g., unfunded maverick researchers pursuing risky, visionary projects). Conversely, \"slave morality\" could manifest in bureaucratic, consensus-driven science (e.g., funding prioritizing collaborative, \"socially responsible\" projects over disruptive ideas). Nietzsche would argue the latter stifles genius, while defenders might see it as ensuring ethical accountability—a parallel to his moral critique."
},
{
    "icl_document": (
      "The coastal town of Willow Creek, once renowned for its pristine beaches, now struggles with rampant pollution. Plastic debris and oil spills have devastated marine life, prompting a decline in tourism and fishing industries. Residents have organized weekly clean-up initiatives, but the scale of the problem overwhelms their efforts.",
      "Technologists at the local university have developed an AI-powered buoy system to combat this. The buoys, equipped with solar panels and filtration technology, can identify and absorb oil spills while collecting microplastics. Data from the buoys is shared publicly, raising awareness and pressuring corporations to adopt sustainable practices. Though costly, the project has sparked hope for revitalizing the ecosystem and economy."
    ),
    "icl_query_1": "How does the technological solution address the economic *and* environmental challenges highlighted in the document?",
    "icl_response_1": "The buoys directly mitigate environmental harm by absorbing spills and collecting plastics, which could restore marine life. Economically, this restoration might revive tourism and fishing, reversing their decline. Public data-sharing also pressures corporations, potentially reducing pollution and creating a sustainable economic cycle aligned with environmental health.",
    
    "icl_query_2": "What implicit values or priorities do the community’s actions (clean-up initiatives) and the technologists’ project reflect, and how do these align or contrast?",
    "icl_response_2": "Both reflect a prioritization of environmental stewardship and community engagement. The clean-ups emphasize collective responsibility, while the buoy project highlights innovation and systemic accountability (targeting corporations). They align in seeking solutions but contrast in scale and approach—manual vs. technological, local vs. systemic.",
    
    "icl_query_3": "Imagine the buoy project succeeds. What unintended consequences might arise from its impact, considering document's themes?",
    "icl_response_3": "Success could lead to reduced community participation in clean-up efforts if residents perceive the technology as a sufficient solution. Economically, revitalization might exacerbate existing resource strains (e.g., overwhelming local infrastructure) or create dependency on the buoy system, undermining long-term sustainability if the technology falters."

}
]

# Load the quality corpus
ds = load_dataset("zitongyang/entigraph-quality-corpus", split='train').remove_columns(['entity', 'entigraph']).rename_columns({'raw': 'document', 'uid': 'document_outline'})

# Add ICLs to the dataset
ds = [ds.map(lambda x: icl) for icl in icls]
ds = concatenate_datasets(ds)

# Add additional columns needed for the knowledge generation pipeline
ds = ds.add_column('domain', ['Article']*len(ds))

# We now have the seed data ready. Let's save it
ds.to_json("seed_data.jsonl", orient='records', lines=True)
```

### Once we have the right set of ICLs, we now start generating the data
- For running nemotron with reasoning we need to add few new custom blocks to the pipeline
    - Block for post-processing the thinking output
    
    Note: Since merging of [#87](https://github.com/Red-Hat-AI-Innovation-Team/sdg_hub/pull/87) you can also simply use the `start` and `end` tag in prompt
    - Block for parsing based on regular expression
- We we also register nemotron thinking prompt with `detailed thinking on`



```python
from sdg_hub.flow import Flow
from sdg_hub.pipeline import Pipeline
from sdg_hub.sdg import SDG
from sdg_hub.prompts import PromptRegistry
from sdg_hub.blocks import BlockRegistry, Block
from transformers import AutoTokenizer
import re
from typing import List
from datasets import load_dataset, Dataset

tokenizer = AutoTokenizer.from_pretrained("nvidia/Llama-3_3-Nemotron-Super-49B-v1")

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

ds = load_dataset('json', data_files="seed_data.jsonl", split="train")
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

knowledge_agentic_pipeline = "flows/synth_knowledge_reasoning_nemotron_super_49b_summary_diversity.yaml"
flow_cfg = Flow(client).get_flow_from_file(knowledge_agentic_pipeline)
sdg = SDG(
    [Pipeline(flow_cfg)],
    num_workers=1,
    batch_size=1,
    save_freq=1000,
)
ds_subset = ds.select(range(1))

# Generate data
generated_data = sdg.generate(ds_subset, checkpoint_dir="Tmp")
```


```python
print("### Question")
print(generated_data['question'][2])
print("### Response")
print(generated_data['response'][2])
```

## Output from reasoning SDGpipeline

### Question
Apply the "necessity vs. cosmetic" dynamic from the article to the field of plastic surgery. How might the strategies dentists use to market discretionary treatments parallel or diverge from those in cosmetic surgery, and what does this reveal about consumer behavior in health industries?  

### Response
```text
<think>
Okay, let's tackle this question. The user wants me to apply the "necessity vs. cosmetic" dynamic from the dentistry article to plastic surgery. I need to compare the marketing strategies between dentists and cosmetic surgeons, see where they're similar or different, and what that says about consumer behavior.

First, I'll recall the article. Dentists shifted from necessity (fillings) to cosmetics (whitening, etc.) using tools like intra-oral cameras, scare tactics, and upselling. They exploited the lack of insurance coverage, making patients pay out-of-pocket for discretionary treatments.

Now, plastic surgery. Cosmetic procedures are already discretionary, so maybe the strategies are similar. Both use technology (like simulations) to show results. Dentists use intra-oral cameras; surgeons might use 3D imaging to show potential outcomes. Both might use marketing based on societal beauty standards.

But there might be differences. Plastic surgery is more invasive, so maybe more emphasis on safety and long-term results. Dentists use scare tactics like mercury in fillings; surgeons might focus on health benefits (e.g., breast reduction for back pain) even if the primary motive is cosmetic.

Consumer behavior: Both industries thrive on discretionary spending driven by vanity and anxiety induced by marketing. However, plastic surgery might have higher barriers due to cost and risk, so marketing might need to balance desire with reassurance about safety. The article mentions that dentists blur necessity and indulgence; surgeons might frame cosmetic procedures as investments in well-being or confidence.

Wait, the question also asks what this reveals about consumer behavior. Probably that consumers prioritize appearance over health needs when given the option, and industries exploit this by creating perceived needs through technology and marketing. The lack of insurance coverage forces consumers to weigh discretionary spending, leading them to prioritize wants over needs if marketed effectively.

Need to structure the answer: parallel strategies (technology, societal ideals, upselling), divergent strategies (invasiveness, health angles), and consumer insights (vanity-driven, influenced by marketing, willing to spend on desires when enabled by financial models).
</think>


Applying the "necessity vs. cosmetic" dynamic to plastic surgery reveals both parallels and divergences in marketing strategies. **Parallels** include the use of technology (e.g., 3D simulation in surgery, akin to intra-oral cameras in dentistry) to amplify perceived flaws and sell discretionary treatments. Both fields leverage societal beauty standards (e.g., "Great American Smile" in dentistry, ideal body types in surgery) and employ upselling tactics ("treatment acceptance" scripts in dentistry, bundling procedures in surgery). **Divergences** arise from surgery’s higher stakes: while dentists focus on low-risk, repeatable cosmetics, surgeons must balance vanity-driven marketing with reassurances about safety and long-term outcomes (e.g., framing breast augmentation as a "confidence investment").  

This dynamic reveals that consumer behavior in health industries is **vulnerability-driven** and **market-shaped**. Patients prioritize discretionary aesthetics over necessary care when industries successfully manufacture anxiety (e.g., "halimeters" for halitosis, or surgical "imperfection" analyses) and align treatments with cultural ideals. The absence of insurance coverage for cosmetics forces a calculus where consumers weigh desires against costs, often favoring indulgences when framed as transformative. Ultimately, both fields exploit a **decoupling of health needs from consumer demands**, proving that profitability in discretionary health markets hinges on blurring clinical necessity with emotional and social aspiration.  
```

## Output from non-reasoning SDG pipeline
### Question
Analyze the economic paradox underlying the dental industry’s shift: Why did patients prioritize discretionary cosmetic treatments over medically necessary care?
### Response
The paradox stems from dentistry’s reliance on out-of-pocket spending (60% of costs). Cosmetic procedures (e.g., bleaching) were marketed as immediate, desirable investments in appearance, while necessary care (e.g., gum surgery) was perceived as inconvenient or costly with delayed benefits. Dentists capitalized on this by aggressively selling "wants" to offset the resistance to "needs," ensuring profitability in a market where insurance coverage was limited.


```python
print("### Faithfulness Evaluation")
print(generated_data['explanation'][2])
print("### Judgment")
print(generated_data['judgment'][2])
```

### Faithfulness Evaluation
The question references a specific "article" without providing its content, violating Non-Referential Clarity as it relies on unprovided external context. While it requires specialized knowledge (Subject-Aware Completeness), the core framework ("necessity vs. cosmetic") is not defined within the question, making it dependent on external information.
### Judgment
YES

### Data Mixing for training

### Prepare Nemotron Replay Buffer


```python
from datasets import load_dataset, concatenate_datasets

nemotron_ds = load_dataset("nvidia/Llama-Nemotron-Post-Training-Dataset-v1", "SFT")
nemotron_ds = nemotron_ds.filter(lambda x: x['used_in_training'] == 'yes')
nemotron_ds = nemotron_ds.map(lambda x: {'question': x['input'][x['input'].find("user<|end_header_id|>") + len("user<|end_header_id|>") : x['input'].find("<|eot_id|><|start_header_id|>assistant")].strip()})
nemotron_ds = concatenate_datasets(nemotron_ds.values())
nemotron_ds = nemotron_ds.shuffle(seed=894375).select(range(200000))
nemotron_ds = nemotron_ds.add_column('unmask', [False]*nemotron_ds.num_rows)
nemotron_ds = nemotron_ds.map(lambda x: {'messages': [{'role': 'system', 'content': 'detailed thinking on'}, {'role': 'user', 'content': x['question']}, {'role': 'assistant', 'content': x['output']}]})
nemotron_ds = nemotron_ds.remove_columns(['input', 'output', 'category', 'license', 'reasoning', 'generator', 'used_in_training', 'question'])
nemotron_ds.to_json("nemotron_replay_buffer_data.jsonl", orient='records', lines=True)
```

### Create functions for data mixng with conversation templates


```python
from utils import create_training_mix
```

### Create quality training mix of: reasoning dataset, with non-reasoning dataset, nemotron replay buffer


```python
### For this tutorial, we will use the following document uids from the quality dataset:
DOC_UIDS = [
    ' Defining Decay Down by David Plotz',
    ' Fight Clubbed by David Plotz',
    ' I, Antichrist? by Jeffrey Goldberg',
    " It's Time To Keelhaul U-Haul! by Jeffrey Goldberg",
    " My Father's Estate by Ben Stein",
    '"Phone Me in Central Park" by McConnell, James V.',
    '...After a Few Words... by Garrett, Randall', 
    '...And It Comes Out Here by Del Rey, Lester',
    'A Coffin for Jacob by Ludwig, Edward W.',
    'A Fall of Glass by Lee, Stanley R.',
    'A Filbert Is a Nut by Raphael, Rick',
    'A Gift from Earth by Banister, Manly',
    'A Gleeb for Earth by Schafhauser, Charles',
    'A Good Year for the Roses? by David Edelstein',
    'A Pail of Air by Leiber, Fritz',
    'A Planet Named Joe by Hunter, Evan',
    "AI: what's the worst that could happen? by Harry Armstrong",
    'Accidental Death by Baily, Peter',
    'All Day September by Kuykendall, Roger',
    'Ambition by Bade, William L.',
    'And Then the Town Took Off by Wilson, Richard',
    'Atom Mystery [Young Atom Detective] by Coombs, Charles Ira',
    'Beach Scene by King, Marshall',
    'Big Ancestor by Wallace, F. L. (Floyd L.)',
    'Birds of a Feather by Silverberg, Robert',
    'Bodyguard by Gold, H. L. (Horace Leonard)'
]
```


```python
from datasets import load_dataset
 
# Load tokenizer for pre-training formatting
tokenizer = AutoTokenizer.from_pretrained("nvidia/Llama-3.1-Nemotron-Nano-8B-v1")

# Load non-reasoning dataset from nemotron super 49b
nemotron_non_reasoning_ds = load_dataset("json", data_dir="/new_data/knowledge_rh/quality/knowledge1.25_nemotron_super_49b/", split="train")
nemotron_non_reasoning_ds = nemotron_non_reasoning_ds.filter(lambda x: x['score'] == '2' and x['judgment'] == 'YES')
nemotron_non_reasoning_ds = nemotron_non_reasoning_ds.filter(lambda x: x['document_outline'] in DOC_UIDS)
print(nemotron_non_reasoning_ds)

# Load reasoning dataset from nemotron super 49b
nemotron_reasoning_ds = load_dataset("json", data_dir="/new_data/knowledge_rh/quality/synth_knowledge_reasoning_nemotron_super_49b/gen", split="train")
nemotron_reasoning_ds = nemotron_reasoning_ds.filter(lambda x: x['score'] == '2' and x['judgment'] == 'YES')
nemotron_reasoning_ds = nemotron_reasoning_ds.filter(lambda x: x['document_outline'] in DOC_UIDS)
print(nemotron_reasoning_ds)
# Load nemotron replay buffer
nemotron_ds_replay_buffer = load_dataset("json", data_files="/new_data/knowledge_rh/quality/training_mix/nemotron_replay_buffer_data.jsonl", split="train")


# Create non-reasoning training mix
nemotron_ds_training_mix = create_training_mix(nemotron_non_reasoning_ds, tokenizer, 'off').shuffle(seed=894375)

# Create reasoning training mix
nemotron_reasoning_ds = create_training_mix(nemotron_reasoning_ds, tokenizer, 'on')

# Concatenate reasoning and non-reasoning training mixes
quality_reasoning_ds = concatenate_datasets([nemotron_reasoning_ds, nemotron_ds_training_mix]).remove_columns(['metadata', 'id']) # .select(range(40000))
print(quality_reasoning_ds)

# Concatenate training mix with replay buffer
training_mix = concatenate_datasets([quality_reasoning_ds, nemotron_ds_replay_buffer.shuffle(seed=894375).select(range(len(quality_reasoning_ds)))])

print(training_mix)
training_mix.to_json("/new_data/knowledge_rh/quality/training_mix/quality_knowledge_1.25_nemotron_49b_first_24.jsonl", orient='records', lines=True)
```

### Train student model
- Setup the training by cloning `https://github.com/instructlab/training` and following the instructions in the README
- The create `train.py` using below code
    ```python
    import argparse
    from instructlab.training.config import TorchrunArgs,TrainingArgs,DistributedBackend,FSDPOptions
    from instructlab.training.main_ds import run_training
    import os
    def parse_args():
        parser = argparse.ArgumentParser(description='Training script with configurable paths')
        parser.add_argument('--data_path', type=str, required=True,
                        help='Path to the training data file')
        parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the model or model identifier')
        parser.add_argument('--chat_tmpl_path', type=str, required=True,
                        help='Path to the chat template file')
        parser.add_argument('--exp_dir', type=str, required=True,
                        help='Path to the experiment directory')
        parser.add_argument('--parent_exp_dir', type=str, required=True,
                        help='Path to the parent experiment directory')
        return parser.parse_args()

    def main():
        args = parse_args()
        
        torch_args = TorchrunArgs(
            nproc_per_node=8,
            nnodes=1,
            node_rank=0,
            rdzv_id=123,
            rdzv_endpoint="0.0.0.0:8888",
        )
        output_dir = os.path.join(args.parent_exp_dir, args.exp_dir)
        train_args = TrainingArgs(
            model_path=args.model_path,
            data_path=args.data_path,
            ckpt_output_dir=output_dir,
            data_output_dir="data/processed-data",
            max_seq_len=20000,
            max_batch_len=25000,
            num_epochs=5,
            effective_batch_size=256,
            learning_rate=5e-6,
            warmup_steps=25,
            save_samples=0,
            use_dolomite=False,
            checkpoint_at_epoch = True,
            accelerate_full_state_at_epoch = False,
            process_data=True,
            chat_tmpl_path=args.chat_tmpl_path,
            distributed_backend=DistributedBackend.FSDP,
            fsdp_options=FSDPOptions(cpu_offload_params=False),
        )

        run_training(torch_args=torch_args,train_args=train_args)

    if __name__ == "__main__":
        main()
    ```

- Now create bash script with run command

    ```shell
    python train.py \
    --data_path "quality_knowledge_1.25_nemotron_49b_first_24.jsonl" \
    --model_path "nvidia/Llama-3.1-Nemotron-Nano-8B-v1" \
    --chat_tmpl_path "<chat_template_path>" \
    --exp_dir "nano_customized_thinking_quality_model" \
    --parent_exp_dir "<parent_exp_dir>"
    ```

## For reference: scripts for running SDG, starting student model and training

### Start Teacher Model
```shell
export HUGGINGFACE_HUB_CACHE="/new_data/hf_cache"   
export HF_DATASETS_CACHE="/dev/shm/hf"
export HF_HOME="/new_data/hf_cache"
export HF_MODEL_CACHE="/new_data/hf_cache"

for i in $(seq 0 2 6); do
    port=$((8000 + i/2))
    CUDA_VISIBLE_DEVICES=$i,$((i+1)) python -m vllm.entrypoints.openai.api_server \
        --model nvidia/Llama-3_3-Nemotron-Super-49B-v1 \
        --dtype float16 \
        --tensor-parallel-size 2 \
        --port $port \
        --trust-remote-code > log_$((i/2+1)).log 2>&1 &
done
```

### Run SDG in parallel
```shell
# Get dataset size and save into variable
dataset_size=$(wc -l seed_data.jsonl | awk '{print $1}')
number_of_processes=4
port=8000
for i in {0..4}; do
    # Continue until i=4
    if [ $i -eq 4 ]; then
        break
    fi
    dataset_start_index=$((i * dataset_size / number_of_processes))
    dataset_end_index=$((dataset_start_index + dataset_size / number_of_processes))
    python generate.py --ds_path  seed_data.jsonl \
        --bs 2 --num_workers 10 \
        --save_path /new_data/knowledge_rh/quality/knowledge1.25_nemotron_super_49b/gen.jsonl \
        --flow flows/synth_knowledge1.5_nemotron_super_49b.yaml \
        --endpoint http://localhost:$port/v1 \
        --checkpoint_dir /new_data/knowledge_rh/quality/knowledge1.25_nemotron_super_49b/data_checkpoints \
        --save_freq 1000 \
        --dataset_start_index $dataset_start_index \
        --dataset_end_index $dataset_end_index > run_sdg_$i.log 2>&1 &
    echo "Starting process $i with dataset from $dataset_start_index to $dataset_end_index on port $port"
    port=$((port + 1))
done
```

### Run SDG on new reasoning pipeline
```shell
# Get dataset size and save into variable
dataset_size=$(wc -l seed_data.jsonl | awk '{print $1}')
number_of_processes=4
port=8000
for i in {0..4}; do
    # Continue until i=4
    if [ $i -eq 4 ]; then
        break
    fi
    dataset_start_index=$((i * dataset_size / number_of_processes))
    dataset_end_index=$((dataset_start_index + dataset_size / number_of_processes))
    python generate.py --ds_path  seed_data.jsonl \
        --bs 2 --num_workers 10 \
        --save_path /new_data/knowledge_rh/quality/synth_knowledge_reasoning_nemotron_super_49b/gen.jsonl \
        --flow synth_knowledge_reasoning_nemotron_super_49b.yaml \
        --endpoint http://localhost:$port/v1 \
        --checkpoint_dir /new_data/knowledge_rh/quality/synth_knowledge_reasoning_nemotron_super_49b/data_checkpoints \
        --save_freq 1000 \
        --dataset_start_index $dataset_start_index \
        --dataset_end_index $dataset_end_index > run_sdg_$i.log 2>&1 &
    echo "Starting process $i with dataset from $dataset_start_index to $dataset_end_index on port $port"
    port=$((port + 1))
done
```


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