# SPDX-License-Identifier: Apache-2.0

# Standard
from pathlib import Path
from typing import List
import json
import os
import random
import re
import uuid

# Third Party
from datasets import Dataset, concatenate_datasets
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter
from tabulate import tabulate
from transformers import AutoTokenizer
import yaml

# First Party
from sdg_hub.logger_config import setup_logger
from sdg_hub.utils.datautils import safe_concatenate_datasets
import sdg_hub

logger = setup_logger(__name__)
_DEFAULT_CHUNK_OVERLAP = 100


def create_auxiliary_dataset(generated_dataset: Dataset):
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
    rec["messages"] = [
        {
            "role": "pretraining",
            "content": f"<|user|>\n{rec['messages'][0]['content']}\n<|assistant|>\n{rec['messages'][1]['content']}",
        }
    ]
    return rec


def mask_qa_per_doc(ds: Dataset, keep_no_qa_per_doc: int = 3) -> Dataset:
    """
    Mark QA entries per document for pre-training vs fine-tuning.

    Parameters
    ----------
    ds : Dataset
        Input dataset containing documents and QA pairs
    keep_no_qa_per_doc : int, default=3
        Number of QA entries per document to mark as unmask (pre-training)

    Returns
    -------
    Dataset
        Dataset with added 'unmask' boolean column indicating pre-training entries
    """

    unmask_entries = []
    mask_entries = []
    doc_count = {}

    for i, doc in enumerate(ds["document"]):
        if doc not in doc_count:
            doc_count[doc] = 1
        else:
            doc_count[doc] += 1

        entry = ds[i].copy()
        if doc_count[doc] <= keep_no_qa_per_doc:
            entry["unmask"] = True
            unmask_entries.append(entry)
        else:
            entry["unmask"] = False
            mask_entries.append(entry)

    ds_new = concatenate_datasets(
        [Dataset.from_list(unmask_entries), Dataset.from_list(mask_entries)]
    )
    return ds_new


def generate_knowledge_qa_dataset(
    generated_dataset: Dataset,
    keep_context_separate: bool = False,
    keep_document_outline: bool = False,
    keep_columns: List[str] = None,
    filter_non_pre_training: bool = True,
    keep_no_qa_per_doc: int = 3,
):
    generated_dataset = generated_dataset.map(
        lambda x: {
            "response": x["response"]
            .replace("[END]", "")
            .replace("[ANSWER]", "")
            .strip()
        },
        num_proc=10,
    )
    generated_dataset = mask_qa_per_doc(
        generated_dataset, keep_no_qa_per_doc=keep_no_qa_per_doc
    )
    if filter_non_pre_training:
        generated_dataset = generated_dataset.filter(lambda x: x["unmask"])

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
        __create_qa_row,
        remove_columns=[
            e
            for e in generated_dataset.column_names
            if e not in keep_columns + ["unmask"]
        ],
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
    Create a knowledge dataset for the Skills Phase of knowledge tuning.  
    
    This function generates QA datasets with RAFT-style context separation  
    and optionally includes auxiliary datasets for enhanced training.  
    
    Parameters  
    ----------  
    generated_dataset : Dataset  
        The input dataset containing generated knowledge content  
        
    Returns  
    -------  
    Dataset  
        Processed dataset ready for skills phase training 
    """
    knowledge_ds = generate_knowledge_qa_dataset(
        generated_dataset, keep_context_separate=True
    )
    knowledge_ds = build_raft_dataset(knowledge_ds, p=0.4)

    auxiliary_dataset = create_auxiliary_dataset(generated_dataset)
    if auxiliary_dataset is not None:
        knowledge_ds = safe_concatenate_datasets([knowledge_ds, auxiliary_dataset])
    return knowledge_ds


def create_knowledge_pretraining_ds(generated_dataset: Dataset, add_auxiliary_dataset: bool = True):
    # Phase 0.7 (Knowledge Phase)
    """  
    Create a knowledge dataset for the Knowledge Phase of knowledge tuning.  
    
    This function generates QA datasets for pretraining-style knowledge tuning  
    with optional auxiliary dataset inclusion.  
    
    Parameters  
    ----------  
    generated_dataset (Dataset): The dataset containing generated knowledge data.  
    add_auxiliary_dataset (bool): Whether to include an auxiliary dataset.  
    
    Returns  
    -------  
    Dataset: The generated knowledge dataset.  
    """
    knowledge_ds = generate_knowledge_qa_dataset(
        generated_dataset, keep_context_separate=False)
    knowledge_ds = knowledge_ds.map(_conv_pretrain)

    auxiliary_dataset = create_auxiliary_dataset(generated_dataset)
    if auxiliary_dataset is not None and add_auxiliary_dataset:
        auxiliary_dataset = auxiliary_dataset.map(_conv_pretrain)
        knowledge_ds = safe_concatenate_datasets([knowledge_ds, auxiliary_dataset])
    return knowledge_ds


def fuse_texts(text_list, short_length_threshold=100):
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
    pass


def create_tokenizer():
    return AutoTokenizer.from_pretrained("instructlab/granite-7b-lab")


def get_token_count(text, tokenizer):
    return len(tokenizer.tokenize(text))


def add_heading_formatting(text):
    text = text.split(".")
    # TODO: Change this from hardcoded to something that makes sense
    if len(text) > 1 and len(text[0].split(" ")) < 3:
        text = f"**{text[0]}**" + ".".join(text[1:])
    else:
        text = ".".join(text)
    return text


def generate_table_from_parsed_rep(item):
    """
    Generate the table from the parsed representation and return
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
    parts = table_ref.split("/")
    table_text = generate_table_from_parsed_rep(json_book[parts[1]][int(parts[2])])
    return table_text


def get_table_page_number(json_book, idx):
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


class DocProcessor:
    def __init__(
        self,
        parsed_doc_dir: Path,
        tokenizer: str = "instructlab/granite-7b-lab",
        user_config_path: Path = None,
    ):
        self.parsed_doc_dir = self._path_validator(parsed_doc_dir)
        self.user_config = self._load_user_config(
            self._path_validator(user_config_path)
        )
        self.docling_jsons = list(self.parsed_doc_dir.glob("*.json"))
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)

    def _path_validator(self, path) -> Path:
        """
        Validate the path and return a Path object.
        Args:
            path (str): Path to be validated.

        Returns
        -------
            Path`: Path object.
        """
        if isinstance(path, str):
            path = Path(path)
            if not path.exists():
                raise FileNotFoundError(f"{path} does not exist.")
        return path

    def _load_user_config(self, user_config_path: Path) -> dict:
        """
        Load the user config file.
        Args:
            user_config_path (Path): Path to the user config file.

        Returns
        -------
            dict: User config dictionary.
        """
        # load user config as yaml
        with open(user_config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _process_parsed_docling_json(self, json_fp: Path) -> Dataset:
        """
        Process the parsed docling json file and return a dataset.
        Args:
            json_fp (str): Path to the parsed docling json file.

        Returns
        -------
            Dataset: Dataset object.
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
        Add the ICLS label to the dataset.
        Args:
            dataset (Dataset): Dataset object.

        Returns
        -------
            Dataset: Dataset object with ICLS label.
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
        Process all the parsed docling json files and return a dataset.

        Returns
        -------
            Dataset: Dataset object.
        """
        datasets = []
        for json_fp in self.docling_jsons:
            chunk_ds = self._process_parsed_docling_json(json_fp)
            chunk_ds_with_icls = self._add_icls(chunk_ds)
            datasets.append(chunk_ds_with_icls)
        return safe_concatenate_datasets(datasets)

    def get_processed_markdown_dataset(self, list_md_files: list[Path]) -> Dataset:
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
