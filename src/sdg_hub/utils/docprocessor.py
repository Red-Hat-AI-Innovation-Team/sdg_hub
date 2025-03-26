# Standard
from pathlib import Path
import json

# Third Party
from datasets import Dataset
from tabulate import tabulate
from transformers import AutoTokenizer
import yaml

# First Party
from sdg_hub.logger_config import setup_logger

# Local
from .datautils import safe_concatenate_datasets
from .chunking import chunk_document

logger = setup_logger(__name__)


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
        Returns:
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
        Returns:
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
        Returns:
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
        Returns:
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
        Returns:
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
                chunks_mds.append({
            "document": text,
            "document_outline": self.user_config["document_outline"],
            "document_title": md_file,
            "domain": self.user_config["domain"],
            })
        chunk_ds = Dataset.from_list(chunks_mds)
        chunk_ds_with_icls = self._add_icls(chunk_ds)
        return chunk_ds_with_icls