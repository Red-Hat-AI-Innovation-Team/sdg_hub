from typing import Any, List, Optional
import json

import polars as pl


def get_avg_summaries_per_raw_doc(df: pl.DataFrame) -> float:
    """
    Calculate average summaries per raw document in the dataset.

    Args:
        df: Input dataframe with summary and document columns

    Returns:
        Average number of summaries per raw document
    """
    summary_counts = df.group_by("document").agg(
        pl.col("summary").n_unique().alias("unique_summaries")
    )
    avg_summaries = summary_counts["unique_summaries"].mean()

    return avg_summaries


def sample_docs(df: pl.DataFrame, n_docs_per_raw: int = 50) -> pl.DataFrame:
    """
    Sample unique summaries per document.

    Args:
        df: Input dataframe with 'summary', 'document', 'document_outline' columns
        n_docs_per_raw: Maximum number of unique summaries to sample per document (cut size)

    Returns:
        Sampled dataframe with 'summary', 'document', 'document_outline'
    """
    required_cols = [
        "summary",
        "document",
        "document_outline",
    ]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    avg_summaries = get_avg_summaries_per_raw_doc(df)
    if avg_summaries < n_docs_per_raw:
        print(
            f"⚠️ Warning: Cut size {n_docs_per_raw} exceeds available summaries (avg: {avg_summaries:.1f} per raw document)"
        )

    df_unique = df.group_by("summary").agg(
        [
            pl.col("document").first().alias("document"),
            pl.col("document_outline").first().alias("document_outline"),
        ]
    )

    sampled_docs = df_unique.group_by("document").map_groups(
        lambda g: g.sample(n=min(n_docs_per_raw, g.height))
    )

    return sampled_docs


def sample_doc_qa(
    df: pl.DataFrame, n_docs_per_raw: int = 50, qa_per_doc: int = 3
) -> pl.DataFrame:
    """
    Sample Q&A pairs from summaries with optional reasoning.

    Note: 'summary' column contains summaries, 'document' contains original documents.
    n_docs_per_raw is the number of unique summaries to sample per document.

    Args:
        df: Input dataframe with summary and Q&A data
        n_docs_per_raw: Maximum number of unique summaries to sample per document (cut size)
        qa_per_doc: Maximum number of Q&A pairs per summary

    Returns:
        Sampled dataframe with Q&A pairs
    """
    required_cols = [
        "question",
        "response",
        "summary",
        "document",
        "document_outline",
    ]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    avg_summaries = get_avg_summaries_per_raw_doc(df)
    if avg_summaries < n_docs_per_raw:
        print(
            f"⚠️ Warning: Cut size {n_docs_per_raw} exceeds available summaries (avg: {avg_summaries:.1f} per raw document)"
        )

    df = df.with_columns([pl.struct(["question", "response"]).alias("qa_pair")])

    agg_cols = [
        pl.col("qa_pair"),
        pl.col("document").first(),
        pl.col("document_outline").first(),
    ]

    if "parse_response_dict_reasoning_content" in df.columns:
        df = df.with_columns(
            [pl.col("parse_response_dict_reasoning_content").alias("reasoning")]
        )
        agg_cols.append(pl.col("reasoning").first())

    df = df.group_by("summary").agg(agg_cols)

    sampled_docs = df.group_by("document").map_groups(
        lambda g: g.sample(n=min(n_docs_per_raw, g.height))
    )

    sampled_docs = sampled_docs.with_columns(
        pl.col("qa_pair").list.slice(0, qa_per_doc)
    ).explode(pl.col("qa_pair"))

    sampled_docs = sampled_docs.with_columns(
        [
            pl.col("qa_pair").struct.field("question").alias("question"),
            pl.col("qa_pair").struct.field("response").alias("response"),
        ]
    ).drop("qa_pair")

    return sampled_docs


def _clean_response_text(df: pl.DataFrame) -> pl.DataFrame:
    """Clean response text by removing markers and whitespace."""
    return df.with_columns(
        pl.col("response")
        .str.replace_all(r"\[END\]", "")
        .str.replace_all(r"\[ANSWER\]", "")
        .str.strip_chars()
        .alias("response")
    )


def _create_metadata(df: pl.DataFrame) -> pl.Expr:
    """Create metadata JSON structure."""
    return (
        pl.struct(
            [
                pl.col("summary").alias("sdg_document"),
                pl.lit("document_knowledge_qa").alias("dataset"),
                pl.col("document"),
            ]
        )
        .map_elements(json.dumps)
        .alias("metadata")
    )


def _create_messages_with_reasoning(record: dict) -> List[dict]:
    """Create message structure with reasoning (thinking)."""
    return [
        {
            "role": "user",
            "content": f"{record['document_outline']}\n{record['summary']}\n\n{record['question']}",
            "thinking": None,
        },
        {
            "role": "assistant",
            "content": record["response"],
            "thinking": record["reasoning"],
        },
    ]


def _create_messages_with_reasoning_no_document(record: dict) -> List[dict]:
    """Create message structure with reasoning."""
    return [
        {
            "role": "user",
            "content": f"In {record['document_outline']}, {record['question']}",
            "thinking": None,
        },
        {
            "role": "assistant",
            "content": record["response"],
            "thinking": record["reasoning"],
        },
    ]


def _create_messages_without_reasoning(record: dict) -> List[dict]:
    """Create message structure without reasoning."""
    return [
        {
            "role": "user",
            "content": f"{record['document_outline']}\n{record['summary']}\n\n{record['question']}",
            "thinking": None,
        },
        {"role": "assistant", "content": record["response"], "thinking": ""},
    ]


def _create_messages_without_reasoning_no_document(record: dict) -> List[dict]:
    """Create message structure without reasoning."""
    return [
        {
            "role": "user",
            "content": f"In {record['document_outline']}, {record['question']}",
            "thinking": None,
        },
        {"role": "assistant", "content": record["response"], "thinking": ""},
    ]


def generate_knowledge_qa_dataset(
    generated_dataset: pl.DataFrame,
    keep_columns: Optional[List[str]] = None,
    pre_training: bool = False,
    dataset_name: str = "document_knowledge_qa",
    keep_document_in_context: bool = False,
) -> pl.DataFrame:
    """
    Generate knowledge Q&A dataset in chat format.

    Args:
        generated_dataset: Input dataframe with Q&A data
        keep_columns: Additional columns to keep in output
        pre_training: Whether to add unmask column for pre-training
        dataset_name: Name for the dataset metadata

    Returns:
        Formatted dataset with messages and metadata
    """
    if keep_columns is None:
        keep_columns = []

    required_cols = [
        "question",
        "response",
        "summary",
        "document_outline",
        "document",
    ]
    missing_cols = [
        col for col in required_cols if col not in generated_dataset.columns
    ]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    generated_dataset = _clean_response_text(generated_dataset)

    base_columns = [_create_metadata(generated_dataset)]

    has_reasoning = "reasoning" in generated_dataset.columns

    if has_reasoning and not keep_document_in_context:
        message_columns = [
            "question",
            "response",
            "summary",
            "document_outline",
            "reasoning",
        ]
        messages_expr = (
            pl.struct(message_columns)
            .map_elements(_create_messages_with_reasoning_no_document)
            .alias("messages")
        )
    elif has_reasoning and keep_document_in_context:
        message_columns = [
            "question",
            "response",
            "summary",
            "document_outline",
            "reasoning",
        ]
        messages_expr = (
            pl.struct(message_columns)
            .map_elements(_create_messages_with_reasoning)
            .alias("messages")
        )
    elif keep_document_in_context:
        message_columns = ["question", "response", "summary", "document_outline"]
        messages_expr = (
            pl.struct(message_columns)
            .map_elements(_create_messages_without_reasoning)
            .alias("messages")
        )
    else:
        message_columns = ["question", "response", "summary", "document_outline"]
        messages_expr = (
            pl.struct(message_columns)
            .map_elements(_create_messages_without_reasoning_no_document)
            .alias("messages")
        )

    base_columns.append(messages_expr)

    knowledge_ds = generated_dataset.with_columns(base_columns)

    final_columns = keep_columns + ["messages", "metadata"]
    knowledge_ds = knowledge_ds.select(final_columns)
    if pre_training:
        knowledge_ds = knowledge_ds.with_columns(pl.lit(True).alias("unmask"))
    else:
        knowledge_ds = knowledge_ds.with_columns(pl.lit(False).alias("unmask"))

    return knowledge_ds


def count_len_in_tokens(
    df: pl.DataFrame, tokenizer: Any, column_name: str = "messages"
) -> pl.DataFrame:
    """
    Count token length of messages using tokenizer.

    Args:
        df: Input dataframe
        tokenizer: HuggingFace tokenizer with apply_chat_template method
        column_name: Column containing messages to tokenize

    Returns:
        Dataframe with added token_length column
    """
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in dataframe")

    def apply_chat_template(messages: List[dict]) -> str:
        """Apply chat template to messages."""
        return tokenizer.apply_chat_template(messages, tokenize=False)

    def count_tokens(text: str) -> int:
        """Count tokens in text."""
        return len(tokenizer.encode(text))

    if column_name == "messages":
        expr = (
            pl.col(column_name)
            .map_elements(apply_chat_template, return_dtype=pl.String)
            .map_elements(count_tokens, return_dtype=pl.Int32)
            .alias("token_length")
        )
    else:
        expr = (
            pl.col(column_name)
            .map_elements(count_tokens, return_dtype=pl.Int32)
            .alias("token_length")
        )
    return df.with_columns(expr)
