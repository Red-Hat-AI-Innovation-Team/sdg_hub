# Third Party
from datasets import Dataset, concatenate_datasets
import numpy as np

# Local
from .error_handling import FlowValidationError


def safe_concatenate_datasets(datasets: list):
    """Concatenate datasets safely, ignoring any datasets that are None or empty."""
    filtered_datasets = [ds for ds in datasets if ds is not None and ds.num_rows > 0]

    if not filtered_datasets:
        return None

    return concatenate_datasets(filtered_datasets)


def validate_no_duplicates(dataset: Dataset) -> None:
    """
    Validate that the input dataset contains only unique rows.

    Uses pandas `.duplicated()` for efficient duplicate detection, with preprocessing
    to handle numpy arrays that cause TypeError in pandas duplicate detection.
    Raises FlowValidationError if duplicates are found, including a count
    of the duplicate rows detected.

    Parameters
    ----------
    dataset : Dataset
        Input dataset to validate.

    Raises
    ------
    FlowValidationError
        If duplicate rows are detected in the dataset.
    """
    if len(dataset) == 0:
        return

    df = dataset.to_pandas()

    def is_hashable(x):
        try:
            hash(x)
            return True
        except TypeError:
            return False

    def make_hashable(x):
        if is_hashable(x):
            # int, float, str, bytes, None etc. are already hashable
            return x
        elif isinstance(x, dict):
            return tuple(sorted((k, make_hashable(v)) for k, v in x.items()))
        elif hasattr(x, "__iter__"):
            # list, tuple, set, ndarray, or other customized iterables
            return tuple(make_hashable(i) for i in x)

    # Apply to the whole dataframe to ensure every cell is hashable
    df = df.map(make_hashable)

    duplicate_count = int(df.duplicated(keep="first").sum())

    if duplicate_count > 0:
        raise FlowValidationError(
            f"Input dataset contains {duplicate_count} duplicate rows. "
            f"SDG Hub operations require unique input rows. "
            f"Please deduplicate your dataset before processing."
        )


def safe_concatenate_with_validation(
    datasets: list, context: str = "datasets"
) -> Dataset:
    """Safely concatenate datasets with schema validation and clear error messages.

    Parameters
    ----------
    datasets : list[Dataset]
        List of datasets to concatenate
    context : str
        Description of what's being concatenated for error messages

    Returns
    -------
    Dataset
        Concatenated dataset

    Raises
    ------
    FlowValidationError
        If schema mismatch prevents concatenation or no valid datasets
    """
    # Filter out None and empty datasets first
    valid_datasets = [ds for ds in datasets if ds is not None and len(ds) > 0]

    if not valid_datasets:
        raise FlowValidationError(f"No valid datasets to concatenate in {context}")

    if len(valid_datasets) == 1:
        return valid_datasets[0]

    try:
        return concatenate_datasets(valid_datasets)
    except Exception as e:
        # Schema mismatch or other concatenation error
        schema_info = []
        for i, ds in enumerate(valid_datasets):
            schema_info.append(f"Dataset {i}: columns={ds.column_names}")

        schema_details = "\n".join(schema_info)
        raise FlowValidationError(
            f"Schema mismatch when concatenating {context}. "
            f"All datasets must have compatible schemas (same columns/types). "
            f"Original error: {e}\n"
            f"Dataset schemas:\n{schema_details}"
        ) from e
