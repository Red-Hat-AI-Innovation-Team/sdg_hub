# Third Party
import numpy as np
import pandas as pd

# Local
from .error_handling import FlowValidationError


def safe_concatenate_datasets(datasets: list):
    """Concatenate datasets safely, ignoring any datasets that are None or empty."""
    filtered_datasets = [ds for ds in datasets if ds is not None and len(ds) > 0]

    if not filtered_datasets:
        return None

    return pd.concat(filtered_datasets, ignore_index=True)


def validate_no_duplicates(dataset: pd.DataFrame) -> None:
    """
    Validate that the input dataset contains only unique rows.

    Uses pandas `.duplicated()` for efficient duplicate detection, with preprocessing
    to handle numpy arrays that cause TypeError in pandas duplicate detection.
    Raises FlowValidationError if duplicates are found, including a count
    of the duplicate rows detected.

    Parameters
    ----------
    dataset : pd.DataFrame
        Input dataset to validate.

    Raises
    ------
    FlowValidationError
        If duplicate rows are detected in the dataset.
    """
    if len(dataset) == 0:
        return

    # Ensure we're working with a pandas DataFrame
    if not isinstance(dataset, pd.DataFrame):
        # If it's a HuggingFace Dataset, convert to pandas
        if hasattr(dataset, "to_pandas"):
            df = dataset.to_pandas()
        else:
            raise ValueError(
                f"Expected pandas DataFrame or HuggingFace Dataset, got {type(dataset)}"
            )
    else:
        df = dataset

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
        if isinstance(x, np.ndarray):
            if x.ndim == 0:
                return make_hashable(x.item())
            return tuple(make_hashable(i) for i in x)
        if isinstance(x, dict):
            # sort robustly even with heterogeneous key types
            return tuple(
                sorted(
                    ((k, make_hashable(v)) for k, v in x.items()),
                    key=lambda kv: repr(kv[0]),
                )
            )
        if isinstance(x, (set, frozenset)):
            # order‑insensitive
            return frozenset(make_hashable(i) for i in x)
        if hasattr(x, "__iter__"):
            # lists, tuples, custom iterables
            return tuple(make_hashable(i) for i in x)
        # last‑resort fallback to a stable representation
        return repr(x)

    # Apply to the whole dataframe to ensure every cell is hashable
    if hasattr(df, "map"):
        df = df.map(make_hashable)
    else:
        df = df.applymap(make_hashable)

    duplicate_count = int(df.duplicated(keep="first").sum())
    if duplicate_count > 0:
        raise FlowValidationError(
            f"Input dataset contains {duplicate_count} duplicate rows. "
            f"SDG Hub operations require unique input rows. "
            f"Please deduplicate your dataset before processing."
        )


def safe_concatenate_with_validation(
    datasets: list, context: str = "datasets"
) -> pd.DataFrame:
    """Safely concatenate datasets with schema validation and clear error messages.

    Parameters
    ----------
    datasets : list[pd.DataFrame]
        List of datasets to concatenate
    context : str
        Description of what's being concatenated for error messages

    Returns
    -------
    pd.DataFrame
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
        return pd.concat(valid_datasets, ignore_index=True)
    except Exception as e:
        # Schema mismatch or other concatenation error
        schema_info = []
        for i, ds in enumerate(valid_datasets):
            schema_info.append(f"Dataset {i}: columns={ds.columns.tolist()}")

        schema_details = "\n".join(schema_info)
        raise FlowValidationError(
            f"Schema mismatch when concatenating {context}. "
            f"All datasets must have compatible schemas (same columns/types). "
            f"Original error: {e}\n"
            f"Dataset schemas:\n{schema_details}"
        ) from e
