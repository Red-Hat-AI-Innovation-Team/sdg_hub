"""Utility functions for dataset manipulation and processing.

This module provides utility functions for working with datasets, particularly
focusing on safe operations that handle edge cases and potential errors.
"""

# Third Party
from datasets import concatenate_datasets


def safe_concatenate_datasets(datasets: list):
    """Concatenate datasets safely, ignoring any datasets that are None or empty.

    This function provides a safe way to concatenate multiple datasets by:
    1. Filtering out None values and empty datasets
    2. Only attempting concatenation if there are valid datasets
    3. Returning None if no valid datasets are found

    Parameters
    ----------
    datasets : list
        A list of datasets to concatenate. Each element should be a dataset
        from the Hugging Face datasets library, or None.

    Returns
    -------
    Dataset or None
        The concatenated dataset if there are valid datasets to concatenate,
        None otherwise.

    Examples
    --------
    >>> from datasets import Dataset
    >>> ds1 = Dataset.from_dict({"a": [1, 2]})
    >>> ds2 = Dataset.from_dict({"a": [3, 4]})
    >>> result = safe_concatenate_datasets([ds1, ds2])
    >>> print(result.num_rows)
    4
    >>> result = safe_concatenate_datasets([None, ds1])
    >>> print(result.num_rows)
    2
    >>> result = safe_concatenate_datasets([None, None])
    >>> print(result)
    None
    """
    # Filter out None values and empty datasets
    filtered_datasets = [ds for ds in datasets if ds is not None and ds.num_rows > 0]

    # Return None if no valid datasets are found
    if not filtered_datasets:
        return None

    # Concatenate the valid datasets
    return concatenate_datasets(filtered_datasets)
