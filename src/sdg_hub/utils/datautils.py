# Third Party
from datasets import concatenate_datasets


def safe_concatenate_datasets(datasets: list):
    """
    Concatenate datasets safely, ignoring any datasets that are None or empty.
    """
    filtered_datasets = [ds for ds in datasets if ds is not None and ds.num_rows > 0]

    if not filtered_datasets:
        return None

    return concatenate_datasets(filtered_datasets)
