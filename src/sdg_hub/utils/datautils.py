# Third Party
from datasets import concatenate_datasets

from os import access, R_OK
from os.path import exists, isfile


def safe_concatenate_datasets(datasets: list):
    """
    Concatenate datasets safely, ignoring any datasets that are None or empty.
    """
    filtered_datasets = [ds for ds in datasets if ds is not None and ds.num_rows > 0]

    if not filtered_datasets:
        return None

    return concatenate_datasets(filtered_datasets)


def assert_valid_file(file_path: str):
    """
    Assert that the file exists and is not empty.

    Parameters
    ----------
    file_path : str
        The path to the file to check.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file is empty.
    """
    if not exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    if not isfile(file_path):
        raise ValueError(f"Path is not a file: {file_path}")

    if not access(file_path, R_OK):
        raise ValueError(f"File is not readable: {file_path}")
