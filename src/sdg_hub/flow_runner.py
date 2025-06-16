"""Script for running data generation flows with configurable parameters."""

# Standard
import os

# Third Party
from datasets import load_dataset
from openai import OpenAI
import click

# First Party
from sdg_hub.flow import Flow
from sdg_hub.logger_config import setup_logger
from sdg_hub.sdg import SDG


logger = setup_logger(__name__)


def run_flow(
    ds_path: str,
    save_path: str,
    endpoint: str,
    flow_path: str,
    checkpoint_dir: str,
    batch_size: int = 8,
    num_workers: int = 32,
    save_freq: int = 2,
    debug: bool = False,
    dataset_start_index: int = 0,
    dataset_end_index: int = None,
) -> None:
    """Process the dataset using the specified configuration.

    Parameters
    ----------
    ds_path : str
        Path to the dataset file.
    save_path : str
        Path where the output will be saved.
    endpoint : str
        API endpoint for data processing.
    flow_path : str
        Path to the flow configuration file.
    checkpoint_dir : str
        Directory path for saving checkpoints.
    batch_size : int, optional
        Batch size for processing, by default 8.
    num_workers : int, optional
        Number of worker processes to use, by default 32.
    save_freq : int, optional
        Frequency (in batches) at which to save checkpoints, by default 2.
    debug : bool, optional
        If True, enables debug mode with a smaller dataset subset, by default False.

    Returns
    -------
    None

    Raises
    ------
    FileNotFoundError
        If the flow configuration file is not found.
    """
    logger.info(f"Generation configuration: {locals()}\n\n")
    ds = load_dataset("json", data_files=ds_path, split="train")
    if dataset_start_index is not None and dataset_end_index is not None:
        ds = ds.select(range(dataset_start_index, dataset_end_index))
        logger.info(f"Dataset sliced from {dataset_start_index} to {dataset_end_index}")
    if debug:
        ds = ds.shuffle(seed=42).select(range(30))
        logger.info("Debug mode enabled. Using a subset of the dataset.")

    openai_api_key = os.environ.get("OPENAI_API_KEY", "EMPTY")
    openai_api_base = endpoint

    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )

    if not os.path.exists(flow_path):
        raise FileNotFoundError(f"Flow file not found: {flow_path}")

    flow = Flow(client).get_flow_from_file(flow_path)
    sdg = SDG(
        flows=[flow],
        num_workers=num_workers,
        batch_size=batch_size,
        save_freq=save_freq,
    )

    generated_data = sdg.generate(ds, checkpoint_dir=checkpoint_dir)
    if dataset_end_index is not None and dataset_start_index is not None:
        save_path = save_path.replace(".jsonl", f"_{dataset_start_index}_{dataset_end_index}.jsonl")
    generated_data.to_json(save_path, orient="records", lines=True)
    logger.info(f"Data saved to {save_path}")


@click.command()
@click.option(
    "--ds_path",
    type=click.Path(exists=True),
    required=True,
    help="Path to the dataset.",
)
@click.option(
    "--bs",
    type=int,
    default=8,
    show_default=True,
    help="Batch size for processing.",
)
@click.option(
    "--num_workers",
    type=int,
    default=32,
    show_default=True,
    help="Number of worker processes to use.",
)
@click.option(
    "--save_path",
    type=click.Path(),
    required=True,
    help="Path to save the output.",
)
@click.option(
    "--endpoint",
    type=str,
    required=True,
    help="API endpoint for data processing.",
)
@click.option(
    "--flow",
    type=click.Path(exists=True),
    required=True,
    help="Flow configuration for the process.",
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
@click.option(
    "--debug",
    is_flag=True,
    help="Enable debug mode with a smaller dataset subset.",
)
@click.option("--dataset_start_index", type=int, default=0, help="Start index of the dataset.")
@click.option("--dataset_end_index", type=int, default=None, help="End index of the dataset.")
def main(
    ds_path: str,
    bs: int,
    num_workers: int,
    save_path: str,
    endpoint: str,
    flow: str,
    checkpoint_dir: str,
    save_freq: int,
    debug: bool,
    dataset_start_index: int,
    dataset_end_index: int,
) -> None:
    """CLI entry point for running data generation flows.

    Parameters
    ----------
    ds_path : str
        Path to the dataset file.
    bs : int
        Batch size for processing.
    num_workers : int
        Number of worker processes to use.
    save_path : str
        Path where the output will be saved.
    endpoint : str
        API endpoint for data processing.
    flow : str
        Path to the flow configuration file.
    checkpoint_dir : str
        Directory path for saving checkpoints.
    save_freq : int
        Frequency (in batches) at which to save checkpoints.
    debug : bool
        If True, enables debug mode with a smaller dataset subset.

    Returns
    -------
    None
    """
    run_flow(
        ds_path=ds_path,
        batch_size=bs,
        num_workers=num_workers,
        save_path=save_path,
        endpoint=endpoint,
        flow_path=flow,
        checkpoint_dir=checkpoint_dir,
        save_freq=save_freq,
        debug=debug,
        dataset_start_index=dataset_start_index,
        dataset_end_index=dataset_end_index,
    )


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    main()
