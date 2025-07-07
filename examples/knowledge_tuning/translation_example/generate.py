# Third Party
from datasets import load_dataset
from openai import OpenAI
import click

# First Party
from sdg_hub.flow import Flow
from sdg_hub.logger_config import setup_logger
from sdg_hub.sdg import SDG
from blocks.translation_block import TranslationBlock

logger = setup_logger(__name__)


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
    "--llm_endpoint", type=str, required=True, help="LLM Endpoint for data processing."
)
@click.option(
    "--translation_endpoint",
    type=str,
    required=True,
    help="Endpoint for Translation Model.",
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
@click.option(
    "--dataset_start_index", type=int, default=0, help="Start index of the dataset."
)
@click.option(
    "--dataset_end_index", type=int, default=None, help="End index of the dataset."
)
def main(
    ds_path: str,
    bs: int,
    num_workers: int,
    save_path: str,
    llm_endpoint: str,
    translation_endpoint: str,
    flow: str,
    checkpoint_dir: str,
    save_freq: int,
    debug: bool,
    dataset_start_index: int,
    dataset_end_index: int,
):
    """
    Process dataset through translation and generation pipeline.

    Parameters
    ----------
    ds_path : str
        Path to the dataset.
    bs : int
        Batch size for processing.
    num_workers : int
        Number of workers for parallel processing.
    save_path : str
        Path to save the output.
    llm_endpoint : str
        LLM endpoint for data processing.
    translation_endpoint : str
        Translation model endpoint.
    flow : str
        Flow configuration file path.
    checkpoint_dir : str
        Path to save checkpoints.
    save_freq : int
        Frequency to save checkpoints.
    debug : bool
        Enable debug mode with reduced dataset.
    dataset_start_index : int
        Start index for dataset slicing.
    dataset_end_index : int
        End index for dataset slicing.
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
        ds = ds.shuffle(seed=42).select(range(100))

    logger.warning(f"Dataset: {ds}")

    # Use environment variables or secure storage for API keys
    import os

    openai_api_key = os.environ.get("OPENAI_API_KEY", "EMPTY")
    if openai_api_key == "EMPTY":
        logger.warning(
            "Using empty API key. Make sure your endpoints don't require authentication."
        )
    llm_openai_api_base = llm_endpoint

    llm_client = OpenAI(
        api_key=openai_api_key,
        base_url=llm_openai_api_base,
    )

    # Verify we can see the model
    teacher_model = llm_client.models.list().data[0].id
    logger.warning(f"Connected to model: {teacher_model}")

    translation_openai_api_base = translation_endpoint

    translation_client = OpenAI(
        api_key=openai_api_key,
        base_url=translation_openai_api_base,
    )

    flow_cfg = Flow(llm_client=llm_client).get_flow_from_file(flow)

    # Track if we found any TranslationBlock instances
    translation_blocks_found = False

    for index in range(len(flow_cfg.chained_blocks)):
        try:
            if issubclass(
                flow_cfg.chained_blocks[index]["block_type"], TranslationBlock
            ):
                flow_cfg.chained_blocks[index]["block_config"][
                    "client"
                ] = translation_client
                translation_blocks_found = True
                logger.warning(f"Set client to {translation_client}")
        except (KeyError, TypeError) as e:
            logger.error(f"Error processing flow config at index {index}: {e}")
            raise ValueError(f"Invalid flow configuration at index {index}") from e

    if not translation_blocks_found:
        logger.warning("No TranslationBlock instances found in the flow configuration.")

    sdg = SDG(
        [flow_cfg],
        num_workers=num_workers,
        batch_size=bs,
        save_freq=save_freq,
    )

    generated_data = sdg.generate(ds, checkpoint_dir=checkpoint_dir)

    save_path = save_path.replace(
        ".jsonl", f"_{dataset_start_index}_{dataset_end_index}.jsonl"
    )
    generated_data.to_json(save_path, orient="records", lines=True, force_ascii=False)
    logger.info(f"Data saved to {save_path}")


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    main()
