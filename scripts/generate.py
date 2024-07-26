# Third Party
from datasets import load_dataset
from openai import OpenAI
import click

# First Party
from instructlab.sdg.default_flows import DEFAULT_FLOW_FILE_MAP, Flow
from instructlab.sdg.logger_config import setup_logger
from instructlab.sdg.pipeline import Pipeline
from instructlab.sdg.sdg import SDG


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
    "--endpoint", type=str, required=True, help="Endpoint for data processing."
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
def main(
    ds_path,
    bs,
    num_workers,
    save_path,
    endpoint,
    flow,
    checkpoint_dir,
    save_freq,
    debug,
):
    """
    Main function to process the dataset.

    Parameters:
    ds_path (str): Path to the dataset.
    bs (int): Batch size.
    num_workers (int): Number of workers.
    save_path (str): Path to save the output.
    endpoint (str): Endpoint for data processing.
    flow (str): Flow configuration for the process.
    checkpoint_dir (str): Path to save checkpoints.
    save_freq (int): Frequency to save checkpoints.
    debug (bool): Enable debug mode.
    """
    logger.info(f"Generation configuration: {locals()}\n\n")
    ds = load_dataset("json", data_files=ds_path, split="train")

    if debug:
        # For debugging, use a smaller subset of the dataset
        ds = ds.shuffle(seed=42).select(range(30))

    openai_api_key = "EMPTY"
    openai_api_base = endpoint

    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )

    flow_cfg = Flow(client).get_flow_from_file(DEFAULT_FLOW_FILE_MAP[flow])
    sdg = SDG(
        [Pipeline(flow_cfg)],
        num_workers=num_workers,
        batch_size=bs,
        save_freq=save_freq,
    )

    generated_data = sdg.generate(ds, checkpoint_dir=checkpoint_dir)
    generated_data.to_json(save_path, orient="records", lines=True)


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    main()
