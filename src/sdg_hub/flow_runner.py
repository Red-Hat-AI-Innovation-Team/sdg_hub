"""Script for running data generation flows with configurable parameters."""

# Standard
from importlib import resources
from typing import Optional
import os
import sys
import traceback

# Third Party
from datasets import load_dataset
from openai import OpenAI
import click
import yaml

# First Party
from sdg_hub.flow import Flow
from sdg_hub.logger_config import setup_logger
from sdg_hub.sdg import SDG
from sdg_hub.utils.error_handling import (
    APIConnectionError,
    DataGenerationError,
    DataSaveError,
    DatasetLoadError,
    FlowConfigurationError,
    FlowRunnerError,
)
from sdg_hub.utils.path_resolution import resolve_path

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
    dataset_end_index: Optional[int] = None,
    api_key: Optional[str] = None,
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
    dataset_start_index : int, optional
        Start index for dataset slicing, by default 0.
    dataset_end_index : Optional[int], optional
        End index for dataset slicing, by default None.
    api_key : Optional[str], optional
        API key for the remote endpoint. If not provided, will use OPENAI_API_KEY environment variable, by default None.

    Returns
    -------
    None

    Raises
    ------
    DatasetLoadError
        If the dataset cannot be loaded or processed.
    FlowConfigurationError
        If the flow configuration is invalid or cannot be loaded.
    APIConnectionError
        If connection to the API endpoint fails.
    DataGenerationError
        If data generation fails during processing.
    DataSaveError
        If saving the generated data fails.
    """
    logger.info(f"Generation configuration: {locals()}\n\n")

    try:
        # Load and validate dataset
        try:
            ds = load_dataset("json", data_files=ds_path, split="train")
            logger.info(
                f"Successfully loaded dataset from {ds_path} with {len(ds)} rows"
            )
        except Exception as e:
            raise DatasetLoadError(
                f"Failed to load dataset from '{ds_path}'. "
                f"Please check if the file exists and is a valid JSON file.",
                details=str(e),
            ) from e

        # Apply dataset slicing if specified
        try:
            if dataset_start_index is not None and dataset_end_index is not None:
                if dataset_start_index >= len(ds) or dataset_end_index > len(ds):
                    raise DatasetLoadError(
                        f"Dataset slice indices ({dataset_start_index}, {dataset_end_index}) "
                        f"are out of bounds for dataset with {len(ds)} rows"
                    )
                if dataset_start_index >= dataset_end_index:
                    raise DatasetLoadError(
                        f"Start index ({dataset_start_index}) must be less than end index ({dataset_end_index})"
                    )
                ds = ds.select(range(dataset_start_index, dataset_end_index))
                logger.info(
                    f"Dataset sliced from {dataset_start_index} to {dataset_end_index}"
                )

            if debug:
                if len(ds) < 30:
                    logger.warning(
                        f"Debug mode requested 30 samples but dataset only has {len(ds)} rows"
                    )
                ds = ds.shuffle(seed=42).select(range(min(30, len(ds))))
                logger.info(
                    f"Debug mode enabled. Using {len(ds)} samples from the dataset."
                )
        except DatasetLoadError:
            raise
        except Exception as e:
            raise DatasetLoadError(
                "Failed to process dataset slicing or debug mode.", details=str(e)
            ) from e

        # Validate API configuration
        openai_api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not openai_api_key or openai_api_key == "EMPTY":
            logger.warning("API key not provided and OPENAI_API_KEY not set or is 'EMPTY'. API calls may fail.")

        openai_api_base = endpoint
        if not openai_api_base:
            raise APIConnectionError("API endpoint cannot be empty")

        # Initialize OpenAI client
        try:
            client = OpenAI(
                api_key=openai_api_key or "EMPTY",
                base_url=openai_api_base,
            )
            # test connection with a model list
            models = client.models.list()
            logger.info(f"Initialized OpenAI client with endpoint: {openai_api_base}")
            logger.info(f"Available models: {[model.id for model in models.data]}")
        except Exception as e:
            raise APIConnectionError(
                f"Failed to initialize OpenAI client with endpoint '{openai_api_base}'. "
                f"Please check if the endpoint is valid and accessible.",
                details=str(e),
            ) from e

        # Load and validate flow configuration
        try:
            base_path = str(resources.files(__package__))
            flow_path = resolve_path(flow_path, [".", base_path])
            if not os.path.exists(flow_path):
                raise FlowConfigurationError(
                    f"Flow configuration file not found: {flow_path}"
                )

            # Validate flow file is readable YAML
            try:
                with open(flow_path, "r", encoding="utf-8") as f:
                    flow_config = yaml.safe_load(f)
                if not flow_config:
                    raise FlowConfigurationError(
                        f"Flow configuration file is empty: {flow_path}"
                    )
                logger.info(f"Successfully loaded flow configuration from {flow_path}")
            except yaml.YAMLError as e:
                raise FlowConfigurationError(
                    f"Flow configuration file '{flow_path}' contains invalid YAML.",
                    details=str(e),
                ) from e
            except Exception as e:
                raise FlowConfigurationError(
                    f"Failed to read flow configuration file '{flow_path}'.",
                    details=str(e),
                ) from e

            flow = Flow(client).get_flow_from_file(flow_path)
            logger.info("Successfully initialized flow from configuration")
        except FlowConfigurationError:
            raise
        except Exception as e:
            raise FlowConfigurationError(
                f"Failed to create flow from configuration file '{flow_path}'. "
                f"Please check the flow configuration format and block definitions.",
                details=str(e),
            ) from e

        # Initialize SDG and generate data
        try:
            sdg = SDG(
                flows=[flow],
                num_workers=num_workers,
                batch_size=batch_size,
                save_freq=save_freq,
            )
            logger.info(
                f"Initialized SDG with {num_workers} workers, batch size {batch_size}"
            )

            # Ensure checkpoint directory exists if specified
            if checkpoint_dir and not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir, exist_ok=True)
                logger.info(f"Created checkpoint directory: {checkpoint_dir}")

            generated_data = sdg.generate(ds, checkpoint_dir=checkpoint_dir)

            if generated_data is None or len(generated_data) == 0:
                raise DataGenerationError(
                    "Data generation completed but no data was generated. "
                    "This may indicate issues with the flow configuration or input data."
                )

            logger.info(f"Successfully generated {len(generated_data)} rows of data")

        except Exception as e:
            if isinstance(e, DataGenerationError):
                raise
            raise DataGenerationError(
                "Data generation failed during processing. This could be due to:"
                "\n- API connection issues with the endpoint"
                "\n- Invalid flow configuration or block parameters"
                "\n- Insufficient system resources (try reducing batch_size or num_workers)"
                "\n- Input data format incompatibility",
                details=f"Endpoint: {openai_api_base}, Error: {e}",
            ) from e

        # Save generated data
        try:
            # Adjust save path for dataset slicing
            final_save_path = save_path
            if dataset_end_index is not None and dataset_start_index is not None:
                final_save_path = save_path.replace(
                    ".jsonl", f"_{dataset_start_index}_{dataset_end_index}.jsonl"
                )

            # Ensure save directory exists
            save_dir = os.path.dirname(final_save_path)
            if save_dir and not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)
                logger.info(f"Created save directory: {save_dir}")

            generated_data.to_json(final_save_path, orient="records", lines=True)
            logger.info(f"Data successfully saved to {final_save_path}")

        except Exception as e:
            raise DataSaveError(
                f"Failed to save generated data to '{final_save_path}'. "
                f"Please check write permissions and disk space.",
                details=str(e),
            ) from e

    except (
        DatasetLoadError,
        FlowConfigurationError,
        APIConnectionError,
        DataGenerationError,
        DataSaveError,
    ):
        # Re-raise our custom exceptions with their detailed messages
        raise
    except Exception as e:
        # Catch any unexpected errors
        logger.error(f"Unexpected error during flow execution: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise FlowRunnerError(
            "An unexpected error occurred during flow execution. "
            "Please check the logs for more details.",
            details=str(e),
        ) from e


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
@click.option(
    "--dataset_start_index", type=int, default=0, help="Start index of the dataset."
)
@click.option(
    "--dataset_end_index", type=int, default=None, help="End index of the dataset."
)
@click.option(
    "--api_key",
    type=str,
    default=None,
    help="API key for the remote endpoint. If not provided, will use OPENAI_API_KEY environment variable.",
)
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
    dataset_end_index: Optional[int],
    api_key: Optional[str],
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
    dataset_start_index : int
        Start index for dataset slicing.
    dataset_end_index : Optional[int]
        End index for dataset slicing.
    api_key : Optional[str]
        API key for the remote endpoint. If not provided, will use OPENAI_API_KEY environment variable.

    Returns
    -------
    None
    """
    try:
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
            api_key=api_key,
        )
    except (
        DatasetLoadError,
        FlowConfigurationError,
        APIConnectionError,
        DataGenerationError,
        DataSaveError,
        FlowRunnerError,
    ) as e:
        logger.error(f"Flow execution failed: {e}")
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Flow execution interrupted by user")
        click.echo("Flow execution interrupted by user", err=True)
        sys.exit(130)  # Standard exit code for SIGINT
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        click.echo(
            f"Unexpected error occurred. Please check the logs for details. Error: {e}",
            err=True,
        )
        sys.exit(1)


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    main()
