# Third Party
from datasets import load_dataset
from openai import OpenAI
import click
import json
import os

# First Party
from sdg_hub.flow import Flow
from sdg_hub.logger_config import setup_logger
from sdg_hub.pipeline import Pipeline
from sdg_hub.sdg import SDG
from sdg_hub.prompts import PromptRegistry
from sdg_hub.blocks import BlockRegistry, Block

logger = setup_logger(__name__)

@click.command()
@click.option(
    "--input_file",
    type=click.Path(exists=True),
    required=True,
    help="Path to the input JSON file containing product specifications.",
)
@click.option(
    "--output_file",
    type=click.Path(),
    required=True,
    help="Path to save the generated product descriptions.",
)
@click.option(
    "--endpoint",
    type=str,
    required=True,
    help="Endpoint for the LLM API.",
)
@click.option(
    "--flow",
    type=str,
    default="flows/product_description_generation.yaml",
    help="Path to the flow configuration file.",
)
@click.option(
    "--batch_size",
    type=int,
    default=1,
    show_default=True,
    help="Batch size for processing.",
)
@click.option(
    "--num_workers",
    type=int,
    default=1,
    show_default=True,
    help="Number of workers for parallel processing.",
)
@click.option(
    "--checkpoint_dir",
    type=click.Path(),
    default="checkpoints",
    help="Directory to save checkpoints.",
)
@click.option(
    "--save_freq",
    type=int,
    default=1,
    show_default=True,
    help="Frequency to save checkpoints.",
)
@click.option(
    "--debug",
    is_flag=True,
    help="Enable debug mode for testing with a small dataset.",
)
def main(
    input_file,
    output_file,
    endpoint,
    flow,
    batch_size,
    num_workers,
    checkpoint_dir,
    save_freq,
    debug,
):
    """
    Generate product descriptions from technical specifications using the SDG pipeline.
    
    Parameters:
    input_file (str): Path to the input JSON file containing product specifications
    output_file (str): Path to save the generated product descriptions
    endpoint (str): Endpoint for the LLM API
    flow (str): Path to the flow configuration file
    batch_size (int): Batch size for processing
    num_workers (int): Number of workers for parallel processing
    checkpoint_dir (str): Directory to save checkpoints
    save_freq (int): Frequency to save checkpoints
    debug (bool): Enable debug mode for testing
    """
    logger.info(f"Starting product description generation with configuration: {locals()}\n")
    
    # Create checkpoint directory if it doesn't exist
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Load the input dataset
    try:
        with open(input_file, 'r') as f:
            input_data = json.load(f)
        
        # Convert to the format expected by the dataset
        if isinstance(input_data, list):
            dataset_data = input_data
        else:
            dataset_data = [input_data]
            
        # Create a dataset with the required format
        ds = load_dataset("json", data_files={"train": dataset_data}, split="train")
        
        if debug:
            # For debugging, use a smaller subset of the dataset
            ds = ds.shuffle(seed=42).select(range(min(3, len(ds))))
            logger.info(f"Debug mode: Using {len(ds)} samples")
    except Exception as e:
        logger.error(f"Error loading input file: {str(e)}")
        raise
    
    # Initialize the OpenAI client
    openai_api_key = "EMPTY"  # Replace with your API key if needed
    openai_api_base = endpoint
    
    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )
    
    # Load the flow configuration
    flow_cfg = Flow(client).get_flow_from_file(flow)
    
    # Initialize the SDG pipeline
    sdg = SDG(
        [Pipeline(flow_cfg)],
        num_workers=num_workers,
        batch_size=batch_size,
        save_freq=save_freq,
    )
    
    # Generate product descriptions
    try:
        logger.info("Starting product description generation...")
        generated_data = sdg.generate(ds, checkpoint_dir=checkpoint_dir)
        
        # Save the generated data
        generated_data.to_json(output_file, orient="records", lines=True)
        logger.info(f"Generated product descriptions saved to {output_file}")
        
    except Exception as e:
        logger.error(f"Error during generation: {str(e)}")
        raise

if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    main() 