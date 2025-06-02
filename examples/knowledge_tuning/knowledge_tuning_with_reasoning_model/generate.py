# Third Party
from datasets import load_dataset
from openai import OpenAI
import click

# First Party
from sdg_hub.flow import  Flow
from sdg_hub.logger_config import setup_logger
from sdg_hub.pipeline import Pipeline
from sdg_hub.sdg import SDG
from sdg_hub.prompts import PromptRegistry
from sdg_hub.blocks import BlockRegistry, Block
from transformers import AutoTokenizer
import re
from typing import List
from datasets import Dataset

logger = setup_logger(__name__)

### Nemotron Chat Template with detailed thinking on
@PromptRegistry.register("nvidia/Llama-3_3-Nemotron-Super-49B-v1")
def nemotron_chat_template():
    return """{{- bos_token }}
{{- "<|start_header_id|>system<|end_header_id|>\n\n" }}detailed thinking on{{- "<|eot_id|>" }}
{%- for message in messages %}
  {%- if message['role'] == 'assistant' and '</think>' in message['content'] %}
    {%- set content = message['content'].split('</think>')[-1].lstrip() %}
  {%- else %}
    {%- set content = message['content'] %}
  {%- endif %}
  {{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' + content | trim + '<|eot_id|>' }}
{%- endfor %}
{%- if add_generation_prompt %}
  {{- '<|start_header_id|>assistant<|end_header_id|>\n\n' }}
{%- endif %}"""


@BlockRegistry.register("PostProcessThinkingBlock")
class PostProcessThinkingBlock(Block):
    def __init__(self, block_name: str, column_name: str) -> None:
        super().__init__(block_name=block_name)  
        self.column_name = column_name
    
    
    def generate(self, samples: Dataset):
        def post_process_thinking(x):
            if '</think>' in x[self.column_name]:
                x[self.column_name] = x[self.column_name].split('</think>')[-1].lstrip()
            return x
        samples = samples.map(post_process_thinking)
        return samples

@BlockRegistry.register("RegexParserBlock")
class RegexParserBlock(Block):
    def __init__(self, block_name: str, column_name: str, parsing_pattern: str="", parser_cleanup_tags: List[str]=[], output_cols: List[str]=[]) -> None:
        super().__init__(block_name=block_name)
        self.column_name = column_name
        self.parsing_pattern = parsing_pattern
        self.parser_cleanup_tags = parser_cleanup_tags
        self.output_cols = output_cols

    def generate(self, samples: Dataset):
        
        if self.parsing_pattern:
            new_data = []
            for sample in samples:
                parsed_outputs = self._parse(sample[self.column_name])
                
                max_length = max(len(value) for value in parsed_outputs.values())
                for values in zip(*(lst[:max_length] for lst in parsed_outputs.values())):
                    new_data.append({**sample, **dict(zip(parsed_outputs.keys(), values))})
            samples = Dataset.from_list(new_data)
        if self.parser_cleanup_tags:
            for clean_tag in self.parser_cleanup_tags:
               samples = samples.map(lambda x: {column_name: x[column_name].replace(clean_tag, "") for column_name in self.output_cols})
        return samples

    def _parse(self, generated_string):      
        pattern = re.compile(self.parsing_pattern, re.DOTALL)
        all_matches = pattern.findall(generated_string)
        matches = {column_name: [] for column_name in self.output_cols}
        if all_matches and isinstance(all_matches[0], tuple):
            for match in all_matches:
                for column_name, value in zip(self.output_cols, match):
                    value = value.strip()
                    # for clean_tag in self.parser_cleanup_tags:
                    #     value = value.replace(clean_tag, "")
                    matches[column_name].append(value)
        else:
            matches[self.output_cols[0]] = (
                [match.strip() for match in all_matches] if all_matches else []
            )
        return matches

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
@click.option("--dataset_start_index", type=int, default=0, help="Start index of the dataset.")
@click.option("--dataset_end_index", type=int, default=None, help="End index of the dataset.")
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
    dataset_start_index,
    dataset_end_index,
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
    if dataset_start_index is not None and dataset_end_index is not None:
        if dataset_end_index > len(ds):
            dataset_end_index = len(ds)
        ds = ds.select(range(dataset_start_index, dataset_end_index))
        logger.info(f"Dataset sliced from {dataset_start_index} to {dataset_end_index}")

    if debug:
        # For debugging, use a smaller subset of the dataset
        ds = ds.shuffle(seed=42).select(range(30))

    openai_api_key = "EMPTY"
    openai_api_base = endpoint

    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )

    flow_cfg = Flow(client).get_flow_from_file(flow)
    sdg = SDG(
        [Pipeline(flow_cfg)],
        num_workers=num_workers,
        batch_size=bs,
        save_freq=save_freq,
    )
    generated_data = sdg.generate(ds, checkpoint_dir=checkpoint_dir)
    
    save_path = save_path.replace(".jsonl", f"_{dataset_start_index}_{dataset_end_index}.jsonl")
    generated_data.to_json(save_path, orient="records", lines=True)
    logger.info(f"Data saved to {save_path}")


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    main()
