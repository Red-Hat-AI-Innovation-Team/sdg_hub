# Product Description Generation Pipeline

This example demonstrates how to use the `sdg` package to generate compelling product descriptions from technical specifications. The pipeline uses a series of LLM blocks to transform technical product details into engaging, customer-friendly descriptions.

## Overview

The workflow includes:

1. Analyzing technical specifications to create a clear summary
2. Extracting key features that would appeal to customers
3. Generating a compelling product description
4. Evaluating the quality of the generated description

## Pipeline Structure

The pipeline consists of several LLM blocks that process the input in sequence:

1. **Technical Summary Generation**: Analyzes the technical specifications and creates a clear, concise summary
2. **Key Features Extraction**: Identifies the most compelling features from the technical summary
3. **Product Description Generation**: Creates an engaging product description based on the key features
4. **Quality Evaluation**: Assesses the quality of the generated description

## How to Use

### Prerequisites

- Python 3.8 or higher
- The `sdg` package installed
- Access to an LLM API endpoint

### Input Format

The input should be a JSON file containing product specifications. See `sample_input.json` for an example format.

### Running the Pipeline

To generate product descriptions, use the `generate.py` script:

```bash
python generate.py \
  --input_file sample_input.json \
  --output_file generated_descriptions.json \
  --endpoint YOUR_LLM_API_ENDPOINT \
  --flow flows/product_description_generation.yaml \
  --batch_size 1 \
  --num_workers 1 \
  --checkpoint_dir checkpoints \
  --save_freq 1
```

### Command Line Arguments

- `--input_file`: Path to the input JSON file containing product specifications
- `--output_file`: Path to save the generated product descriptions
- `--endpoint`: Endpoint for the LLM API
- `--flow`: Path to the flow configuration file (default: flows/product_description_generation.yaml)
- `--batch_size`: Batch size for processing (default: 1)
- `--num_workers`: Number of workers for parallel processing (default: 1)
- `--checkpoint_dir`: Directory to save checkpoints (default: checkpoints)
- `--save_freq`: Frequency to save checkpoints (default: 1)
- `--debug`: Enable debug mode for testing with a small dataset

### Output Format

The output will be a JSON file containing the generated product descriptions along with quality scores and feedback. Each entry will include:

- Original product specifications
- Technical summary
- Key features
- Generated product description
- Quality scores and feedback

## Customization

You can customize the pipeline by:

1. Modifying the prompts in the `prompts/` directory
2. Adjusting the flow configuration in `flows/product_description_generation.yaml`
3. Changing the evaluation criteria in the quality check

## Example

See `sample_input.json` for an example of the input format and the expected output structure.

## Troubleshooting

If you encounter any issues:

1. Check that your input file is in the correct format
2. Verify that your LLM API endpoint is accessible
3. Ensure you have the required permissions to access the API
4. Check the logs for any error messages

## License

This example is provided under the same license as the `sdg` package. 