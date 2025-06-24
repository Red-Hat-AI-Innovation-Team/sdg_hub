# Synthetic Data Generation Tutorial using LLaMA and Mixtral

This tutorial demonstrates how to use SDG repository to generate synthetic question-answer pairs from documents using large language models like LLaMA 3.3 70B. We will also generate data using Mixtral model for comparison. We'll cover:

1. Setting up the environment
2. Connecting to LLM servers
3. Configuring the data generation pipeline
4. Generating data with different models
5. Comparing results


```python
# Enable auto-reloading of modules - useful during development
%load_ext autoreload
%autoreload 2
```

### Setup Instructions

Before running this notebook, you'll need to:

```bash 
pip install sdg-hub==0.1.0a4
```


```python
# Import required libraries
# datasets: For handling our data
# OpenAI: For interfacing with the LLM servers
# SDG components: For building our data generation pipeline
from datasets import load_dataset, Dataset
from openai import OpenAI

from sdg_hub.flow import Flow
from sdg_hub.pipeline import Pipeline
from sdg_hub.sdg import SDG
from sdg_hub.registry import PromptRegistry
```

### Setting up LLaMA 3.3 70B Model

First, we need to host the LLaMA model using vLLM. This creates an OpenAI-compatible API endpoint.

1. Start the vLLM server (run in terminal):
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.3-70B-Instruct \
    --dtype float16 \
    --tensor-parallel-size 8 
```

2. Connect to the model using OpenAI client below:


```python
# Configure OpenAI client to connect to our local vLLM server
endpoint = f"http://localhost:8000/v1"
openai_api_key = "EMPTY"  # vLLM doesn't require real API key
openai_api_base = endpoint

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

# Verify we can see the model
teacher_model = client.models.list().data[0].id
print(f"Connected to model: {teacher_model}")
```

### Configure the Data Generation Pipeline

Now we'll set up our Synthetic Data Generation (SDG) pipeline with the following components:
1. SDG Flow configuration from YAML
2. SDG Pipeline setup
3. SDG configuration with batch processing, number of workers, and save frequency parameters


```python
# Load the flow configuration from YAML file
flow_cfg = Flow(client).get_flow_from_file("synth_knowledge1.5_llama3.3.yaml")

# Initialize the SDG pipeline with processing parameters
sdg = SDG(
    [Pipeline(flow_cfg)],
    num_workers=1,      # Number of parallel workers
    batch_size=1,       # Batch size for processing
    save_freq=1000,     # How often to save checkpoints
)
```

### Load and Prepare Seed Data

We'll load our seed data (documents) that will be used to generate question-answer pairs.


```python
# Load the seed data from JSON file
seed_data_path = "Your seed data path"  # Replace with your data path
ds = load_dataset('json', data_files=seed_data_path, split='train')

# For testing, we'll use just one example
ds = ds.select(range(1))
```

### Generate Data with LLaMA 3.3

Now we'll use our configured pipeline to generate synthetic question-answer pairs.


```python
# Generate synthetic data and save checkpoints
generated_data = sdg.generate(ds, checkpoint_dir="Tmp")
```

### Setting up Mixtral Model

For comparison, we'll also generate data using the Mixtral model. First, start the Mixtral server:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.3-70B-Instruct \
    --dtype float16 \
    --tensor-parallel-size 8 
```


```python
# Connect to Mixtral model running on a different server
mistral_client = OpenAI(
    api_key="EMPTY",
    base_url=f"http://10.7.0.15:8000/v1",  # Update with your Mixtral server address
)

# Verify connection to Mixtral model
mistral_client_teacher_model = mistral_client.models.list().data[0].id
print(f"Connected to Mixtral model: {mistral_client_teacher_model}")
```

### Configure Mixtral Pipeline

Set up a similar pipeline for Mixtral model generation.


```python
# Create flow configuration for Mixtral
flow_cfg_mistral = Flow(mistral_client).get_flow_from_file(
    "../../src/sdg_hub/flows/generation/knowledge/synth_knowledge1.5.yaml"
)

# Initialize SDG pipeline for Mixtral
sdg_mistral = SDG(
    [Pipeline(flow_cfg_mistral)],
    num_workers=1,
    batch_size=1,
    save_freq=1000,
)
```

### Generate Data with Mixtral

Generate synthetic data using the Mixtral model for comparison.


```python
# Generate data using Mixtral model
generated_data_mistral = sdg_mistral.generate(ds, checkpoint_dir="Tmp")
```

### Compare Generated Data

Let's compare the outputs from both models by saving them to a markdown file for easy review.


```python
# Save comparison results to markdown file
k = 5  # Number of examples to compare
output_file = "model_comparison.md"

with open(output_file, "w") as f:
    # Write the source document first
    f.write(f"### Document \n{generated_data[0]['document']}")
    
    # Compare generated Q&A pairs
    for i in range(min(len(generated_data), len(generated_data_mistral))):
        f.write("Example #{}\n".format(i+1))
        
        # LLaMA 3.3 results
        f.write("### Result from llama3.3\n")
        f.write(generated_data[i]['question'] + "\n")
        f.write("*******************************\n")
        f.write(generated_data[i]['response'] + "\n")
        f.write("=================================\n")
        
        # Mixtral results
        f.write("### Result from mistral\n") 
        f.write(generated_data_mistral[i]['question'] + "\n")
        f.write("*******************************\n")
        f.write(generated_data_mistral[i]['response'] + "\n")
        f.write("\n\n")

print(f"Wrote {k} examples to {output_file}")
```

### Production Usage

For large-scale data generation, use the command-line script instead of this notebook:

```bash
python scripts/generate.py --ds_path seed_data.jsonl \
    --bs 2 --num_workers 10 \
    --save_path <your_save_path> \
    --flow ../src/sdg_hub/flows/generation/knowledge/synth_knowledge1.5.yaml \
    --checkpoint_dir <your_checkpoint_dir> \
    --endpoint <your_endpoint>
```

Note: For LLaMA 3.3, use `synth_knowledge1.5_llama3.3.yaml` as the flow configuration file.
