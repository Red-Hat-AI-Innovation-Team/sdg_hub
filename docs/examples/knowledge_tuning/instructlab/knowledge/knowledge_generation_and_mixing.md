```python
%load_ext autoreload
%autoreload 2
```

### Install SDG
```bash 
pip install sdg-hub==0.1.0a2
pip install rich datasets tabulate transformers
```
 - If you haven't already, run the document pre-processing notebook to create the seed data


```python
# Third Party
from datasets import load_dataset
from openai import OpenAI

# First Party
from sdg_hub.flow import Flow
from sdg_hub.pipeline import Pipeline
from sdg_hub.sdg import SDG
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '..', '..')))
from knowledge_utils import DocProcessor
import sys
```

### Setup OpenAI Client for interacting with the model


```python
endpoint = f"http://localhost:8000/v1"
openai_api_key = "EMPTY"
openai_api_base = endpoint

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)
teacher_model = client.models.list().data[0].id
print(teacher_model)
```

### Run SDG
- This will create knowledge flow from provided yaml file
- We will run this on small dataset for demo purposes
- For large scale generation, please use the python command provided in the next cell
- You can analyze the generated data to ensure the quality is similar to proivded QnA pairs


```python
knowledge_agentic_pipeline = "../../../src/instructlab/sdg/flows/generation/knowledge/synth_knowledge1.5.yaml"
flow_cfg = Flow(client).get_flow_from_file(knowledge_agentic_pipeline)
sdg = SDG(
    [Pipeline(flow_cfg)],
    num_workers=1,
    batch_size=1,
    save_freq=1000,
)
```


```python
number_of_samples = 5
seed_data_dir = f"sdg_demo_output/"
ds = load_dataset('json', data_files=f'{seed_data_dir}/seed_data.jsonl', split='train')
ds = ds.shuffle(seed=42).select(range(number_of_samples))
```


```python
# Checkpoint directory is used to save the intermediate datasets
generated_data = sdg.generate(ds, checkpoint_dir="Tmp")
```

### Run SDG through python command (For large scale generation)

```python
python /home/lab/sdg/scripts/generate.py --ds_path {output_dir}/seed_data.jsonl --bs 8 --num_workers 8 --save_path {output_dir}/gen.jsonl --flow ../src/instructlab/sdg/flows/generation/knowledge/synth_knowledge1.5.yaml --endpoint {teacher_endpoint_url} --checkpoint_dir {output_dir}/data_checkpoints --save_freq 2
```

### Save the generated data into training format


```python
from knowledge_utils import create_knowledge_regular_ds, create_knowledge_pretraining_ds

from datasets import concatenate_datasets

output_dir = f"sdg_demo_output/"

# Add the system prompt to final dataset if needed. For 
#  we use system prompt similar to below
system_prompt_lab = (
    "I am a LAB Instruct Model, an AI language model developed by Red Hat and IBM Research based on the granite-3.1-8b-base model. My primary role is to serve as a chat assistant."
)

# This is a general instruction tuning dataset that is mixed with generated knowledge to train LLM simultaneously on your knowledge and general instructions.
precomputed_skills_path = "<LAB precomputed skills path>"
precomputed_skills = load_dataset('json', data_files=precomputed_skills_path, split='train')

generated_ds = load_dataset('json', data_files=f'{output_dir}/gen.jsonl', split='train')

# Create Pretraining Knowledge Dataset (Also known as Phase 0.7/Phase 7)
phase_0_7_ds = create_knowledge_pretraining_ds(generated_ds)
phase_0_7_ds.to_json(f'{output_dir}/phase_0_7_ds.jsonl', orient='records', lines=True)

# Create Regular Knowledge Dataset (Also known as Phase 1.0/Phase 10)
phase_1_ds = create_knowledge_regular_ds(generated_ds)

# Mix the pre-computed skills with the regular knowledge dataset. If more than one dataset were generated simply add those in this concatenation stage.
# If you have any generated instruction data, that can be also mixed in this stage. If you only have generated skills phase 07 generation and training can be skipped.
phase_1_ds = concatenate_datasets([phase_1_ds, precomputed_skills])
phase_1_ds.to_json(f'{output_dir}/phase_1_ds.jsonl', orient='records', lines=True)
```
