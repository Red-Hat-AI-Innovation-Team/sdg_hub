# %% [markdown]
"""
# Hello World - SDG Hub Simple Example

This notebook demonstrates the simplest way to get started with SDG Hub
using the basic knowledge generation flow.
"""

# %%
from sdg_hub.flow_runner import run_flow

# %% [markdown]
"""
## Basic Knowledge Generation Flow

This example shows how to run a simple knowledge generation flow using SDG Hub.
The flow will generate document-grounded questions and answers for factual memorization.
"""

# %%
# Run a basic knowledge generation flow
run_flow(
    ds_path="my_data.jsonl",
    save_path="output.jsonl", 
    endpoint="http://0.0.0.0:8000/v1",
    flow_path="flows/generation/knowledge/synth_knowledge.yaml"
)

# %% [markdown]
"""
## Advanced Configuration Example

You can also configure additional parameters for more control over the flow execution:
- `checkpoint_dir`: Directory to save checkpoints for resuming interrupted flows
- `batch_size`: Number of samples to process in each batch
- `num_workers`: Number of parallel workers for processing
- `save_freq`: How often to save intermediate results
"""

# %%
# Example with advanced configuration
run_flow(
    ds_path="path/to/dataset.jsonl",
    save_path="path/to/output.jsonl",
    endpoint="http://0.0.0.0:8000/v1",
    flow_path="path/to/flow.yaml",
    checkpoint_dir="path/to/checkpoints",
    batch_size=8,
    num_workers=32,
    save_freq=2,
)

# %% [markdown]
"""
## Available Built-in Flows

SDG Hub comes with several pre-configured flows:

### Knowledge Flows
- `synth_knowledge.yaml`: Document-grounded Q&A for factual memorization
- `synth_knowledge1.5.yaml`: Improved version with intermediate representations

### Skills Flows  
- `synth_skills.yaml`: Freeform skills Q&A generation
- `synth_grounded_skills.yaml`: Domain-specific skill generation
- `improve_responses.yaml`: Planning and critique-based response refinement

All flows are located in the `src/sdg_hub/flows` directory.
"""

# %% [markdown]
"""
## Next Steps

Try running this example with your own data and LLM endpoint to generate 
synthetic question-answer pairs for training or evaluation purposes.
"""