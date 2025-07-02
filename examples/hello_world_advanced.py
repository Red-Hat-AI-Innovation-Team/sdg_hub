# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     notebook_metadata_filter: all
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   language_info:
#     name: python
# ---

# %%
"""
# Hello World - SDG Hub Simple Example

This notebook demonstrates the simplest way to get started with SDG Hub
using the advanced knowledge generation flow.
"""

# %%
from sdg_hub.flow_runner import run_flow

# %%
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

# %%
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
