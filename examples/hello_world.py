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
## Next Steps

Move next to `hello_world_advanced.ipynb` for more custom usages.
"""

# %% [markdown]
#
