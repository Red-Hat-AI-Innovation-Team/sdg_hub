# InstructLab: Synthetic Data Generation for Knowledge Tuning
![InstructLab Banner](../../../assets/imgs/instructlab-banner.png)

The provided notebooks show a sdg_hub pipeline for generating high-quality synthetic data from your documents. By following the methodology of the **LAB (Large-scale Alignment for Chatbots)** framework, as detailed in our [research paper](https://arxiv.org/pdf/2403.01081), you can effectively tune language models to master the knowledge contained within your specific-domain documentation.

## How It Works: A Three-Phase Pipeline

Our data generation process is designed to operate in three distinct phases, ensuring the creation of a robust and reliable dataset for model training.

### 1\. Document Summarization

To kickstart the process, we generate three unique summaries of your source documents. This multi-faceted approach helps the model to thoroughly memorize and recall the key information. The summaries include:

  * **Detailed Summaries:** Comprehensive overviews of the content.
  * **Extractive Summaries:** Key sentences and passages pulled directly from the text.
  * **Atomic Facts:** A list of the most critical, standalone pieces of information.

### 2\. Synthetic Q\&A Generation

Next, our pipeline leverages user-provided "seed examples"—sample questions and answers—to generate a wealth of synthetic Q\&A pairs. These new pairs are contextually grounded in the summarized documents, effectively scaling up your initial examples into a diverse training dataset.

### 3\. Quality Control

To ensure the integrity of our generated data, we employ a quality-checking phase. Using a "teacher" model, we perform a faithfulness evaluation by:

1.  Providing the model with a generated answer and the original source document.
2.  Tasking the model to extract every claim made in the answer.
3.  Verifying that each claim is factually supported by the provided document.

This process filters out inaccuracies and ensures that only high-quality, faithful Q\&A pairs make it into the final dataset.

## Getting Started

To begin using the pipeline, simply install the `sdg_hub` library. From there, you can instantiate and run the synthetic data generation process with the following code:

```python
from sdg_hub.flow import Flow
from sdg_hub.sdg import SDG

# Load the knowledge generation pipeline from the YAML file
knowledge_agentic_pipeline = "../../../src/instructlab/sdg/flows/generation/knowledge/synth_knowledge1.5.yaml"
flow = Flow(openai_client).get_flow_from_file(knowledge_agentic_pipeline)

# Initialize the Synthetic Data Generator
generator = SDG(
    flows=[flow],
    num_workers=1,
    batch_size=1,
    save_freq=1000,
)
```

## InstructLab Training Methodology

Our training process is structured to build upon a pre-trained model, systematically enhancing its skills and knowledge.

1.  **Foundation Training:** We begin by training a pre-trained model on foundational skills such as logic, coding, and math. The instruction data in this phase features short, direct responses.
2.  **Foundation Knowledge:** Next, we expand the model's general knowledge base, by training it on general textbooks and benchmarking it against MMLU. The result of these first two stages is what we term the **starter model**.

This **starter model** then serves as the base for our specialized, two-phase knowledge tuning:

  * **Phase 1: Knowledge Tuning:** We do pre-training style training on the document generated data by our pipeline. This allows the model to internalize the new knowledge and be able to recall and answer questions based on it.
  * **Phase 2: Skills Tuning:** Building on the previous phase, we do instruction tuning on general skills (combination of instruction tuning dataset and skills generated with sdg_hub). To prevent the model from forgetting the newly acquired knowledge, we mix in data from the previous stage. We also incorporate [RAFT-style](https://openreview.net/forum?id=rzQGHXNReU) data to enhance the model's robustness for RAG on the target documents.

## Data Post-Processing

After generating your data, use the provided utility functions to prepare it for the two-phase training process. All helper functions are located in `examples/knowledge_tuning/knowledge_utils.py`.

### 1\. Knowledge Dataset (for Phase 1)

This dataset is used for the initial knowledge-tuning phase. You can also merge datasets from multiple documents to train on a set of documents simultaneously.

This function also creates a summarization dataset that formats the generated summaries as task: document + instruction -> document summary.

```python
from knowledge_utils import create_knowledge_pretraining_ds

# Create the dataset for knowledge pre-training
knowledge_data = create_knowledge_pretraining_ds(generated_dataset=generated_data)
```

### 2\. Skills Dataset (for Phase 2)

This dataset combines the knowledge-specific data with RAFT-style examples for the second phase of tuning. It can also be mixed with general instruction-tuning data to grant the model broad instruction-following abilities while retaining the specialized knowledge.

```python
from knowledge_utils import create_knowledge_regular_ds
from datasets import concatenate_datasets

# Create the dataset with RAFT and summary data
raft_and_summary_data = create_knowledge_regular_ds(generated_dataset=generated_data)

# Create the core knowledge dataset. 
# add_auxiliary_dataset parameter controls wheter to add the summarization dataset that was mentioned above in the returned dataset
knowledge_data = create_knowledge_pretraining_ds(generated_dataset=generated_data, add_auxiliary_dataset=False)

# Combine the datasets for the skills tuning phase
knowledge_skills_data = concatenate_datasets([raft_and_summary_data, knowledge_data])
```

---

## Generation Statistics

Default generation parameters (based on `llama-3.3-70B`) are defined in:
[`synth_knowledge1.5.yaml`](../../src/sdg_hub/flows/generation/knowledge/synth_knowledge1.5.yaml)

* The pipeline converts each input document into **3 summaries** 
* Outputs vary based on teacher model and generation parameters (e.g. `temperature`, `top_p`, `top_k`) and can be entered in the `gen_kwargs` section of the flow.
* Generation currently uses temperature=0.0 and is deterministic.
