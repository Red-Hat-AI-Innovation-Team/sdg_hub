# Synthetic Data Generation for Knowledge Tuning

## What is Knowledge Tuning?

**Knowledge tuning** is the process of adapting a large language model (LLM) to new factual content by training it on specific documents. The goal is to enable the model to **recall and reason over document-grounded information** when performing downstream tasks such as:

* Question Answering
* Summarization
* Entity Extraction
* Other document-specific reasoning tasks

This adaptation can be used:

* As a **standalone fine-tuned model**, or
* As part of a **Retrieval-Augmented Generation (RAG)** pipeline to enhance factual accuracy and contextuality.

---

### Setup Instructions

#### Install sdg-hub

```bash 
pip install sdg-hub==0.1.0a4
```

#### Install with optional dependencies

If you want to use the vLLM server, you can install it with the following command:

```bash 
pip install sdg-hub[vllm] 
```

In order to use docling, you need to install it with the following command:

```bash
pip install sdg-hub[examples]
```

### Serving the Teacher Model

#### vLLM Server

Launch the vLLM server with the following command:

```bash
vllm serve meta-llama/Llama-3.3-70B-Instruct --tensor-parallel-size 4
```

## Repository Structure

This repository demonstrates how to generate synthetic data for knowledge tuning using different approaches:

### Examples

1. [`instructlab/`](instructlab/):
   Implements knowledge tuning using the **InstructLab** pipeline, which supports a two-phase approach:

   * Phase 1: Knowledge tuning via synthetic QAs
   * Phase 2: Instruction tuning to generalize reasoning skills

2. [`knowledge_tuning_with_reasoning_model/`](knowledge_tuning_with_reasoning_model/):
   Uses **Nemotron Super** as the teacher model to generate reasoning-focused synthetic data grounded in document content. We also show how to edit the knowledge pipeline to introduce new types of summaries

Each example includes:

* Source document processing
* QA generation with a teacher model
* Filtering and validation logic
* Dataset formatting for fine-tuning

3. [`translation_example`](translation_example/):
    Implements a translation block to translate article into a target language for generating knowledge QA. The example scripts show how to translate Kannada Wikipedia article into English and generate synthetic QA to train a model.
---

## Data Post-Processing

Once synthetic QA data is generated, you’ll need to prepare it for training:

### Key Practices

* Append source document content to the generated QA to improve memorization and coverage.
* During training, backpropagate on both the **prompt** (document + question) and the **response** (answer).
* For `instructlab.training`, you can use the `unmask` field to enable pretraining-style loss computation over the full prompt-response.

### Creating QA dataset

* You can use below function to transform the generated dataset into Prompt + Response pair for training in messages format.
* You can control various parameters like appending document to question, adding document outline to document etc.
```python
from knowledge_utils import generate_knowledge_qa_dataset

knowl_train = generate_knowledge_qa_dataset(
    generated_dataset=generated_data,
    keep_context_separate=False,
    keep_document_outline=True,
    keep_columns=['document', 'document_outline', 'raw_document']
)
```
* `keep_context_separate=False`: Includes the document in the prompt
* `keep_document_outline=True`: Adds structure to the prompt using outline
* `keep_columns`: Retains metadata for record-keeping (not used in training)


### Workflow: InstructLab (Knowledge + RAFT)
You can find steps for data post-processing [here](instructlab/README.md#data-post-processing)

### Workflow: Fine-tuning Instruct Model

* You can simply take the generated data and continue instruction tuning an existing instruct model (e.g. Qwen 2.5 8B instruct, LLama 3.3 8B/70B etc.)
* Simply follow [Creating QA dataset](#creating-qa-dataset) section for creating the training data.
* Note: The model might suffer catastrophic forgetting and might need a replay buffer of instruction data. Or you might need to explore alternate methods like Parameter efficient fine-tuning.


