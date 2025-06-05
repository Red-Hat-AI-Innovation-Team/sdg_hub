# InstructLab Skills Synthetic Data Generation

![InstructLab Banner](../../../assets/imgs/instructlab-banner.png)

The provided notebooks demonstrate how to customize language models by generating training data for specific skills, following the methodology outlined in the LAB (Large-scale Alignment for Chatbots) framework [[paper link](https://arxiv.org/pdf/2403.01081)].

### Customizing Model Behavior

The LAB framework enables us to shape how a model responds to various tasks by training it on carefully crafted examples. Want your model to write emails in your company's tone? Need it to follow specific formatting guidelines? This customization is achieved through what the paper defines as compositional skills.

Compositional skills are tasks that combine different abilities to handle complex queries. For example, if you want your model to write company emails about quarterly performance, it needs to:
- Understand financial concepts
- Perform basic arithmetic
- Write in your preferred communication style
- Follow your organization's email format

### Examples Overview

This directory contains four example notebooks that demonstrate different skills you can teach to language models:

1. [**Unstructured to Structured**](unstructured_to_structured.ipynb): Shows how to convert unstructured text data into structured formats, making it easier to process and analyze.

2. [**Table Manipulation**](table_manipulation.ipynb): Teaches models to understand and manipulate tabular data, including operations like filtering, sorting, and data transformation.

3. [**Structured Summary**](structured_summary.ipynb): Demonstrates how to generate structured summaries from text, converting free-form content into organized, formatted outputs.

4. [**Annotation Classification**](annotation_classification.ipynb): Shows how to train a model to classify and categorize text annotations, useful for data labeling and organization tasks.



### Providing the Seed Data

When teaching a language model a new skill, carefully crafted seed examples are the foundation. Seed examples show the model what good behavior looks like by pairing inputs with ideal outputs, allowing the model to learn patterns, structure, reasoning, and formatting that generalize beyond the examples themselves.

A strong seed example, regardless of domain, should:

✅ Clearly define the task context and expected behavior

✅ Provide a realistic, natural input that mimics what users or systems would actually produce

✅ Include a high-quality output that fully satisfies the task requirements—accurate, complete, and formatted correctly

✅ Minimize ambiguity: avoid examples where multiple interpretations are possible without explanation

✅ Reflect diverse edge cases: cover a variety of structures, phrasings, or difficulty levels to help the model generalize



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

This will host the model endpoint with default address being `http://localhost:8000`

> ⚠️ Make sure your system has sufficient GPU memory.  
> 🔧 Adjust `--tensor-parallel-size` based on available GPUs.  
> ⏱️ First-time model loading may take several minutes.

#### Optional: Using a Llama Stack Inference Server

Set Up Llama Stack (OpenAI-Compatible Interface)

1. Clone and install Llama Stack (OpenAI-compatible branch)
```bash
git clone https://github.com/bbrowning/llama-stack.git
cd llama-stack
git checkout openai_server_compat
pip install -e .
```

2. Install the Python client
```bash
pip install llama-stack-client
```

3. Launch the Llama Stack Server (connected to vLLM)
```bash
export INFERENCE_MODEL=meta-llama/Llama-3.3-70B-Instruct
llama stack build --template remote-vllm
```

The server will start at: `http://localhost:8321`

You can use the CLI to verify the setup:

```bash
llama-stack-client   --endpoint http://localhost:8321   inference chat-completion   --model-id $INFERENCE_MODEL   --message "write a haiku about language models"
```