# InstructLab Skills Synthetic Data Generation

![InstructLab Banner](../../../assets/imgs/instructlab-banner.png)

The provided notebooks demonstrates how to customize language models by generating training data for specific skills, following the methodology outlined in the LAB (Large-scale Alignment for Chatbots) framework [[paper link](https://arxiv.org/pdf/2403.01081)].

### Customizing Model Behavior

The LAB framework enables us to shape how a model responds to various tasks by training it on carefully crafted examples. Want your model to write emails in your company's tone? Need it to follow specific formatting guidelines? This customization is achieved through what the paper defines as compositional skills.

Compositional skills are tasks that combine different abilities to handle complex queries. For example, if you want your model to write company emails about quarterly performance, it needs to:
- Understand financial concepts
- Perform basic arithmetic
- Write in your preferred communication style
- Follow your organization's email format

### Demo Overview

This notebook will show you how to:
1. Set up a teacher model for generating training data
2. Create examples that reflect your preferred style and approach
3. Generate Synthetic Data
4. Validate that the generated data matches your requirements

The end goal is to create training data that will help align the model with your specific needs, whether that's matching your company's communication style, following particular protocols, or handling specialized tasks in your preferred way.

### Instructlab Grounded Skills Generation Pipeline 
InstructLab uses a multi-step process of generation and evaluation to generate synthetic data. For grounded skills it looks like this: 

<table>
<tr>
  <td>
    <img src="../../../assets/imgs/IL_skills_pipeline.png" alt="Skills Pipeline" width="250">
  </td>
  <td>
    <ul>
      <li>
        <strong>Context Generation (<code>gen_contexts</code>)</strong><br>
        Generates diverse, relevant contexts for the skill<br>
        Produces 10 unique contexts per run<br><br>
      </li>
      <li>
        <strong>Question Generation & Validation</strong><br>
        <code>gen_grounded_questions</code>: Creates 3 questions per context<br>
        <code>eval_grounded_questions</code>: Evaluates question quality<br>
        <code>filter_grounded_questions</code>: Keeps only perfect scores (1.0)<br><br>
      </li>
      <li>
        <strong>Response Generation & Quality Control</strong><br>
        <code>gen_grounded_responses</code>: Generates appropriate responses<br>
        <code>evaluate_grounded_qa_pair</code>: Scores Q&A pair quality<br>
        <code>filter_grounded_qa_pair</code>: Retains high-quality pairs (score ≥ 2.0)<br><br>
      </li>
      <li>
        <strong>Final Processing</strong><br>
        <code>combine_question_and_context</code>: Merges context with questions for complete examples<br><br>
      </li>
    </ul>
  </td>
</tr>
</table>

### Providing the Seed Data

TODO: Add details on how to provide the seed data

### Setup Instructions

#### Install sdg-hub

```bash 
pip install sdg-hub==0.1.0a3
```

#### Install vLLM

```bash 
pip install vllm
```

### Serving the Teacher Model

#### vLLM Server

Launch the vLLM server with the following command:
```bash
vllm serve mistralai/Mixtral-8x7B-Instruct-v0.1 --tensor-parallel-size 2
```

This will use 2 GPUs for inference. You can adjust the number of GPUs by changing the `--tensor-parallel-size` argument.

For instance, if your model needs 4 GPUs, you can run:

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
export INFERENCE_MODEL=mistralai/Mixtral-8x7B-Instruct-v0.1
llama stack build --template remote-vllm
```

The server will start at: `http://localhost:8321`

You can use the CLI to verify the setup:

```bash
llama-stack-client   --endpoint http://localhost:8321   inference chat-completion   --model-id $INFERENCE_MODEL   --message "write a haiku about language models"
```