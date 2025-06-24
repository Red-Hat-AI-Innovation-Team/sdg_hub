## Notebook Summary

This notebook demonstrates how to the training dataset to train a
student model. This notebook continues where left before. Recall in the
reasoning notebook we saw how to modify current knowledge flow, swap a
teacher model with reasoning model, and generate reasoning synthetic
data. This notebook will focus on **data mixing and replay buffer
strategies** to enhance the diversity and quality of training data. The
notebook shows how to:

-   Show you how to create training mix using generated data and
    existing released instruct data
-   How instructlab data mixing for knowledge works



### Prepare Nemotron Replay Buffer



``` python
from datasets import load_dataset, concatenate_datasets

nemotron_ds = load_dataset("nvidia/Llama-Nemotron-Post-Training-Dataset-v1", "SFT")
nemotron_ds = nemotron_ds.filter(lambda x: x['used_in_training'] == 'yes')
nemotron_ds = nemotron_ds.map(lambda x: {'question': x['input'][x['input'].find("user<|end_header_id|>") + len("user<|end_header_id|>") : x['input'].find("<|eot_id|><|start_header_id|>assistant")].strip()})
nemotron_ds = concatenate_datasets(nemotron_ds.values())
nemotron_ds = nemotron_ds.shuffle(seed=894375).select(range(200000))
nemotron_ds = nemotron_ds.add_column('unmask', [False]*nemotron_ds.num_rows)
nemotron_ds = nemotron_ds.map(lambda x: {'messages': [{'role': 'system', 'content': 'detailed thinking on'}, {'role': 'user', 'content': x['question']}, {'role': 'assistant', 'content': x['output']}]})
nemotron_ds = nemotron_ds.remove_columns(['input', 'output', 'category', 'license', 'reasoning', 'generator', 'used_in_training', 'question'])
nemotron_ds.to_json("nemotron_replay_buffer_data.jsonl", orient='records', lines=True)
```



### Create functions for data mixng with conversation templates



This section uses utility functions for creating training data that
combines generated reasoning examples. The functions handle tasks like:

-   Converting documents and Q&A pairs into chat format
-   Adding system prompts for controlling model behavior
-   Mixing in auxiliary data like summaries

These functions are similar to instructlab.sdg\'s data mixing functions

The core functions are defined in:

-   `sdg_hub/examples/reasoning_knowledge_generation/utils.py`
-   `sdg_hub/examples/knowledge_tuning/knowledge_utils.py`



``` python
from datasets import Dataset
import json
import uuid
import os
import sdg_hub
import yaml
import random

def generate_knowledge_qa_dataset(
    generated_dataset: Dataset, keep_context_separate=False, keep_document_outline=False
):
    """
    Generate a knowledge QA dataset from the input dataset by transforming document/question/response pairs into a chat format.
    
    Args:
        generated_dataset (Dataset): Input dataset containing documents, questions and responses
        keep_context_separate (bool): If True, keeps context separate from the messages. If False, includes context in user message
        keep_document_outline (bool): If True, includes document outline in user message when context is not separate
        
    Returns:
        Dataset: Transformed dataset with chat messages format
    """
    def __create_qa_row(rec):
        context = rec["document"]
        instruction = rec["question"]
        response = rec["response"]
        metadata = {
            "sdg_document": rec["document"],
            "domain": rec["domain"],
            "dataset": "document_knowledge_qa",
        }
        if "raw_document" in rec and "dataset_type" in rec:
            metadata.update(
                {
                    "raw_document": rec["raw_document"],
                    "dataset_type": rec["dataset_type"],
                }
            )
        metadata = json.dumps(metadata)
        if keep_context_separate:
            messages = [
                {"role": "user", "content": f"{instruction}"},
                {"role": "assistant", "content": response},
            ]
            return {
                "messages": messages,
                "metadata": metadata,
                "id": str(uuid.uuid4()),
                "context": context,
            }
        else:
            if keep_document_outline:
                messages = [
                    {
                        "role": "user",
                        "content": f"{rec['document_outline']}\n{context}\n\n{instruction}",
                    },
                    {"role": "assistant", "content": response},
                ]
            else:
                messages = [
                    {"role": "user", "content": f"{context}\n\n{instruction}"},
                    {"role": "assistant", "content": response},
                ]
            return {"messages": messages, "metadata": metadata, "id": str(uuid.uuid4())}

    knowledge_ds = generated_dataset.map(
        __create_qa_row, remove_columns=generated_dataset.column_names
    )
    return knowledge_ds


def _conv_pretrain(rec, tokenizer):
    """
    Convert messages to pretraining format or set unmask flag based on tokenizer.
    
    Args:
        rec (dict): Record containing messages
        tokenizer: Tokenizer object, if None uses pretraining format
        
    Returns:
        dict: Modified record
    """
    if tokenizer is not None:
        rec['unmask'] = True
        return rec
    rec["messages"] = [
        {
            "role": "pretraining",
            "content": f"<|user|>\n{rec['messages'][0]['content']}\n<|assistant|>\n{rec['messages'][1]['content']}",
        }
    ]
    return rec

def create_auxiliary_dataset(generated_dataset: Dataset):
    """
    Create auxiliary dataset from non-base documents using predefined instructions.
    
    Args:
        generated_dataset (Dataset): Input dataset containing documents and metadata
        
    Returns:
        Dataset: Auxiliary dataset with chat messages, or None if requirements not met
    """
    if "dataset_type" not in generated_dataset.column_names:
        return None

    aux_inst_path = os.path.join(
        os.path.dirname(sdg_hub.__file__),
        "configs/knowledge/auxilary_instructions.yaml",
    )
    print(aux_inst_path)

    if os.path.isfile(aux_inst_path):
        with open(aux_inst_path, "r", encoding="utf-8") as fp:
            auxiliary_inst = yaml.safe_load(fp)
    else:
        print(f"auxiliary instructions file not found at {aux_inst_path}")
        return None
    
    # Filter and process auxiliary documents
    auxiliary_ds = generated_dataset.filter(
        lambda x: x["dataset_type"] != "base_document"
    )
    unique_document_auxiliary = auxiliary_ds.to_pandas().drop_duplicates(
        subset=["document"]
    )
    unique_document_auxiliary = Dataset.from_pandas(unique_document_auxiliary)
    unique_document_auxiliary = unique_document_auxiliary.remove_columns(
        [
            col
            for col in unique_document_auxiliary.column_names
            if col
            not in [
                "raw_document",
                "document_outline",
                "domain",
                "dataset_type",
                "document",
            ]
        ]
    )
    unique_document_auxiliary = unique_document_auxiliary.rename_columns(
        {"raw_document": "context", "document": "response"}
    )

    def __create_auxiliary_ds(rec):
        instruction = random.choice(auxiliary_inst[rec["dataset_type"]])
        messages = [
            {"role": "user", "content": f"{rec['context']}\n\n{instruction}"},
            {"role": "assistant", "content": rec["response"]},
        ]
        metadata = json.dumps(
            {
                "dataset_type": rec["dataset_type"],
                "raw_document": rec["context"],
                "dataset": f"document_{rec['dataset_type']}",
                "domain": rec["domain"],
            }
        )
        return {"messages": messages, "metadata": metadata, "id": str(uuid.uuid4())}

    unique_document_auxiliary = unique_document_auxiliary.map(
        __create_auxiliary_ds, remove_columns=unique_document_auxiliary.column_names
    )
    return unique_document_auxiliary

    
def create_training_mix(ds, tokenizer, thinking="on", create_summary=True, nemotron_format=True, keep_context_separate=False, no_pretrain=False, keep_document_outline=False):
    """
    Create a mixed training dataset combining knowledge QA and optional summary data.
    
    Args:
        ds (Dataset): Input dataset
        tokenizer: Tokenizer for pretraining format
        thinking (str): Thinking mode for system message ("on"/"off")
        create_summary (bool): Whether to include summary dataset
        nemotron_format (bool): Whether to add system messages in nemotron format
        keep_context_separate (bool): Whether to keep context separate in knowledge QA
        no_pretrain (bool): Skip pretraining format conversion if True
        keep_document_outline (bool): Include document outline in messages
        
    Returns:
        Dataset: Combined training dataset
    """
    # Generate knowledge QA dataset
    knowl_train = generate_knowledge_qa_dataset(ds, keep_context_separate=keep_context_separate, keep_document_outline=keep_document_outline)
    
    # Apply pretraining format if needed
    if no_pretrain:
        knowl_train_pretrain = knowl_train
    else:
        knowl_train_pretrain = knowl_train.map(_conv_pretrain, fn_kwargs={"tokenizer": tokenizer}, num_proc=10)
    
    # Add system messages for nemotron format
    if nemotron_format:
        knowl_train_pretrain = knowl_train_pretrain.map(lambda x: {'messages': [{'content': f'detailed thinking {thinking}', 'role': 'system'}] + x['messages']})
    
    # Add summary dataset if requested
    if create_summary:
        summary_ds = create_auxiliary_dataset(ds)
        if no_pretrain and summary_ds:
            summary_ds_pretrain = summary_ds
        else:
            summary_ds_pretrain = summary_ds.map(_conv_pretrain, fn_kwargs={"tokenizer": tokenizer}, num_proc=10)
        if nemotron_format:
            summary_ds_pretrain = summary_ds_pretrain.map(lambda x: {'messages': [{'content': 'detailed thinking off', 'role': 'system'}] + x['messages']})
        return concatenate_datasets([knowl_train_pretrain, summary_ds_pretrain])
    else:
        return knowl_train_pretrain
```



### Create quality training mix of: reasoning dataset, with non-reasoning dataset, nemotron replay buffer



``` python
### For this tutorial, we will use the following document uids from the quality dataset:
DOC_UIDS = [
    ' Defining Decay Down by David Plotz',
    ' Fight Clubbed by David Plotz',
    ' I, Antichrist? by Jeffrey Goldberg',
    " It's Time To Keelhaul U-Haul! by Jeffrey Goldberg",
    " My Father's Estate by Ben Stein",
    '"Phone Me in Central Park" by McConnell, James V.',
    '...After a Few Words... by Garrett, Randall', 
    '...And It Comes Out Here by Del Rey, Lester',
    'A Coffin for Jacob by Ludwig, Edward W.',
    'A Fall of Glass by Lee, Stanley R.',
    'A Filbert Is a Nut by Raphael, Rick',
    'A Gift from Earth by Banister, Manly',
    'A Gleeb for Earth by Schafhauser, Charles',
    'A Good Year for the Roses? by David Edelstein',
    'A Pail of Air by Leiber, Fritz',
    'A Planet Named Joe by Hunter, Evan',
    "AI: what's the worst that could happen? by Harry Armstrong",
    'Accidental Death by Baily, Peter',
    'All Day September by Kuykendall, Roger',
    'Ambition by Bade, William L.',
    'And Then the Town Took Off by Wilson, Richard',
    'Atom Mystery [Young Atom Detective] by Coombs, Charles Ira',
    'Beach Scene by King, Marshall',
    'Big Ancestor by Wallace, F. L. (Floyd L.)',
    'Birds of a Feather by Silverberg, Robert',
    'Bodyguard by Gold, H. L. (Horace Leonard)'
]
```



``` python
from datasets import load_dataset
from transformers import AutoTokenizer
 
# Load tokenizer for pre-training formatting
tokenizer = AutoTokenizer.from_pretrained("nvidia/Llama-3.1-Nemotron-Nano-8B-v1")

# Load non-reasoning dataset from nemotron super 49b
nemotron_non_reasoning_ds = load_dataset("json", data_dir="data/knowledge/quality/knowledge_nemotron/", split="train")
nemotron_non_reasoning_ds = nemotron_non_reasoning_ds.filter(lambda x: x['score'] == '2' and x['judgment'] == 'YES')
nemotron_non_reasoning_ds = nemotron_non_reasoning_ds.filter(lambda x: x['document_outline'] in DOC_UIDS)
print(nemotron_non_reasoning_ds)

# Load reasoning dataset from nemotron super 49b
nemotron_reasoning_ds = load_dataset("json", data_dir="data/knowledge/quality/synth_knowledge_reasoning/", split="train")
nemotron_reasoning_ds = nemotron_reasoning_ds.filter(lambda x: x['score'] == '2' and x['judgment'] == 'YES')
nemotron_reasoning_ds = nemotron_reasoning_ds.filter(lambda x: x['document_outline'] in DOC_UIDS)
print(nemotron_reasoning_ds)

# Load nemotron replay buffer. 
# Note: This is a replay buffer we created by sub-sampling nvidia/Llama-Nemotron-Post-Training-Dataset from huggingface.
nemotron_ds_replay_buffer = load_dataset("json", data_files="data/knowledge/quality/training_mix/replay_buffer.jsonl", split="train")


# Create non-reasoning training mix
nemotron_ds_training_mix = create_training_mix(nemotron_non_reasoning_ds, tokenizer, 'off').shuffle(seed=894375)

# Create reasoning training mix
nemotron_reasoning_ds = create_training_mix(nemotron_reasoning_ds, tokenizer, 'on')

# Concatenate reasoning and non-reasoning training mixes
quality_reasoning_ds = concatenate_datasets([nemotron_reasoning_ds, nemotron_ds_training_mix]).remove_columns(['metadata', 'id']) # .select(range(40000))
print(quality_reasoning_ds)

# Concatenate training mix with replay buffer
training_mix = concatenate_datasets([quality_reasoning_ds, nemotron_ds_replay_buffer.shuffle(seed=894375).select(range(len(quality_reasoning_ds)))])

print(training_mix)
training_mix.to_json("data/knowledge/quality/training_mix/quality_knowledge_mix.jsonl", orient='records', lines=True)
```



### Train student model

-   Setup the training by cloning
    `https://github.com/instructlab/training` and following the
    instructions in the README

-   The create `train.py` using below code \`\`\`python import argparse
    from instructlab.training.config import
    TorchrunArgs,TrainingArgs,DistributedBackend,FSDPOptions from
    instructlab.training.main_ds import run_training import os def
    parse_args(): parser =
    argparse.ArgumentParser(description=\'Training script with
    configurable paths\') parser.add_argument(\'\--data_path\',
    type=str, required=True, help=\'Path to the training data file\')
    parser.add_argument(\'\--model_path\', type=str, required=True,
    help=\'Path to the model or model identifier\')
    parser.add_argument(\'\--chat_tmpl_path\', type=str, required=True,
    help=\'Path to the chat template file\')
    parser.add_argument(\'\--exp_dir\', type=str, required=True,
    help=\'Path to the experiment directory\')
    parser.add_argument(\'\--parent_exp_dir\', type=str, required=True,
    help=\'Path to the parent experiment directory\') return
    parser.parse_args()

    def main(): args = parse_args()

          torch_args = TorchrunArgs(
              nproc_per_node=8,
              nnodes=1,
              node_rank=0,
              rdzv_id=123,
              rdzv_endpoint="0.0.0.0:8888",
          )
          output_dir = os.path.join(args.parent_exp_dir, args.exp_dir)
          train_args = TrainingArgs(
              model_path=args.model_path,
              data_path=args.data_path,
              ckpt_output_dir=output_dir,
              data_output_dir="data/processed-data",
              max_seq_len=20000,
              max_batch_len=25000,
              num_epochs=5,
              effective_batch_size=256,
              learning_rate=5e-6,
              warmup_steps=25,
              save_samples=0,
              use_dolomite=False,
              checkpoint_at_epoch = True,
              accelerate_full_state_at_epoch = False,
              process_data=True,
              chat_tmpl_path=args.chat_tmpl_path,
              distributed_backend=DistributedBackend.FSDP,
              fsdp_options=FSDPOptions(cpu_offload_params=False),
          )

          run_training(torch_args=torch_args,train_args=train_args)

    if **name** == \"**main**\": main() \`\`\`

-   Now create bash script with run command

    ``` shell
    python train.py \
    --data_path "quality_knowledge_1.25_nemotron_49b_first_24.jsonl" \
    --model_path "nvidia/Llama-3.1-Nemotron-Nano-8B-v1" \
    --chat_tmpl_path "<chat_template_path>" \
    --exp_dir "nano_customized_thinking_quality_model" \
    --parent_exp_dir "<parent_exp_dir>"
    ```

