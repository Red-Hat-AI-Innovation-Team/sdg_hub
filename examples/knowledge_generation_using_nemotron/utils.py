from sdg_hub.utils.parse_and_convert import (
    create_auxiliary_dataset,
    generate_knowledge_qa_dataset
)
from transformers import AutoTokenizer
from datasets import concatenate_datasets

def _conv_pretrain(rec, tokenizer):
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

def create_training_mix(ds, tokenizer, thinking="on", create_summary=True, nemotron_format=True, keep_context_separate=False, no_pretrain=False):
    knowl_train = generate_knowledge_qa_dataset(ds, keep_context_separate=keep_context_separate)
    if no_pretrain:
        knowl_train_pretrain = knowl_train
    else:
        knowl_train_pretrain = knowl_train.map(_conv_pretrain, fn_kwargs={"tokenizer": tokenizer}, num_proc=10)
    if nemotron_format:
        knowl_train_pretrain = knowl_train_pretrain.map(lambda x: {'messages': [{'content': f'detailed thinking {thinking}', 'role': 'system'}] + x['messages']})
    if create_summary:
        summary_ds = create_auxiliary_dataset(ds)
        if no_pretrain and summary_ds:
            summary_ds_pretrain = summary_ds
        else:
            summary_ds_pretrain = summary_ds.map(_conv_pretrain, fn_kwargs={"tokenizer": tokenizer}, num_proc=10)
        if nemotron_format:
            summary_ds = summary_ds.map(lambda x: {'messages': [{'content': 'detailed thinking off', 'role': 'system'}] + x['messages']})
        return concatenate_datasets([knowl_train_pretrain, summary_ds_pretrain])
    else:
        return knowl_train_pretrain