export HUGGINGFACE_HUB_CACHE="/new_data/hf_cache"   
export HF_DATASETS_CACHE="/dev/shm/hf"
export HF_HOME="/new_data/hf_cache"
export HF_MODEL_CACHE="/new_data/hf_cache"

SCRIPT_PATH="/home/lab/abhi/training/src/instructlab/training"
MASTER_ADDR="localhost"
MASTER_PORT=23456
WORLD_SIZE=1
RANK=0

BASE_MODEL="meta-llama/Meta-Llama-3-8B-Instruct"

while [[ $# -gt 0 ]]; do
    case $1 in
        --data_path) data_path="$2"; shift 2 ;;
        --parent_exp_dir) parent_exp_dir="$2"; shift 2 ;;
        --exp_dir) exp_dir="$2"; shift 2 ;;
    esac
done
PARENT_EXP_DIR=/new_data/experiments_rh/$parent_exp_dir
EXP_DIR=$exp_dir
mkdir -p $PARENT_EXP_DIR
dataset_size=$(wc -l $data_path | awk '{print $1}')
echo "Dataset size: $dataset_size"
EXP_DIR="${PARENT_EXP_DIR}/${EXP_DIR}"
mkdir -p $EXP_DIR
echo "Experiment directory: $EXP_DIR"

python $SCRIPT_PATH/data_process.py --logging_level INFO \
  --data_path $data_path \
  --chat-tmpl-path "/home/lab/abhi/training/src/instructlab/training/chat_templates/llama_3_8b_tmp.py" \
  --num_cpu_procs=64 \
  --data_output_path "/dev/shm" \
  --max_seq_len "5000" \
  --model_name_or_path $BASE_MODEL | tee $PARENT_EXP_DIR/data_process.log


  
torchrun --nnodes=$WORLD_SIZE --node_rank=$RANK \
    --nproc_per_node=8 --rdzv_id=101 \
    --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" $SCRIPT_PATH/main_ds.py \
    --model_name_or_path=$BASE_MODEL \
    --data_path="/dev/shm/data.jsonl" \
    --chat-tmpl-path "/home/lab/abhi/training/src/instructlab/training/chat_templates/llama_3_8b_tmp.py" \
    --output_dir="${EXP_DIR}" \
    --num_epochs=7 \
    --learning_rate=5e-06 \
    --lr_scheduler="constant_with_warmup" \
    --num_warmup_steps=25 \
    --effective_batch_size=128 \
    --save_samples=$dataset_size \
    --log_level="INFO" \
    --last_step=0 \
    --fsdp_sharding_strategy="SHARD_GRAD_OP" \
    --distributed_training_framework="deepspeed" \
    --lora_r=0 \
    --lora_alpha=32 \
    --lora_dropout=0.1 \
    --NEFTune_alpha 0 \
    --max_batch_len 39000 \
    --seed=42 | tee $EXP_DIR/$RANK.log
