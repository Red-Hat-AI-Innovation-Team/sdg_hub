export HUGGINGFACE_HUB_CACHE="/new_data/hf_cache"   
export HF_DATASETS_CACHE="/dev/shm/hf"
export HF_HOME="/new_data/hf_cache"
export HF_MODEL_CACHE="/new_data/hf_cache"

port=8000
for i in {0..7}; do
    CUDA_VISIBLE_DEVICES=$i python -m vllm.entrypoints.openai.api_server \
        --model microsoft/phi-4 \
        --dtype float16 \
        --tensor-parallel-size 1 \
        --port $port \
        --trust-remote-code > run_phi4_teacher_model_$i.log 2>&1 &
    port=$((port + 1))
done