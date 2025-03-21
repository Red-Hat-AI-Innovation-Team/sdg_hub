# Get dataset size and save into variable
dataset_size=$(wc -l /new_data/knowledge_rh/quality/base_datasets/quality_seed_example_chunked.jsonl | awk '{print $1}')
number_of_processes=8
port=8000
for i in {0..7}; do
    dataset_start_index=$((i * dataset_size / number_of_processes))
    dataset_end_index=$((dataset_start_index + dataset_size / number_of_processes))
    python scripts/generate.py --ds_path  /new_data/knowledge_rh/quality/base_datasets/quality_seed_example_chunked.jsonl \
        --bs 2 --num_workers 30 \
        --save_path /new_data/knowledge_rh/quality/knowledge1.25_extractive_phi4/gen.jsonl \
        --flow /home/lab/abhi/SDG-Research-Upstream/src/instructlab/sdg/flows/generation/knowledge/synth_knowledge1.5_phi4.yaml \
        --endpoint http://localhost:$port/v1 \
        --checkpoint_dir /new_data/knowledge_rh/quality/knowledge1.25_extractive_phi4/data_checkpoints \
        --save_freq 1000 \
        --dataset_start_index $dataset_start_index \
        --dataset_end_index $dataset_end_index > run_sdg_$i.log 2>&1 &
    echo "Starting process $i with dataset from $dataset_start_index to $dataset_end_index on port $port"
    port=$((port + 1))
done