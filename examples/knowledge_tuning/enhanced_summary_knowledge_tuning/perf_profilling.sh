#!/bin/bash

# Performance profiling script for knowledge generation
# Runs knowledge_generation_script.py with different SEED_DATA_SIZE values

# Array of SEED_DATA_SIZE values to test
SIZES=(1 5 10 20 50 100 500 1000)

# Create output directory for logs
OUTPUT_DIR="perf_results"
mkdir -p "$OUTPUT_DIR"

echo "Starting performance profiling..."
echo "Results will be saved to $OUTPUT_DIR/"

# Loop through each size
for SIZE in "${SIZES[@]}"; do
    echo ""
    echo "=========================================="
    echo "Running with SEED_DATA_SIZE=$SIZE"
    echo "=========================================="
    
    # Set the environment variable and run the script
    export SEED_DATA_SIZE=$SIZE
    
    # Run the script and capture output
    LOG_FILE="$OUTPUT_DIR/run_size_${SIZE}.log"
    
    echo "Logging to: $LOG_FILE"
    
    # Run the Python script
    python knowledge_generation_script.py 2>&1 | tee "$LOG_FILE"
    echo "Completed SEED_DATA_SIZE=$SIZE"
done

echo ""
echo "=========================================="
echo "Performance profiling complete!"
echo "Results saved in $OUTPUT_DIR/"
echo "=========================================="
