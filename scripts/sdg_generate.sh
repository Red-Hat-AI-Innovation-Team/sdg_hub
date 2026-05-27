#!/usr/bin/env bash
# Execute synthetic data generation using a flow.
#
# Usage: sdg_generate.sh <flow> <input-file> [--output FILE] [--sample N]
# Output: JSON with status, row counts, and output file path.
#
# Reads model config from .sdg-hub/config.json (override via
# SDG_HUB_CONFIG env var).
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"

CONFIG_PATH="${SDG_HUB_CONFIG:-.sdg-hub/config.json}"

usage() {
    echo "Usage: sdg_generate.sh [OPTIONS] <flow> <input-file>"
    echo ""
    echo "Options:"
    echo "  --output FILE      Output file path (default: <input>_generated.jsonl)"
    echo "  --sample N         Dry-run with N samples before full generation"
    echo "  --concurrency N    Max parallel LLM requests (default: from config or 5)"
    exit 1
}

die() { echo "ERROR: $1" >&2; exit 1; }

# Parse arguments
FLOW=""
INPUT_FILE=""
OUTPUT_FILE=""
SAMPLE=""
CONCURRENCY=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --output) OUTPUT_FILE="$2"; shift 2 ;;
        --sample) SAMPLE="$2"; shift 2 ;;
        --concurrency) CONCURRENCY="$2"; shift 2 ;;
        --help) usage ;;
        -*)  die "Unknown option: $1" ;;
        *)
            if [ -z "$FLOW" ]; then
                FLOW="$1"
            elif [ -z "$INPUT_FILE" ]; then
                INPUT_FILE="$1"
            else
                die "Unexpected argument: $1"
            fi
            shift
            ;;
    esac
done

[ -z "$FLOW" ] && die "No flow specified. Usage: sdg_generate.sh <flow> <input-file>"
[ -z "$INPUT_FILE" ] && die "No input file specified. Usage: sdg_generate.sh <flow> <input-file>"
[ -f "$INPUT_FILE" ] || die "Input file not found: $INPUT_FILE"

# Read config if available
MODEL=""
API_KEY=""
API_BASE=""
if [ -f "$CONFIG_PATH" ]; then
    MODEL=$(CONFIG_PATH="$CONFIG_PATH" $PYTHON -c "import json,os; print(json.load(open(os.environ['CONFIG_PATH'])).get('model', ''))" 2>/dev/null || echo "")
    API_KEY=$(CONFIG_PATH="$CONFIG_PATH" $PYTHON -c "import json,os; print(json.load(open(os.environ['CONFIG_PATH'])).get('api_key', ''))" 2>/dev/null || echo "")
    API_BASE=$(CONFIG_PATH="$CONFIG_PATH" $PYTHON -c "import json,os; print(json.load(open(os.environ['CONFIG_PATH'])).get('api_base', ''))" 2>/dev/null || echo "")
    [ -z "$CONCURRENCY" ] && CONCURRENCY=$(CONFIG_PATH="$CONFIG_PATH" $PYTHON -c "import json,os; print(json.load(open(os.environ['CONFIG_PATH'])).get('max_concurrency', 5))" 2>/dev/null || echo "5")
fi
[ -z "$CONCURRENCY" ] && CONCURRENCY="5"

# Default output file
if [ -z "$OUTPUT_FILE" ]; then
    BASENAME="${INPUT_FILE%.*}"
    OUTPUT_FILE="${BASENAME}_generated.jsonl"
fi

# Execute generation
SDG_FLOW="$FLOW" SDG_INPUT="$INPUT_FILE" SDG_OUTPUT="$OUTPUT_FILE" \
SDG_SAMPLE="${SAMPLE:-0}" SDG_CONCURRENCY="$CONCURRENCY" \
SDG_MODEL="$MODEL" SDG_API_KEY="$API_KEY" SDG_API_BASE="$API_BASE" \
SDG_CONFIG="$CONFIG_PATH" \
$PYTHON -c "
import json, os, sys

flow_name = os.environ['SDG_FLOW']
input_file = os.environ['SDG_INPUT']
output_file = os.environ['SDG_OUTPUT']
sample_n = int(os.environ['SDG_SAMPLE'])
model = os.environ.get('SDG_MODEL', '')
api_key = os.environ.get('SDG_API_KEY', '')

from sdg_hub import Flow, FlowRegistry
from datasets import Dataset

# Load flow
if flow_name.endswith('.yaml') or flow_name.endswith('.yml'):
    flow = Flow.from_yaml(flow_name)
else:
    flow = FlowRegistry.get_flow(flow_name)

# Load dataset
ds = Dataset.from_json(input_file)
print(f'Loaded {len(ds)} rows from {input_file}')

# Dry run if requested
if sample_n > 0:
    sample_ds = ds.select(range(min(sample_n, len(ds))))
    print(f'Running dry-run with {len(sample_ds)} samples...')
    result = flow.generate(sample_ds)
    print(f'Dry-run produced {len(result)} rows')

# Full generation
print(f'Running full generation on {len(ds)} rows...')
result = flow.generate(ds)

# Save output
result.to_json(output_file, orient='records', lines=True)
print(json.dumps({
    'status': 'complete',
    'flow': flow_name,
    'input_rows': len(ds),
    'output_rows': len(result),
    'output_file': output_file
}))
"
