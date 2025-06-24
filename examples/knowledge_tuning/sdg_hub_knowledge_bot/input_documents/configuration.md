# Configuration Guide

This guide covers configuration options for SDG Hub, including logging, environment variables, and runtime settings.

## Logging Configuration

SDG Hub provides rich, configurable logging to help you monitor and debug flow execution.

### Log Levels

Configure logging verbosity using the `SDG_HUB_LOG_LEVEL` environment variable or programmatically:

| Level | Description | Output |
|-------|-------------|--------|
| `quiet` | No output | Suppresses all flow execution logs |
| `normal` | Basic progress | Shows block execution status (default) |
| `verbose` | Detailed info | Adds rich tables with dataset metrics |
| `debug` | Full debugging | Includes dataset content dumps |

### Environment Variable Configuration

Set the log level globally using the environment variable:

```bash
# Quiet mode - no output
export SDG_HUB_LOG_LEVEL=quiet

# Normal mode - basic progress (default)
export SDG_HUB_LOG_LEVEL=normal

# Verbose mode - progress + dataset info tables
export SDG_HUB_LOG_LEVEL=verbose

# Debug mode - verbose + dataset content dumps
export SDG_HUB_LOG_LEVEL=debug
```

Then run your script normally:

```bash
python your_script.py
```

### Programmatic Configuration

You can also set the log level when creating a Flow instance:

```python
from sdg_hub.flow import Flow

# Create flow with specific log level
flow = Flow(llm_client, log_level="verbose")

# Or override environment variable
flow = Flow(llm_client, log_level="debug")
```


## Environment Variables

### Core Configuration

| Variable | Description | Default | Example |
|----------|-------------|---------|---------|
| `SDG_HUB_LOG_LEVEL` | Logging verbosity level | `normal` | `verbose` |


## Flow Runner Configuration

The `run_flow` function accepts various configuration parameters:

### Required Parameters

```python
run_flow(
    ds_path="input.json",           # Input dataset path
    save_path="output.json",        # Output save path  
    endpoint="http://localhost:8000/v1",  # LLM endpoint
    flow_path="flows/my_flow.yaml", # Flow configuration
)
```

### Optional Parameters

```python
run_flow(
    # ... required params ...
    
    # Performance tuning
    batch_size=8,                   # Items per batch
    num_workers=32,                 # Parallel workers
    
    # Reliability  
    checkpoint_dir="./checkpoints", # Checkpoint directory
    save_freq=2,                    # Save every N batches
    
    # Logging (alternative to env var)
    log_level="verbose",            # Override log level
)
```

### Performance Tuning

| Parameter | Description | Recommended Values |
|-----------|-------------|--------------------|
| `batch_size` | Number of items processed together | 4-16 for GPT models, 1-4 for large models |
| `num_workers` | Parallel processing workers | 2x CPU cores, but respect API rate limits |
| `save_freq` | Save checkpoint frequency | 1-5 batches depending on batch size |

## Flow Configuration

### YAML Structure

Flows are defined in YAML files with this basic structure:

```yaml
- block_type: LLMBlock
  block_config:
    block_name: "unique_name"
    config_path: "path/to/prompt.yaml"
    model_id: "gpt-4"
    output_cols: ["response"]
  gen_kwargs:
    max_tokens: 512
    temperature: 0.7
  drop_columns: ["intermediate_col"]
  drop_duplicates: ["key_column"]
```

### Block Configuration Options

| Option | Description | Required |
|--------|-------------|----------|
| `block_type` | Type of block to instantiate | Yes |
| `block_config` | Block-specific configuration | Yes |
| `gen_kwargs` | LLM generation parameters | No |
| `drop_columns` | Columns to remove after processing | No |
| `drop_duplicates` | Columns to use for deduplication | No |

## Debugging and Troubleshooting

### Common Issues

1. **No output visible**
   - Check `SDG_HUB_LOG_LEVEL` is not set to `quiet`
   - Verify your script is actually running flows

2. **Too much output**
   - Set `SDG_HUB_LOG_LEVEL=normal` or `quiet`
   - Reduce verbosity for production runs

3. **Performance monitoring**
   - Use `verbose` mode to see dataset sizes
   - Use `debug` mode to inspect data transformations
   - Monitor checkpoint saves in production

### Best Practices

1. **Development**: Use `verbose` or `debug` mode
2. **Production**: Use `normal` mode  
3. **CI/CD**: Use `quiet` mode to reduce log noise
4. **Debugging**: Use `debug` mode to inspect data flow

## Advanced Configuration

### Custom Logging

For more advanced logging needs, you can access the underlying logger:

```python
import logging
from sdg_hub.logger_config import setup_logger

# Get the SDG Hub logger
logger = setup_logger("my_module")

# Configure additional handlers
handler = logging.FileHandler("sdg_hub.log")
logger.addHandler(handler)
```

### Performance Optimization

For large-scale processing:

```python
run_flow(
    # ... other params ...
    
    # Optimize for throughput
    batch_size=16,
    num_workers=64,
    save_freq=1,
    
    # Enable checkpointing for reliability
    checkpoint_dir="./checkpoints",
    
    # Reduce logging overhead
    log_level="quiet",
)
```