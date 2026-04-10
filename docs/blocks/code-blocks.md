# Code Blocks

Code blocks execute and validate generated code inside sandboxed runtimes. They are useful when your flow needs to verify that synthesized code actually runs before it is used for training or benchmarking.

## Available Code Blocks

### PythonInterpreterBlock

Executes Python code from one input column and stores structured execution results in one output column. The default runtime is the `monty` connector, which runs code in a restricted sandbox.

**Configuration:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_cols` | `list[str]` | - | Exactly one column containing Python code |
| `output_cols` | `list[str]` | - | Exactly one column to store execution results |
| `interpreter_framework` | `str` | `"monty"` | Code interpreter connector name |
| `timeout` | `float` | `30.0` | Max execution time per row in seconds |
| `max_concurrency` | `int` | `10` | Maximum concurrent code executions |

**YAML Example:**

```yaml
- block_type: "PythonInterpreterBlock"
  block_config:
    block_name: "verify_code"
    interpreter_framework: "monty"
    input_cols:
      - "executable_code"
    output_cols:
      - "execution_result"
    timeout: 10.0
    max_concurrency: 5
```

**Output Schema (per row):**

- `success` - `true` when execution completes without runtime errors
- `output` - captured stdout content
- `error` - runtime error message when execution fails
- `return_value` - returned value from the interpreter
- `execution_time_ms` - execution duration in milliseconds

## Runtime Setup

Install the optional code runtime dependency:

```bash
uv pip install sdg-hub[code]
```

## Next Steps

- **[Block Overview](overview.md)** - How block composition works end-to-end
- **[Flow Catalog](../flows/available-flows.md)** - Flows that use code execution
- **[Custom Blocks](custom-blocks.md)** - Build your own block types
