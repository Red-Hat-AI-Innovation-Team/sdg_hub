# Code Blocks

Code blocks execute Python code from dataset rows in sandboxed interpreters and capture structured results. They are designed for validating synthetic code datasets by testing whether generated code runs successfully.

!!! note "Optional dependency"
    Code blocks require the `code` optional dependency group. Install with:
    `uv pip install sdg-hub[code]` or `pip install sdg-hub[code]`

---

## PythonInterpreterBlock

Executes Python code stored in a DataFrame column using a sandboxed code interpreter connector. Each row's code is executed independently with configurable timeouts and concurrency. Results are written as structured dicts containing success status, captured output, errors, and execution timing.

The block reads code from a single input column and writes a `CodeExecutionResult` dict to a single output column. It also creates a flat boolean column `{output_col}_success` for convenient downstream filtering (e.g., with `ColumnValueFilterBlock`).

### Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `block_name` | `str` | required | Unique identifier for this block instance |
| `input_cols` | `list[str]` | required | Single-element list with the column containing code to execute |
| `output_cols` | `list[str]` | required | Single-element list with the column name for execution results |
| `interpreter_framework` | `str` | `"monty"` | Code interpreter connector to use (registered in `ConnectorRegistry`) |
| `timeout` | `float` | `30.0` | Maximum execution time per code snippet in seconds (must be > 0) |
| `max_concurrency` | `int` | `10` | Maximum concurrent executions (must be > 0) |

### Input/Output Contract

- **Input:** Exactly one column (`input_cols` must be a single-element list) containing Python code as strings.
- **Output:** Exactly one column (`output_cols` must be a single-element list) containing `CodeExecutionResult` dicts, plus a flat boolean column `{output_col}_success`.

### Output Format

Each row receives a `CodeExecutionResult` dict in the output column:

| Field | Type | Description |
|-------|------|-------------|
| `success` | `bool` | Whether the code executed without errors |
| `output` | `str` or `null` | Captured stdout/print output |
| `error` | `str` or `null` | Error message if execution failed |
| `return_value` | `Any` or `null` | Return value from execution |
| `execution_time_ms` | `float` or `null` | Execution time in milliseconds |

The additional `{output_col}_success` boolean column allows direct use with `ColumnValueFilterBlock` to filter out failed executions without inspecting the result dict.

### Python Example

```python
from sdg_hub.core.blocks import PythonInterpreterBlock
import pandas as pd

block = PythonInterpreterBlock(
    block_name="validate_code",
    input_cols=["code"],
    output_cols=["result"],
    timeout=5.0,
)

df = pd.DataFrame({
    "code": [
        "print('Hello, world!')",
        "1 / 0",
        "x = sum(range(10))\nprint(x)",
    ],
})

result = block(df)

# Successful execution
print(result["result"].iloc[0])
# {'success': True, 'output': 'Hello, world!\n', 'error': None,
#  'return_value': None, 'execution_time_ms': 1.23}

# Failed execution
print(result["result"].iloc[1])
# {'success': False, 'output': None, 'error': 'ZeroDivisionError: ...',
#  'return_value': None, 'execution_time_ms': 0.45}

# Filter to only successful executions
print(result["result_success"].tolist())
# [True, False, True]
```

### YAML Example

```yaml
- block_type: PythonInterpreterBlock
  block_config:
    block_name: validate_generated_code
    interpreter_framework: monty
    input_cols:
      - generated_code
    output_cols:
      - execution_result
    timeout: 10.0
    max_concurrency: 5
```

### Combining with ColumnValueFilterBlock

A common pattern is to execute code and then filter out rows where execution failed:

```yaml
- block_type: PythonInterpreterBlock
  block_config:
    block_name: verify_execution
    interpreter_framework: monty
    input_cols:
      - executable_code
    output_cols:
      - execution_result
    timeout: 10.0

- block_type: ColumnValueFilterBlock
  block_config:
    block_name: filter_failed
    input_cols:
      - execution_result_success
    filter_value: true
    operation: "eq"
```

### Execution Model

- Code snippets are executed concurrently using `asyncio.Semaphore` with `max_concurrency` controlling the parallelism.
- Empty, `NaN`, or non-string code values produce an error result without calling the interpreter.
- The block handles both sync and async event loop contexts automatically via `ThreadPoolExecutor`.
- Progress is tracked with a `tqdm` progress bar during execution.
