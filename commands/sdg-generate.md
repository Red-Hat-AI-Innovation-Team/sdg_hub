---
description: "Run synthetic data generation using a flow"
argument-hint: "<flow> <input-file> [--output <file>] [--sample N]"
allowed-tools: ["Bash(${CLAUDE_PLUGIN_ROOT}/scripts/sdg_generate.sh:*)", "Bash(${CLAUDE_PLUGIN_ROOT}/scripts/sdg_detect.sh:*)"]
---

# sdg-hub Generate

Run synthetic data generation using a flow on input data.

## Step 1: Check Environment

```!
"${CLAUDE_PLUGIN_ROOT}/scripts/sdg_detect.sh"
```

If `config=missing`, tell the user to run `/sdg-setup` first.

## Step 2: Parse Arguments

Extract from `$ARGUMENTS`:
- `flow` — flow name from the registry or path to a YAML file (required)
- `input-file` — path to input dataset, JSONL format (required)
- `--output` — output file path (default: `<input>_generated.jsonl`)
- `--sample N` — dry-run with N samples before full generation (recommended for first use)

If the user doesn't specify a flow, suggest they run `/sdg-flows` to browse available options.

## Step 3: Execute Generation

Run the generation script:

```!
"${CLAUDE_PLUGIN_ROOT}/scripts/sdg_generate.sh" $ARGUMENTS
```

## Step 4: Present Results

Parse the JSON response and present clearly:

1. **Generation status** — Whether generation completed successfully
2. **Row counts** — Input rows processed and output rows generated
3. **Output location** — Path to the generated dataset
4. **Errors** — If any rows failed, report the count and suggest checking the output for error entries

If generation failed, show the error and suggest troubleshooting:
- Missing columns: Check that the input dataset has the columns required by the flow
- API errors: Verify API key and endpoint in `.sdg-hub/config.json`
- Rate limiting: Reduce `--concurrency` or add delay
