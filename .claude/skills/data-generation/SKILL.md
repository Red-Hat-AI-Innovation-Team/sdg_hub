---
name: data-generation
description: "Use when the user wants to run synthetic data generation via scripts — detect environment, execute a flow, and present results. For detailed guidance on approaches, blocks, flow authoring, and troubleshooting, consult the synthetic-data-generation skill."
allowed-tools: ["Bash(${CLAUDE_PLUGIN_ROOT}/scripts/sdg_generate.sh:*)", "Bash(${CLAUDE_PLUGIN_ROOT}/scripts/sdg_detect.sh:*)", "Bash(${CLAUDE_PLUGIN_ROOT}/scripts/sdg_flows.sh:*)"]
---

# Run Data Generation

Execute synthetic data generation using sdg_hub flows. For approach selection, custom flow authoring, and block reference, consult the `synthetic-data-generation` skill.

## Step 1: Check Environment

```!
"${CLAUDE_PLUGIN_ROOT}/scripts/sdg_detect.sh"
```

### If not ready

- `library=missing` or `config=missing`: invoke the `setup-guide` skill.

### If ready (`library=installed`, `config=found`)

Proceed to Step 2.

## Step 2: Execute Generation

If the user doesn't specify a flow, invoke the `flow-browser` skill to find one.

Recommend starting with `--sample 2` for a dry run.

```!
"${CLAUDE_PLUGIN_ROOT}/scripts/sdg_generate.sh" $ARGUMENTS
```

## Step 3: Present Results

1. **Generation status** — Whether generation completed successfully
2. **Row counts** — Input rows processed and output rows generated
3. **Output location** — Path to the generated dataset
4. **Errors** — If any rows failed, report the count

If generation failed, consult the `synthetic-data-generation` skill for troubleshooting.
