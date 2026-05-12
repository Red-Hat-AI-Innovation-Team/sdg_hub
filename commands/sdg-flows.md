---
description: "List, search, and inspect available SDG flows"
argument-hint: "[list|search <query>|inspect <flow>]"
allowed-tools: ["Bash(${CLAUDE_PLUGIN_ROOT}/scripts/sdg_flows.sh:*)", "Bash(${CLAUDE_PLUGIN_ROOT}/scripts/sdg_detect.sh:*)"]
---

# sdg-hub Flows

Browse and inspect available synthetic data generation flows.

## Step 1: Check Environment

```!
"${CLAUDE_PLUGIN_ROOT}/scripts/sdg_detect.sh"
```

If `library=missing`, tell the user to install sdg_hub first.

## Step 2: Route by Action

Parse `$ARGUMENTS` to determine the action:

### List all flows (default)

```!
"${CLAUDE_PLUGIN_ROOT}/scripts/sdg_flows.sh" list
```

Present flows in a formatted table with name, description, and tags.

### Search flows

```!
"${CLAUDE_PLUGIN_ROOT}/scripts/sdg_flows.sh" search "<query>"
```

Show matching flows. If no results, suggest broader search terms.

### Inspect a flow

```!
"${CLAUDE_PLUGIN_ROOT}/scripts/sdg_flows.sh" inspect "<flow>"
```

Show detailed information:
1. **Flow name and description**
2. **Block pipeline** — ordered list of blocks in the flow
3. **Required columns** — input dataset columns needed
4. **Usage example** — suggest a `/sdg-generate` command

If no action is specified, default to `list`.
