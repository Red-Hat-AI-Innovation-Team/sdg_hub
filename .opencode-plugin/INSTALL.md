# Installing sdg-hub for OpenCode

## Prerequisites

- [OpenCode.ai](https://opencode.ai) installed
- Python 3.10+ with uv or pip

## Installation

1. Add sdg-hub to the `plugin` array in your `opencode.json` (global or project-level):

   ```json
   {
     "plugin": ["sdg-hub@git+https://github.com/Red-Hat-AI-Innovation-Team/sdg_hub.git"]
   }
   ```

2. Restart OpenCode. The plugin auto-installs and registers all skills.

3. Install the Python library (needed for generation):
   ```bash
   uv pip install sdg_hub
   ```

Verify by asking: "What synthetic data generation skills do you have?"

## Usage

Use OpenCode's native `skill` tool:

```
use skill tool to list skills
use skill tool to load sdg-hub/data-generation
```

Or just describe what you want: "Generate QA pairs from my knowledge base using the knowledge infusion flow."

## Updating

Restart OpenCode to pull the latest version.

To pin a specific version:

```json
{
  "plugin": ["sdg-hub@git+https://github.com/Red-Hat-AI-Innovation-Team/sdg_hub.git#v1.0.0"]
}
```

## Tool Mapping

When skills reference Claude Code tools:
- `Bash(...)` → your native shell execution tool
- `Read`, `Write`, `Edit` → your native file tools
- `/sdg-setup`, `/sdg-generate`, `/sdg-flows` commands → invoke the matching skill directly

## Troubleshooting

### Plugin not loading

1. Check logs: `opencode run --print-logs "hello" 2>&1 | grep -i sdg-hub`
2. Verify the plugin line in your `opencode.json`

### Skills not found

1. Use `skill` tool to list what's discovered
2. Check that the plugin is loading (see above)

## Uninstalling

Remove the `sdg-hub` entry from the `plugin` array in `opencode.json` and restart OpenCode.
