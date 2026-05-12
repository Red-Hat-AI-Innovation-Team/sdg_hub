# Installing sdg-hub for Codex

Enable sdg-hub synthetic data generation skills in Codex via native skill discovery.

## Prerequisites

- Git
- Python 3.10+ with uv or pip
- [Codex CLI](https://github.com/openai/codex) installed

## Installation

1. **Clone the sdg-hub repository:**
   ```bash
   git clone https://github.com/Red-Hat-AI-Innovation-Team/sdg_hub.git ~/.codex/sdg-hub
   ```

2. **Install the Python library:**
   ```bash
   cd ~/.codex/sdg-hub && uv sync --extra dev
   ```
   Or with pip:
   ```bash
   pip install -e ~/.codex/sdg-hub
   ```

3. **Create the skills symlink:**
   ```bash
   mkdir -p ~/.agents/skills
   ln -s ~/.codex/sdg-hub/skills ~/.agents/skills/sdg-hub
   ```

   **Windows (PowerShell):**
   ```powershell
   New-Item -ItemType Directory -Force -Path "$env:USERPROFILE\.agents\skills"
   cmd /c mklink /J "$env:USERPROFILE\.agents\skills\sdg-hub" "$env:USERPROFILE\.codex\sdg-hub\skills"
   ```

4. **Restart Codex** to discover the skills.

## Path Resolution

When skills reference `${CLAUDE_PLUGIN_ROOT}/scripts/...`, use the clone path instead:
```bash
~/.codex/sdg-hub/scripts/sdg_detect.sh
~/.codex/sdg-hub/scripts/sdg_generate.sh
~/.codex/sdg-hub/scripts/sdg_flows.sh
```

## Verify

```bash
ls -la ~/.agents/skills/sdg-hub
```

You should see a symlink pointing to the sdg-hub skills directory.

## Updating

```bash
cd ~/.codex/sdg-hub && git pull
```

Skills update instantly through the symlink.

## Uninstalling

```bash
rm ~/.agents/skills/sdg-hub
rm -rf ~/.codex/sdg-hub
```
