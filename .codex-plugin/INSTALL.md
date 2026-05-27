# Installing sdg-hub for Codex

## Via Marketplace (Recommended)

```bash
codex plugin marketplace add Red-Hat-AI-Innovation-Team/plugins
```

Then install the plugin from the marketplace. The Python library will need to be installed separately:

```bash
pip install sdg-hub
```

## Manual Installation

If you prefer to install manually:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Red-Hat-AI-Innovation-Team/sdg_hub.git ~/.codex/sdg-hub
   ```

2. **Install the Python library:**
   ```bash
   pip install -e ~/.codex/sdg-hub
   ```

3. **Create the skills symlink** (skills are in `.claude/skills/` — shared between Claude Code and Codex):
   ```bash
   mkdir -p ~/.agents/skills
   ln -s ~/.codex/sdg-hub/.claude/skills ~/.agents/skills/sdg-hub
   ```

4. **Restart Codex** to discover the skills.

## Updating

Marketplace installs update automatically. For manual installs:
```bash
cd ~/.codex/sdg-hub && git pull
```

## Uninstalling

For manual installs:
```bash
rm ~/.agents/skills/sdg-hub
rm -rf ~/.codex/sdg-hub
```
