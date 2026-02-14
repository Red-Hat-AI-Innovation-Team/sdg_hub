# User Guide: Overview

This guide provides a complete overview of the SDG Hub UI and how to navigate its features.

## Main Interface

The UI consists of three main areas:

```
┌──────────────────────────────────────────────────────────────┐
│                     Header Bar                                │
├──────────┬───────────────────────────────────────────────────┤
│          │                                                    │
│ Sidebar  │              Main Content Area                     │
│          │                                                    │
│ • Home   │   (Changes based on selected navigation)          │
│ • Dashb. │                                                    │
│ • Flows  │                                                    │
│ • History│                                                    │
│          │                                                    │
└──────────┴───────────────────────────────────────────────────┘
```

### Header Bar

- **Sidebar Toggle** — Show/hide the navigation sidebar

### Sidebar Navigation

| Item | Description |
|------|-------------|
| **Home** | Welcome page with quick-start actions and prebuilt flow catalog |
| **Dashboard** | Overview of your flows, statistics, preprocessed datasets, and custom flows |
| **Data Generation Flows** | Main page for managing flow configurations and running generation |
| **Flow Runs History** | View past generation runs, download outputs, and analyze logs |

### Main Content Area

The content changes based on your navigation selection:

- **Home** — Welcome dashboard with prebuilt flows by category and quick-start actions
- **Dashboard** — Your flows overview, statistics, preprocessed datasets, and custom flows
- **Flows Page** — Configuration list, batch operations, and live monitoring
- **Configure Flow** — Multi-step configuration wizard
- **Flow Runs History** — Historical run records with filtering, preview, and download

## Home Dashboard

The landing page when you first open SDG Hub UI:

- **Quick-Start Actions** — Create a new flow configuration with one click
- **Prebuilt Flows** — Browse SDG Hub flows organized by category (QA, summarization, etc.)
- **Navigation Shortcuts** — Jump directly to Dashboard, Flows, or Run History

## Dashboard

A central overview of all your work:

### Your Flows
- Collapsible section showing all saved flow configurations
- Status badges for each configuration

### Flow Statistics
- Total samples generated across all runs
- Number of runs by status (completed, failed, cancelled)
- Average run durations
- LLM request counts

### Preprocessed Datasets
- PDF preprocessing jobs and their status
- Quick access to preprocessed datasets for use in flows

### My Custom Flows
- List of custom flows you've built
- Download or delete custom flow files

## Data Generation Flows Page

This is the main dashboard where you manage all your flow configurations.

### Summary Dashboard

At the top, you'll see status cards showing:

**Configuration Status:**

- **Configured** — Ready-to-run configurations with all settings complete
- **Not Configured** — Partially configured flows needing completion
- **Drafts** — Work-in-progress flows saved locally

**Execution Status:**

- **Running** — Currently executing generations
- **Failed** — Generations that encountered errors
- **Completed** — Successfully finished generations
- **Stopped** — User-cancelled generations

### Configuration Table

The main table displays all your saved configurations:

| Column | Description |
|--------|-------------|
| **Checkbox** | Select for batch operations |
| **Flow Name** | Click to expand details |
| **Status** | Configuration/execution status badge |
| **Model** | Configured LLM model |
| **Dataset** | Loaded dataset file |
| **Actions** | Edit, Clone, Delete, Run/Stop |

### Toolbar Actions

- **Search** — Filter by flow name, model, dataset, or tags
- **Actions Menu** — Batch run, stop, or delete selected configurations
- **Configure Flow** — Start the configuration wizard

### Expanding a Configuration

Click on any flow name to expand its detail view with tabs:

**Overview Tab:**
- Configuration summary with all settings (flow, model, dataset)
- Tags and metadata

**Running Process Tab:**
- Terminal output with real-time logs (when running)
- Live monitoring with block-by-block progress
- Token usage statistics

**Generated Datasets Tab:**
- List of output files from completed runs
- Download and preview options

## Configuration States

Configurations progress through several states:

```
Draft → Not Configured → Configured → Running → Completed
                                        ↓
                                      Failed/Stopped
```

| State | Badge Color | Meaning |
|-------|-------------|---------|
| `draft` | Purple | Locally saved, not yet submitted |
| `not_configured` | Yellow | Missing model or dataset config |
| `configured` | Green | Ready to run |
| `running` | Blue | Currently generating |
| `completed` | Green | Successfully finished |
| `failed` | Red | Encountered an error |
| `cancelled` | Yellow | Stopped by user |

## Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Enter` | In search, adds current filter as tag |
| `Escape` | Closes modals and dropdowns |

## Session Persistence

The UI preserves your state across page refreshes:

- **Navigation** — Current page is remembered (sessionStorage)
- **Expanded Config** — Last viewed configuration stays expanded (sessionStorage)
- **Wizard Progress** — Incomplete wizard sessions are saved (sessionStorage)
- **Execution States** — Running/completed states persist (localStorage)
- **Drafts** — Flow builder drafts auto-save (localStorage)

## Next Steps

- [Flow Configuration](flow-configuration.md) — Learn to create configurations
- [Flow Builder](flow-builder.md) — Build custom flows with the visual editor
- [Running Generation](generation.md) — Execute and monitor flows
