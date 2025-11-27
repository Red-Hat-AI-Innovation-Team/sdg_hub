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
│ • Flows  │   (Changes based on selected navigation)          │
│ • History│                                                    │
│          │                                                    │
└──────────┴───────────────────────────────────────────────────┘
```

### Header Bar

- **SDG Hub Logo** — Returns to home
- **Sidebar Toggle** — Show/hide the navigation sidebar

### Sidebar Navigation

| Item | Description |
|------|-------------|
| **Data Generation Flows** | Main dashboard for managing flow configurations |
| **Flow Runs History** | View past generation runs and download outputs |

### Main Content Area

The content changes based on your navigation selection:

- **Flows Page** — Configuration list and status dashboard
- **Configure Flow** — Step-by-step configuration wizard
- **Flow Runs History** — Historical run records

## Data Generation Flows Page

This is the main dashboard where you manage all your flow configurations.

### Summary Dashboard

At the top, you'll see status cards showing:

**Configuration Status:**

- ✅ **Configured** — Ready-to-run configurations with all settings complete
- ⚠️ **Not Configured** — Partially configured flows needing completion
- 📦 **Drafts** — Work-in-progress flows saved locally

**Execution Status:**

- 🔄 **Running** — Currently executing generations
- ❌ **Failed** — Generations that encountered errors
- ✅ **Completed** — Successfully finished generations
- ⛔ **Stopped** — User-cancelled generations

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

Click on any flow name to expand its detail view:

```
┌─────────────────────────────────────────────────────────────┐
│ Flow Name                                        [Actions]  │
├─────────────────────────────────────────────────────────────┤
│ Configuration Summary    │    Terminal / Monitoring View   │
│ • Flow: ...             │    ┌─────────────────────────┐  │
│ • Model: ...            │    │ $ Starting generation... │  │
│ • Dataset: ...          │    │ Block 1/4: ...           │  │
│ • Samples: ...          │    │ ...                      │  │
└─────────────────────────────────────────────────────────────┘
```

The expanded view shows:

- **Left Panel** — Configuration summary with all settings
- **Right Panel** — Terminal output (when running) or monitoring view

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

- **Navigation** — Current page is remembered
- **Expanded Config** — Last viewed configuration stays expanded
- **Wizard Progress** — Incomplete wizard sessions are saved
- **Execution States** — Running/completed states persist

## Next Steps

- [Flow Configuration](flow-configuration.md) — Learn to create configurations
- [Flow Builder](flow-builder.md) — Build custom flows
- [Running Generation](generation.md) — Execute and monitor flows

