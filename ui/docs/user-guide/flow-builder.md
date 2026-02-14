# User Guide: Flow Builder

The Flow Builder lets you create custom data generation pipelines using a visual node-based editor. Drag nodes onto a canvas, connect them, and configure each block to build your flow.

## Interface Overview

```
┌─────────────────────────────────────────────────────────────────┐
│ ← Back         Visual Flow Builder          [Guide][Clear][Test][Save] │
├─────────────┬───────────────────────────────────┬───────────────┤
│             │                                   │               │
│  Node       │          Canvas                   │  Node Config  │
│  Sidebar    │                                   │  Drawer       │
│             │     ┌─────────┐    ┌──────────┐  │  (appears on  │
│  Node Lib.  │     │  LLM    │───▶│  Parser  │  │   node click) │
│  ─────────  │     │  Node   │    │  Node    │  │               │
│  LLM        │     └─────────┘    └──────────┘  │  Block Name   │
│  Parser     │                          │        │  Input Cols   │
│  Transform  │                          ▼        │  Output Cols  │
│  Eval       │                    ┌──────────┐  │  Settings     │
│             │                    │Transform │  │               │
│  Templates  │                    │  Node    │  │               │
│  ─────────  │                    └──────────┘  │               │
│  QA Gen.    │                                   │               │
│  Summary    │    "Drag nodes here to start"     │               │
│  ...        │    (empty state)                  │               │
│             │                                   │               │
└─────────────┴───────────────────────────────────┴───────────────┘
```

### Left Panel: Node Sidebar

The sidebar has two tabs:

**Node Library** — Draggable node types to add to the canvas:

| Node Type | Purpose | Examples |
|-----------|---------|----------|
| **LLM** | Chat completion and prompt building | ChatCompletionBlock, PromptBuilderBlock |
| **Parser** | Text parsing and extraction | TextParserBlock, JsonExtractorBlock |
| **Transform** | Data transformation | ColumnMapperBlock, FilterBlock |
| **Eval** | Quality evaluation | FaithfulnessBlock, RelevancyBlock |

**Flow Templates** — Pre-built flow configurations organized by category. Click a template to load all its nodes and connections onto the canvas at once.

### Center: Canvas

The main workspace where you build your flow:

- **Drag from sidebar** — Add new nodes to the canvas
- **Click and drag** — Move nodes around
- **Connect ports** — Drag from an output port to an input port to create edges
- **Click node** — Opens the configuration drawer on the right
- **Delete key** — Remove selected node or edge
- **Zoom/Pan** — Scroll to zoom, drag background to pan
- **Empty state** — Shows "Drag nodes here to start building" when no nodes exist

### Right Panel: Node Config Drawer

Click any node to open the configuration drawer:

- Block name and type
- Input/output column configuration
- Block-specific settings
- Prompt template editing (for LLM blocks)
- Close and delete options

### Toolbar

| Action | Description |
|--------|-------------|
| **Interactive Guide** | Launches a guided tour for new users |
| **Clear All** | Removes all nodes from the canvas |
| **Test** | Opens the test runner to execute the flow step-by-step |
| **Save** | Saves the flow (opens metadata form if needed) |

### Workspace Sync

The toolbar shows workspace sync status:

- **Auto-saved** — Changes are saved to the workspace
- **Syncing...** — Save in progress
- **Sync Error** — Failed to save (check connection)

## Adding Nodes

### From Node Library

1. Open the **Node Library** tab in the sidebar
2. Drag a node type onto the canvas
3. The node appears at the drop position
4. Click the node to configure it in the drawer

### From Flow Templates

1. Open the **Flow Templates** tab in the sidebar
2. Browse templates by category
3. Click a template to load it — all nodes and connections are placed on the canvas
4. Modify individual nodes as needed

## Connecting Nodes

### Creating Connections

1. Hover over a node to reveal its output port
2. Click and drag from the output port
3. Drop onto another node's input port
4. A connection edge appears between the nodes

### Connection Validation

The editor validates connections between nodes:

- Ensures output types match input expectations
- Prevents circular dependencies
- Highlights invalid connections

### Data Flow

```
Dataset → Node 1 → Node 2 → Node 3 → Output

Each node:
1. Receives output columns from connected upstream nodes
2. Processes data according to its configuration
3. Provides its output columns to downstream nodes
```

## Node Configuration

Click any node to open the configuration drawer on the right:

### Basic Settings

| Field | Description |
|-------|-------------|
| **Block Name** | Unique identifier for this block instance |
| **Description** | Optional notes about this block's purpose |

### Block-Specific Configuration

Each block type has its own settings. Common examples:

**ChatCompletionBlock:**

```yaml
input_cols:
  - prompt
output_cols:
  - response
model_settings:
  temperature: 0.7
  max_tokens: 2048
```

**PromptBuilderBlock:**

```yaml
template_path: prompts/my_prompt.yaml
input_cols:
  - document
  - query
output_cols:
  - prompt
```

**ColumnMapperBlock:**

```yaml
mappings:
  old_column: new_column
  source: target
```

### Prompt Editing

For blocks that use prompt templates, click **Edit Prompt** to open the prompt editor:

```
┌─────────────────────────────────────────────────────────────┐
│ Edit Prompt: generate_questions                              │
├─────────────────────────────────────────────────────────────┤
│ Template (Jinja2):                                          │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ You are a helpful assistant that generates questions.   │ │
│ │                                                         │ │
│ │ Given the following document:                           │ │
│ │ {{ document }}                                          │ │
│ │                                                         │ │
│ │ Generate {{ num_questions }} thoughtful questions.      │ │
│ └─────────────────────────────────────────────────────────┘ │
│                                                             │
│ Variables Available:                                        │
│ • document (from dataset)                                   │
│ • num_questions (parameter)                                 │
│                                                             │
│                              [Cancel] [Save Prompt]         │
└─────────────────────────────────────────────────────────────┘
```

Prompts use **Jinja2 templating**:

- `{{ variable }}` — Insert variable value
- `{% if condition %}...{% endif %}` — Conditional content
- `{% for item in list %}...{% endfor %}` — Loops

## Step-by-Step Flow Testing

Before saving your flow, you can test it with the **Test Runner**:

### Starting a Test

1. Click **Test** in the toolbar
2. Configure test settings in the **Test Config Modal**:
   - Select or configure a model
   - Set API base and key
3. Click **Run Test**

### Viewing Results

The test executes blocks one at a time:

1. Each block runs sequentially
2. **Node I/O Modal** shows the input and output of each block
3. View intermediate results between blocks
4. Identify issues in specific blocks

### Test Runner Interface

```
┌─────────────────────────────────────────────────────────────┐
│ Test Runner                                                  │
├─────────────────────────────────────────────────────────────┤
│ Block 1: build_prompt          ✅ Complete                   │
│   Input:  {"document": "..."}                               │
│   Output: {"prompt": "You are..."}                          │
│                                                             │
│ Block 2: generate_qa           🔄 Running...                │
│   Input:  {"prompt": "You are..."}                          │
│   Output: (pending)                                         │
│                                                             │
│ Block 3: parse_response        ⏳ Waiting                    │
└─────────────────────────────────────────────────────────────┘
```

## Guided Tour

First-time users are offered a **guided tour** that walks through:

- Adding nodes from the sidebar
- Connecting nodes with edges
- Configuring node settings
- Testing and saving the flow

Click **Interactive Guide** in the toolbar to restart the tour at any time.

## Saving Your Flow

### Save Button

When you have at least one node, click **Save** in the toolbar.

### Metadata Form

Provide flow metadata:

| Field | Required | Description |
|-------|----------|-------------|
| **Name** | Yes | Flow display name |
| **Description** | No | What this flow does |
| **Version** | No | Semantic version (e.g., 1.0.0) |
| **Author** | No | Creator name |
| **Tags** | No | Searchable keywords |
| **Required Columns** | No | Dataset columns this flow needs |

### After Saving

Your custom flow:

1. Appears in the flow list with "(Custom)" suffix
2. Can be selected like any SDG Hub flow
3. Can be edited, cloned, or deleted
4. Is saved to `backend/custom_flows/[flow_name]/`

## Custom Flows Management

### Downloading Flows

From the Dashboard or Custom Flows section:

- **Download Single File** — Download a specific YAML file
- **Download All** — Download the entire flow as a ZIP archive

### Deleting Flows

- Delete individual custom flows
- Delete all custom flows at once

## Flow Files

Custom flows are saved as YAML:

```yaml
# custom_flows/my_flow/flow.yaml
metadata:
  name: My Custom Flow
  description: Generates Q&A pairs from documents
  version: 1.0.0
  author: Your Name
  tags:
    - question-generation
    - custom
  required_columns:
    - document
    - domain

blocks:
  - block_type: PromptBuilderBlock
    block_config:
      block_name: build_prompt
      template_path: prompts/qa_prompt.yaml
      input_cols:
        - document
      output_cols:
        - prompt

  - block_type: ChatCompletionBlock
    block_config:
      block_name: generate_qa
      input_cols:
        - prompt
      output_cols:
        - response
```

## Tips for Building Flows

### Design Principles

1. **Single Responsibility** — Each node does one thing well
2. **Clear Data Flow** — Connect outputs to relevant inputs
3. **Meaningful Names** — Use descriptive block names
4. **Validate Early** — Add validation nodes before expensive LLM calls

### Common Patterns

**Generate and Evaluate:**

```
PromptBuilder → ChatCompletion → TextParser → Evaluator
```

**Multi-Stage Generation:**

```
Summary → QA Generation → Answer Verification → Scoring
```

**Filter and Transform:**

```
QualityFilter → ColumnMapper → Deduplication → Output
```

### Debugging Tips

1. **Start with templates** — Load a flow template and modify it
2. **Use the Test Runner** — Test nodes step-by-step before saving
3. **Add nodes incrementally** — Test after each addition
4. **Use dry runs** — Validate with small samples after saving
5. **Check column names** — Most errors are column mismatches
6. **Review prompts** — Ensure templates reference correct variables
7. **Try the guided tour** — If using the editor for the first time

## Next Steps

- [Model Configuration](model-configuration.md) — Configure your LLM
- [Dataset Configuration](dataset-configuration.md) — Set up your data
- [Running Generation](generation.md) — Execute your custom flow
