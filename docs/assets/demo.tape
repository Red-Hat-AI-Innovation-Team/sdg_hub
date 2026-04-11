# SDG Hub Demo Recording
Output docs/assets/demo.gif

Set Shell zsh
Set Theme "Catppuccin Mocha"
Set WindowBar Colorful
Set WindowBarSize 40
Set FontFamily "JetBrains Mono"
Set FontSize 14
Set Width 900
Set Height 500
Set Padding 20
Set TypingSpeed 40ms
Set CursorBlink false

# Install
Type "pip install sdg-hub"
Enter
Sleep 500ms
Type "# Successfully installed sdg-hub"
Sleep 1.5s
Enter
Enter
Sleep 500ms

# Start Python
Type "python3"
Enter
Sleep 1s

# Import
Type "from sdg_hub import FlowRegistry, Flow"
Enter
Sleep 800ms

# Discover flows
Type "FlowRegistry.discover_flows()"
Enter
Sleep 1.5s

# Show some output
Type "# Found 14 flows across 5 categories"
Enter
Sleep 1s

# Load a flow
Type "flow = Flow.from_yaml(FlowRegistry.get_flow_path('epic-jade-656'))"
Enter
Sleep 1s

# Check requirements
Type "reqs = flow.get_dataset_requirements()"
Enter
Sleep 500ms
Type "print(reqs.required_columns)"
Enter
Sleep 800ms
Type "# ['document', 'document_outline', 'domain']"
Enter
Sleep 1.5s

# Configure model
Enter
Type "flow.set_model_config(model='openai/gpt-4o')"
Enter
Sleep 800ms

# Dry run
Type "result = flow.dry_run(dataset, sample_size=2)"
Enter
Sleep 2s
Type "# Dry run completed in 4.2s - 8 blocks executed"
Enter
Sleep 1.5s

# Generate
Type "output = flow.generate(dataset)"
Enter
Sleep 2s
Type "# Generated 50 rows with 12 columns"
Enter
Sleep 1s
Type "print(output.columns.tolist())"
Enter
Sleep 800ms
Type "# ['question', 'response', 'faithfulness_score', ...]"
Enter
Sleep 3s
