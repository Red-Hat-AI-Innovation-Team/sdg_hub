# Examples

SDG Hub includes comprehensive examples to help you understand different use cases and patterns.

## Knowledge Tuning Examples

### Basic Data Generation with Llama

Location: `examples/knowledge_tuning/data-generation-with-llama-70b/`

This example demonstrates how to generate synthetic knowledge data using Llama models:

- **Notebook**: `data-generation-with-llama-70b.ipynb`
- **Flow Configuration**: `synth_knowledge1.5_llama3.3.yaml`

### InstructLab Integration

Location: `examples/knowledge_tuning/instructlab/`

Shows integration with InstructLab for document processing and Q&A generation:

**Key Files:**
- `document_pre_processing.ipynb` - Document preprocessing pipeline
- `knowledge_generation_and_mixing.ipynb` - Knowledge generation workflow
- `docparser.py` and `docparser_v2.py` - Document parsing utilities

**Sample Data:**
- IBM Annual Report processing example in `document_collection/ibm-annual-report/`

### Advanced Knowledge Tuning with Reasoning

Location: `examples/knowledge_tuning/knowledge_tuning_with_reasoning_model/`

Demonstrates advanced knowledge generation with reasoning capabilities:

**Flow Configurations:**
- `synth_knowledge1.5_nemotron_super_49b.yaml` - Basic knowledge generation
- `synth_knowledge_reasoning_nemotron_super_49b.yaml` - With reasoning
- `synth_knowledge_reasoning_nemotron_super_49b_rewrite_with_diversity.yaml` - With diversity

**Notebooks:**
- `reasoning_sdg_financebench.ipynb` - Financial benchmarking
- `reasoning_sdg_quality.ipynb` - Quality assessment

## Skills Tuning Examples

### InstructLab Skills

Location: `examples/skills_tuning/instructlab/`

Comprehensive skills generation examples:

**Flow Configurations:**
- `detailed_annotation.yaml` - Detailed text annotation
- `grounded_summary_extraction.yaml` - Summary extraction
- `simple_annotation.yaml` - Basic annotation
- `unstructured_to_structured.yaml` - Data structuring

**Custom Blocks:**
- `add_question.py` - Question generation block
- `docling_parse_pdf.py` - PDF parsing block
- `json_format.py` - JSON formatting block

**Notebooks:**
- `annotation_classification.ipynb` - Classification tasks
- `structured_summary.ipynb` - Summary generation
- `table_manipulation.ipynb` - Table processing
- `unstructured_to_structured.ipynb` - Data transformation

## Web Interface

Location: `web_interface/`

A Flask-based web interface for managing flows:

```bash
# Install web interface dependencies
pip install -e .[web_interface]

# Run the web interface
cd web_interface
python app.py
```

Features:
- Flow visualization
- Interactive flow execution
- Real-time progress monitoring

## Running Examples

### Prerequisites

Install examples dependencies:

```bash
pip install -e .[examples]
```

### Knowledge Generation Example

```python
from sdg_hub.flow_runner import run_flow

# Run the basic knowledge generation flow
run_flow(
    ds_path="examples/knowledge_tuning/instructlab/document_collection/ibm-annual-report/ibm-annual-report-2024.json",
    save_path="output/knowledge_qa.json",
    endpoint="https://api.openai.com/v1",
    flow_path="src/sdg_hub/flows/generation/knowledge/synth_knowledge1.5.yaml",
    checkpoint_dir="./checkpoints",
    batch_size=4,
    num_workers=8
)
```

### Skills Generation Example

```python
# Run the skills generation flow
run_flow(
    ds_path="examples/skills_tuning/instructlab/seed_data/financial_call_transcripts.jsonl",
    save_path="output/skills_data.json",
    endpoint="https://api.openai.com/v1",
    flow_path="src/sdg_hub/flows/generation/skills/synth_skills.yaml",
    checkpoint_dir="./checkpoints",
    batch_size=2,
    num_workers=4
)
```

## Custom Block Examples

Many examples include custom blocks that extend SDG Hub's functionality:

### PDF Processing Block

```python
# From examples/skills_tuning/instructlab/blocks/docling_parse_pdf.py
@BlockRegistry.register("DoclingParsePDFBlock")
class DoclingParsePDFBlock(Block):
    def generate(self, dataset: Dataset) -> Dataset:
        # Custom PDF parsing logic
        pass
```

### Question Addition Block

```python
# From examples/skills_tuning/instructlab/blocks/add_question.py
@BlockRegistry.register("AddQuestionBlock")  
class AddQuestionBlock(Block):
    def generate(self, dataset: Dataset) -> Dataset:
        # Add questions to existing data
        pass
```

## Jupyter Notebooks

All examples include detailed Jupyter notebooks with:

- Step-by-step explanations
- Code examples
- Expected outputs
- Troubleshooting tips

### Running Notebooks

```bash
# Start Jupyter
jupyter notebook

# Navigate to examples directory and open any .ipynb file
```

## Example Data

The examples include various sample datasets:

- **Financial Data**: Call transcripts, annual reports
- **PDF Documents**: Various PDF processing examples  
- **JSON/JSONL**: Structured data examples
- **YAML Configurations**: Flow and prompt configurations

## Best Practices from Examples

1. **Data Preprocessing**: Always clean and validate input data
2. **Checkpoint Usage**: Use checkpoints for long-running flows
3. **Batch Processing**: Optimize batch sizes based on your data
4. **Error Handling**: Implement robust error handling in custom blocks
5. **Prompt Engineering**: Iterate on prompts using the examples as templates