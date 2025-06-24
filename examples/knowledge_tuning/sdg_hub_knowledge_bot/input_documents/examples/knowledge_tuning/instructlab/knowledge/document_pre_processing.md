# Preparing Document for Data generation
- This notebook will show you how to do document parsing (Converting document from various formats like pdf, html etc to markdown format for ingesting in a LLM)
- Document Chunking
- And finally mixing it with user QNA yaml to  create seed examples

### Install SDG

```bash 
pip install sdg-hub==0.1.0a2
pip install rich datasets tabulate transformers
```

### Using docling v1


```python
data_dir = 'document_collection/ibm-annual-report'
!OMP_NUM_THREADS=32 mamba run -n docling python ../scripts/docparser.py --input-dir {data_dir} --output-dir {data_dir}
```

### Using docling v2


```python
data_dir = 'document_collection/ibm-annual-report'
!OMP_NUM_THREADS=32 mamba run -n docling python ../scripts/docparser_v2.py --input-dir {data_dir} --output-dir {data_dir} --c docling_v2_config.yaml
```

### Create Seed Examples


```python
from utils import DocProcessor

output_dir = f"sdg_demo_output/"
# This is where your PDFs are stored
data_dir = '../document_collection/ibm-annual-report' 
# It also have your QNA yaml file
dp = DocProcessor(data_dir, user_config_path=f'{data_dir}/qna.yaml')

### Using docling v1 json
seed_data = dp.get_processed_dataset()

### Using markdown file
seed_data = dp.get_processed_markdown_dataset([f"{data_dir}/ibm-annual-report-2024.md"])

# Note: For now v2 json is not supported

seed_data.to_json(f'{output_dir}/seed_data.jsonl', orient='records', lines=True)
```
