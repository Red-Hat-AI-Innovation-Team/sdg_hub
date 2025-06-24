``` python
%load_ext autoreload
%autoreload 2
```



``` python
# Standard
import random

# Third Party
from datasets import Dataset
from openai import OpenAI
from rich import print
from rich.console import Console
from rich.panel import Panel
import yaml

# First Party
from sdg_hub.flow import Flow
from sdg_hub.sdg import SDG
from blocks import *
```



## 🧾 Use Case 3: Structured Summary from Conversations {#-use-case-3-structured-summary-from-conversations}

In many enterprise workflows, unstructured conversations---like meeting
transcripts, support calls, or financial briefings---contain valuable
insights that need to be distilled into structured formats for
downstream processing.

In this use case, we aim to extract **structured summaries** from raw
conversational transcripts. Specifically, we want the model to extract:

-   A concise **summary** of the conversation
-   Relevant **keywords**
-   **Named entities** such as people, organizations, and dates
-   The **sentiment** of the discussion

### Why is this useful?

Rather than just generating freeform text, this task helps transform
unstructured inputs into **machine-readable structured outputs**, making
it suitable for:

-   Reporting dashboards\
-   Automated indexing and retrieval\
-   Analytics pipelines\
-   Compliance and auditing tools

### PDF Data

You can find the pdf data here:
[seed_data/financial_call_transcripts](seed_data/financial_call_transcripts)



## 🧑‍🏫 Step 1: Set Up the Teacher Model {#-step-1-set-up-the-teacher-model}

This demo expects an openai compatible endpoint. You can use your
favorite inference server like vLLM, HFInferenceServer, LlamaStack, etc.
For more details on how to setup an inference server using vLLM, please
refer to the [README](README.md).

For this demo we will use Llama-3.3-70B-Instruct as our teacher model.

#### Let\'s test the connection



``` python
from openai import OpenAI

openai_api_key = "EMPTY"  # replace with your inference server api key
openai_api_base = (
    "http://0.0.0.0:8000/v1"  # replace with your inference server endpoint
)


client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

models = client.models.list()
teacher_model = models.data[0].id

# Test the connection with a simple completion
response = client.chat.completions.create(
    model=teacher_model,
    messages=[{"role": "user", "content": "Hello!"}],
    temperature=0.0,
    max_tokens=10,
)
completion = response.choices[0].message.content

print(f"Connection successful! {teacher_model}: {completion}")
```


``` python
from datasets import load_dataset

seed_data = load_dataset(
    "json", data_files="seed_data/financial_call_transcripts.jsonl", split="train"
)

seed_data[0]
```


```text
    {'conversation_id': 'c47a92e006b54d014a79b447528c55a7',
     'pdf_path': 'seed_data/financial_call_transcripts/c47a92e006b54d014a79b447528c55a7.pdf'}
```



## Setting up the pipeline

In this section, we walk through a pipeline designed to process
financial call transcripts (in PDF format) and extract structured
insights using LLM-powered blocks. This is a classic example of
transforming unstructured text into structured JSON for downstream
analysis.

This YAML defines a flow that reads PDF transcripts, parses them into
text, and then uses LLMs to extract:

-   ✅ Summary of the transcript
-   ✅ Key topics or keywords
-   ✅ Mentioned named entities (people, organizations, locations, etc.)
-   ✅ Overall sentiment of the call

All of these are combined into a final structured json_output.

``` mermaid
graph LR
    A[PDF Transcript] --> B[parse_transcript<br/>DoclingParsePDF]
    B --> C[conversation]
    C --> D[add_question<br/>AddStaticValue]
    D --> E[gen_summary<br/>LLMBlock]
    E --> F[gen_keywords<br/>LLMBlock]
    F --> G[gen_named_entities<br/>LLMBlock]
    G --> H[gen_sentiment<br/>LLMBlock]

    H --> I[format_json<br/>JSONFormat]
    I --> J[json_output]
```


``` python
# Lets look at the skills flow we will be using

import yaml

# flows/grounded_summary_extraction.yaml
with open("flows/grounded_summary_extraction.yaml", "r") as f:
    flow = yaml.safe_load(f)
print(yaml.dump(flow, indent=2))
```

```yaml
- block_config:
    block_name: parse_transcript
    output_column: conversation
    pdf_path_column: pdf_path
  block_type: DoclingParsePDF
- block_config:
    block_name: add_question
    column_name: question
    static_value: Extract summary, keywords, named entities, and sentiment from the
      transcript and return in JSON format.
  block_type: AddStaticValue
- block_config:
    block_name: gen_summary
    config_path: ../prompts/summary.yaml
    model_id: meta-llama/Llama-3.3-70B-Instruct
    output_cols:
    - summary
  block_type: LLMBlock
- block_config:
    block_name: gen_keywords
    config_path: ../prompts/keywords.yaml
    model_id: meta-llama/Llama-3.3-70B-Instruct
    output_cols:
    - keywords
  block_type: LLMBlock
- block_config:
    block_name: gen_named_entities
    config_path: ../prompts/named_entities.yaml
    model_id: meta-llama/Llama-3.3-70B-Instruct
    output_cols:
    - named_entities
  block_type: LLMBlock
- block_config:
    block_name: gen_sentiment
    config_path: ../prompts/sentiment.yaml
    model_id: meta-llama/Llama-3.3-70B-Instruct
    output_cols:
    - sentiment
  block_type: LLMBlock
- block_config:
    block_name: format_json
    output_column: json_output
  block_type: JSONFormat
  drop_columns:
  - summary
  - keywords
  - named_entities
  - sentiment
```



``` python
# Load the flow
flow = Flow(client).get_flow_from_file("flows/grounded_summary_extraction.yaml")

# Initialize the synthetic data generator
generator = SDG(
    flows=[flow],
)
```



``` python
seed_data = seed_data.select(range(10)) # note: this is just for demo purposes, in practice you can use the entire dataset
generated_data = generator.generate(seed_data)
```


``` python
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
import random
import json


rand_idx = random.choice(range(len(generated_data)))
# Your data
data = json.loads(generated_data[rand_idx]["json_output"])

# Convert to JSON string with indentation for pretty printing
json_str = json.dumps(data, indent=2)

# Create syntax highlighted JSON
syntax = Syntax(json_str, "json", theme="github", line_numbers=False)

# Display it inside a panel
console = Console()
console.print(Panel(syntax, title="📊 Extracted Summary", expand=True))
```

```text
╭───────────────────────────────────────────── 📊 Extracted Summary ──────────────────────────────────────────────╮
│ {                                                                                                               │
│   "summary": "The company reported a 23% growth in revenue over the fourth quarter last year, driven by growth  │
│   "keywords": [                                                                                                 │
│     "Sparsentan",                                                                                               │
│     "FSGS",                                                                                                     │
│     "NEPTUNE Consortium",                                                                                       │
│     "FDA",                                                                                                      │
│     "DUET trial",                                                                                               │
│     "Cholbam",                                                                                                  │
│     "bile acid synthesis disorders",                                                                            │
│     "NephCure",                                                                                                 │
│     "statistical modeling",                                                                                     │
│     "patient enrollment",                                                                                       │
│     "commercial portfolio",                                                                                     │
│     "rare disease patients",                                                                                    │
│     "sales force expansion",                                                                                    │
│     "financial growth",                                                                                         │
│     "regulatory pathway",                                                                                       │
│     "Phase II meeting"                                                                                          │
│   ],                                                                                                            │
│   "named_entities": {                                                                                           │
│     "organizations": [                                                                                          │
│       "NEPTUNE Consortium",                                                                                     │
│       "NephCure",                                                                                               │
│       "Food and Drug Administration",                                                                           │
│       "FDA"                                                                                                     │
│     ],                                                                                                          │
│     "people": [                                                                                                 │
│       "Bill",                                                                                                   │
│       "Steve",                                                                                                  │
│       "Laura"                                                                                                   │
│     ],                                                                                                          │
│     "locations": null,                                                                                          │
│     "dates": [                                                                                                  │
│       2016,                                                                                                     │
│       2017,                                                                                                     │
│       "November",                                                                                               │
│       "January",                                                                                                │
│       "Q2",                                                                                                     │
│       "Q4"                                                                                                      │
│     ]                                                                                                           │
│   },                                                                                                            │
│   "sentiment": "Positive"                                                                                       │
│ }                                                                                                               │
╰──────
```




## ✅ Conclusion {#-conclusion}

In this section, we demonstrated how to construct an end-to-end pipeline
that transforms unstructured PDF transcripts into structured JSON
insights using modular building blocks. By parsing the document,
prompting an LLM to extract specific features, and formatting the
results, we've created a scalable workflow for financial document
analysis---or any use case involving long-form text.

## 📝 Homework: Extend the Pipeline {#-homework-extend-the-pipeline}

Your task is to add a new block that extracts a different kind of
structured insight from the conversation context.

Some examples include:

-   🧩 Risk factors mentioned in the call
-   📊 Numerical metrics (e.g., revenue, margin, headcount)
-   📌 Action items or decisions discussed

Once you've done that, you'll have taken the first step toward custom
skill authoring, opening the door to richer document understanding
tailored to your own domain needs.

