

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



# Markdown Table Manipulation Skills

Modern enterprises rely on structured data to drive decisions across
operations, HR, product, and sales. But real-world data is rarely clean.
Tables are often inconsistent, incomplete, or split across sources.
Analysts and engineers spend countless hours fixing formatting issues,
merging data, and applying business logic manually.

This project teaches a language model how to understand, clean,
manipulate, and reason over markdown tables---turning messy or
fragmented tabular inputs into clean, analysis-ready markdown outputs
that can be dropped into dashboards, reports, or downstream systems.

We do this using InstructLab, by providing examples of real-world table
tasks that require reasoning, formatting precision, and consistency.

These tasks develop a model's capabilities in:

-   Cleaning: Normalize inconsistent entries (e.g., "USA", "U.S.",
    "United States" → "US")
-   Filtering: Apply multi-column conditions (e.g., Progress \< 60% and
    Budget \< 100k)
-   Computation: Derive new columns from formulas (e.g., Adjusted
    Revenue = Revenue × Multiplier)
-   Joining: Merge data from multiple markdown tables using a shared key
-   Classification: Infer labels like "Seniority" from unstructured
    title strings
-   Standardization: Enforce markdown formatting, column consistency,
    and data integrity



## Task Examples

### 1. Applying Rules Across Columns {#1-applying-rules-across-columns}

Derive new columns by applying conditional logic to existing data.\
Examples include assigning statuses, flags, or labels based on
thresholds, categories, or rule-based formulas.

```{=html}
<details>
<summary><strong>📄 Example</strong> — Deriving a column</summary>

<br/>

<table>
  <thead>
    <tr>
      <th>Project</th>
      <th>Budget (USD)</th>
      <th>Progress (%)</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>Atlas</td><td>120000</td><td>85</td></tr>
    <tr><td>Beacon</td><td>95000</td><td>78</td></tr>
    <tr><td>Chronos</td><td>87000</td><td>52</td></tr>
    <tr><td>Delta</td><td>110000</td><td>45</td></tr>
    <tr><td>Echo</td><td>78000</td><td>66</td></tr>
  </tbody>
</table>

<br/>

**🧠 Question:**  
Add a `Status` column using the following rules:

- If **Budget > 100,000** and **Progress ≥ 80** → `"On Track"`  
- If **Budget < 100,000** and **Progress ≥ 60** → `"Risk: Underfunded"`  
- If **Progress < 60** → `"Behind"`

<br/>

📤 **Expected Output**

<table>
  <thead>
    <tr>
      <th>Project</th>
      <th>Budget (USD)</th>
      <th>Progress (%)</th>
      <th>Status</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>Atlas</td><td>120000</td><td>85</td><td>On Track</td></tr>
    <tr><td>Beacon</td><td>95000</td><td>78</td><td>Risk: Underfunded</td></tr>
    <tr><td>Chronos</td><td>87000</td><td>52</td><td>Behind</td></tr>
    <tr><td>Delta</td><td>110000</td><td>45</td><td>Behind</td></tr>
    <tr><td>Echo</td><td>78000</td><td>66</td><td>Risk: Underfunded</td></tr>
  </tbody>
</table>

</details>
```



### 2. Cleaning and Normalizing Tabular Data {#2-cleaning-and-normalizing-tabular-data}

Standardize inconsistent entries such as location names, department
labels, or text casing to ensure consistency across rows---essential for
reliable analysis or joins.

```{=html}
<details>
<summary><strong>📄 Example</strong> - Cleaning and Normalizing Tabular Data </summary>

📥 **Input Table**

<table>
  <thead>
    <tr>
      <th>Employee ID</th>
      <th>Name</th>
      <th>Department</th>
      <th>Country</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>001</td><td>Alice Wong</td><td>sales</td><td>usa</td></tr>
    <tr><td>002</td><td>Ben Carter</td><td>HR</td><td>United States</td></tr>
    <tr><td>003</td><td>Carla Diaz</td><td>Sales</td><td>U.S.</td></tr>
  </tbody>
</table>

🧠 **Instruction**

Normalize 'Country' values (e.g., USA, U.S., United States → US) and standardize 'Department' to title case.

📤 **Expected Output**

<table>
  <thead>
    <tr>
      <th>Employee ID</th>
      <th>Name</th>
      <th>Department</th>
      <th>Country</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>001</td><td>Alice Wong</td><td>Sales</td><td>US</td></tr>
    <tr><td>002</td><td>Ben Carter</td><td>HR</td><td>US</td></tr>
    <tr><td>003</td><td>Carla Diaz</td><td>Sales</td><td>US</td></tr>
  </tbody>
</table>
</details>
```



### 3. Inferring Categorical Labels from Text {#3-inferring-categorical-labels-from-text}

Extract or classify values (e.g., seniority, department type, status)
from semi-structured strings using pattern recognition or keyword-based
inference.

```{=html}
<details>
<summary><strong>📄 Example</strong> - Inferring Categorical Labels from Text </summary>

📥 **Input Table**

<table>
  <thead>
    <tr>
      <th>Name</th>
      <th>Role Title</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>Nia Kapoor</td><td>Lead Software Engineer</td></tr>
    <tr><td>Omar Ghali</td><td>UX Designer</td></tr>
    <tr><td>Lin Zhu</td><td>Intern - AI Research</td></tr>
  </tbody>
</table>

🧠 **Instruction**

Add a 'Seniority' column based on keywords:
- 'Lead' → Senior
- 'Engineer', 'Designer' → Mid
- 'Intern' → Junior

📤 **Expected Output**

<table>
  <thead>
    <tr>
      <th>Name</th>
      <th>Role Title</th>
      <th>Seniority</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>Nia Kapoor</td><td>Lead Software Engineer</td><td>Senior</td></tr>
    <tr><td>Omar Ghali</td><td>UX Designer</td><td>Mid</td></tr>
    <tr><td>Lin Zhu</td><td>Intern - AI Research</td><td>Junior</td></tr>
  </tbody>
</table>
</details>
```



### 4. Merging and Enriching Data Across Tables {#4-merging-and-enriching-data-across-tables}

Perform relational joins using keys like ID or Region, and enhance the
dataset by combining fields from multiple sources.

```{=html}
<details>
<summary><strong>📄 Example</strong> - Merging and Enriching Data Across Tables </summary>

📥 **Input Table**

<table>
  <thead>
    <tr>
      <th>Rep</th>
      <th>Region ID</th>
      <th>Revenue</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>Alice</td><td>R1</td><td>85000</td></tr>
    <tr><td>Ben</td><td>R2</td><td>94000</td></tr>
  </tbody>
</table>

<table>
  <thead>
    <tr>
      <th>Region ID</th>
      <th>Country</th>
      <th>Multiplier</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>R1</td><td>US</td><td>1.2</td></tr>
    <tr><td>R2</td><td>EU</td><td>1.1</td></tr>
  </tbody>
</table>

🧠 **Instruction**

Join the two tables on 'Region ID' and compute 'Adjusted Revenue = Revenue × Multiplier'.

📤 **Expected Output**

<table>
  <thead>
    <tr>
      <th>Rep</th>
      <th>Region ID</th>
      <th>Country</th>
      <th>Revenue</th>
      <th>Multiplier</th>
      <th>Adjusted Revenue</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>Alice</td><td>R1</td><td>US</td><td>85000</td><td>1.2</td><td>102000</td></tr>
    <tr><td>Ben</td><td>R2</td><td>EU</td><td>94000</td><td>1.1</td><td>103400</td></tr>
  </tbody>
</table>
</details>
```



### 5. Retrieval and Filtering From the Table {#5-retrieval-and-filtering-from-the-table}

Retrieve specific rows or columns based on conditions or patterns,
useful for ad-hoc queries or filtering out irrelevant data.

```{=html}
<details>
<summary><strong>📄 Example</strong> - Retrieval and Filtering From the Table </summary>

📥 **Input Table**

<table>
  <thead>
    <tr>
      <th>Name</th>
      <th>Department</th>
      <th>Salary</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>Alice Wong</td><td>HR</td><td>72000</td></tr>
    <tr><td>Ben Carter</td><td>Sales</td><td>65000</td></tr>
    <tr><td>David Smith</td><td>HR</td><td>80000</td></tr>
    <tr><td>Erica Zhou</td><td>Sales</td><td>70000</td></tr>
  </tbody>
</table>

🧠 **Instruction**

Retrieve employees whose names start with 'D' or 'E', are in 'Sales' or 'HR', and have salary ≥ 70000.

📤 **Expected Output**

<table>
  <thead>
    <tr>
      <th>Name</th>
      <th>Department</th>
      <th>Salary</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>David Smith</td><td>HR</td><td>80000</td></tr>
    <tr><td>Erica Zhou</td><td>Sales</td><td>70000</td></tr>
  </tbody>
</table>
</details>
```



## From Seed Data to Synthetic Data

At the core of SDG Hub is a simple, composable workflow that transforms
your seed data into rich synthetic data using LLMs. To build these
workflows, you only need to understand three core concepts:

-   **Flows**: The blueprint for your data generation pipeline --- they
    define how blocks are chained and executed.
-   **Blocks**: The building blocks of your pipeline --- each block
    performs a specific task like generation, filtering, or
    transformation.
-   **Prompts**: The brain behind each block --- written in YAML,
    prompts define the task, tone, structure, and constraints for the
    LLM.

``` mermaid
flowchart LR
    A[Flows] --> B[Blocks] --> C[Prompts]
    C --> D[Synthetic Data!]
```

With just these three components, you can create powerful, scalable, and
fully customizable synthetic data pipelines.



## 🧑‍🏫 Step 1: Set Up the Teacher Model {#-step-1-set-up-the-teacher-model}

This demo expects an openai compatible endpoint. You can use your
favorite inference server like vLLM, HFInferenceServer, LlamaStack, etc.
For more details on how to setup an inference server using vLLM, please
refer to the [README](README.md).

For this demo we will use Llama-3.3-70B-Instruct as our teacher model.

#### Let\'s test the connection


``` python
openai_api_key = "EMPTY" # replace with your inference server api key
openai_api_base = "http://0.0.0.0:8000/v1" # replace with your inference server endpoint


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
    max_tokens=10
)
completion = response.choices[0].message.content

print(f"Connection successful! {teacher_model}: {completion}")
```



## ✍️ Step 2: Provide Custom Examples {#️-step-2-provide-custom-examples}

As outlined in the [LAB paper](https://arxiv.org/abs/2311.12850), the
starting point for customizing a new skill is to provide a small set of
**seed examples**. Seed examples show the model what good behavior looks
like by pairing inputs with ideal outputs, allowing the model to learn
patterns, structure, and formatting that generalize beyond the examples
themselves.

A strong seed example, regardless of domain, should:

✅ Clearly define the task context and expected behavior

✅ Provide a realistic, natural input that mimics what users or systems
would actually produce

✅ Include a high-quality output that fully satisfies the task
requirements---accurate, complete, and formatted correctly

✅ Minimize ambiguity: avoid examples where multiple interpretations are
possible without explanation

✅ Reflect diverse edge cases: cover a variety of structures, phrasings,
or difficulty levels to help the model generalize



These examples are specified in a YAML file---typically named
`qna.yaml`.

Each seed example consists of a `question` and its corresponding
`answer`. The `question` is usually a table or semi-structured data
followed by a natural language instruction, while the `answer`
represents the expected transformed output. Together, they demonstrate
the reasoning or manipulation expected of the model. These examples help
bootstrap the generation pipeline by setting the tone, structure, and
task constraints for further synthetic data generation.

Here's an example from the domain of **table manipulation**, where the
task is to derive a new column (`Status`) based on logical rules applied
to an existing table:

``` yaml
created_by: Red Hat AI Innovation Team
domain: Table Manipulation
task_description: |
  Perform advanced table manipulation, including cleaning, joining,
  inferring values, and computing derived columns based on complex rules.
seed_examples:
  - question: |
      | Project | Budget (USD) | Progress (%) | Phase     |
      |---------|--------------|--------------|-----------|
      | Mercury | 120000       | 85           | Alpha     |
      | Venus   | 95000        | 78           | Alpha     |
      | Earth   | 87000        | 52           | Beta      |
      | Mars    | 110000       | 45           | Beta      |
      | Jupiter | 78000        | 66           | Gamma     |

      Question: Add a new column 'Status' using these rules:
      - If Budget > 100k and Progress ≥ 80%, mark as "On Track"
      - If Budget < 100k but Progress ≥ 60%, mark as "Risk: Underfunded"
      - If Progress < 60%, mark as "Behind"
    answer: |
      | Project | Budget (USD) | Progress (%) | Phase  | Status            |
      |---------|--------------|--------------|--------|-------------------|
      | Mercury | 120000       | 85           | Alpha  | On Track          |
      | Venus   | 95000        | 78           | Alpha  | Risk: Underfunded |
      | Earth   | 87000        | 52           | Beta   | Behind            |
      | Mars    | 110000       | 45           | Beta   | Behind            |
      | Jupiter | 78000        | 66           | Gamma  | Risk: Underfunded |
```



This example demonstrates the reasoning logic required to augment the
original table. Note that the answer preserves the original structure
but adds a new computed column based on the instruction.

For this demo, we'll use a pre-populated YAML file located at:
`seed_data/table_manipulation_qna.yaml`

### 🔁 Convert to JSONL Format {#-convert-to-jsonl-format}

Dataflow between blocks and pipelines is handled using Hugging Face
Datasets, which are based on Arrow tables. To make the seed data
compatible with the generation pipeline, we need to convert the YAML
file into a JSONL dataset. This conversion involves extracting the
examples from the YAML file and converting them into a dataset.


``` python
def convert_yaml_to_jsonl(yaml_path):
    # Load YAML file
    with open(yaml_path, 'r') as f:
        yaml_data = yaml.safe_load(f)
    
    # Extract examples into list of dicts
    examples = []
    for example in yaml_data['seed_examples']:
        examples.append({
            'task_description': yaml_data['task_description'],
            'seed_question': example['question'],
            'seed_response': example['answer']
        })
    
    # Convert to HF Dataset
    dataset = Dataset.from_list(examples)
    return dataset

# Load and convert the seed data
seed_data = convert_yaml_to_jsonl('seed_data/table_manipulation_qna.yaml')



print(Panel(
    "\n\n".join(f"[bold]{k}:[/bold] \n\n{v}" for k,v in seed_data[0].items()),
    title="Seed Data Example"
))
```

```{=html}
╭─────────────────────────────────────────────── Seed Data Example ───────────────────────────────────────────────╮
│ <span style="font-weight: bold">task_description:</span>                                                                                               │
│                                                                                                                 │
│ Perform advanced table manipulation, including cleaning, joining,                                               │
│ inferring values, and computing derived columns based on complex rules.                                         │
│                                                                                                                 │
│                                                                                                                 │
│ <span style="font-weight: bold">seed_question:</span>                                                                                                  │
│                                                                                                                 │
│ | Project | Budget (USD) | Progress (%) | Phase     |                                                           │
│ |---------|--------------|--------------|-----------|                                                           │
│ | Mercury | 120000       | 85           | Alpha     |                                                           │
│ | Venus   | 95000        | 78           | Alpha     |                                                           │
│ | Earth   | 87000        | 52           | Beta      |                                                           │
│ | Mars    | 110000       | 45           | Beta      |                                                           │
│ | Jupiter | 78000        | 66           | Gamma     |                                                           │
│                                                                                                                 │
│ Question: Add a new column 'Status' using these rules:                                                          │
│ - If Budget &gt; 100k and Progress ≥ 80%, mark as "On Track"                                                       │
│ - If Budget &lt; 100k but Progress ≥ 60%, mark as "Risk: Underfunded"                                              │
│ - If Progress &lt; 60%, mark as "Behind"                                                                           │
│                                                                                                                 │
│                                                                                                                 │
│ <span style="font-weight: bold">seed_response:</span>                                                                                                  │
│                                                                                                                 │
│ | Project | Budget (USD) | Progress (%) | Phase  | Status            |                                          │
│ |---------|--------------|--------------|--------|-------------------|                                          │
│ | Mercury | 120000       | 85           | Alpha  | On Track          |                                          │
│ | Venus   | 95000        | 78           | Alpha  | Risk: Underfunded |                                          │
│ | Earth   | 87000        | 52           | Beta   | Behind            |                                          │
│ | Mars    | 110000       | 45           | Beta   | Behind            |                                          │
│ | Jupiter | 78000        | 66           | Gamma  | Risk: Underfunded |                                          │
│                                                                                                                 │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```




## 🚀 Step 3: Generate Synthetic Data {#-step-3-generate-synthetic-data}

Now that we have our seed data ready, we can use LAB's Skill Data
Generator to create **high-quality synthetic training examples** for our
custom skill.

This step leverages a predefined **flow configuration** that encodes how
seed examples are expanded --- by generating new contexts, questions,
and responses, and filtering them for quality.

In this demo, we\'ll use the `flows/generation/skills/synth_skills.yaml`
pipeline to generate synthetic data.

### Flows

``` mermaid
flowchart LR
     A[Seed Data] --> B[LLM Block<br/>gen_questions<br/>→ question]
     B --> C[LLM Block<br/>eval_questions<br/>→ evaluation, score]
     C --> D[FilterByValueBlock<br/>filter_questions<br/>score == 1.0]
     D --> E[LLM Block<br/>gen_responses<br/>→ response]
     E --> F[LLM Block<br/>evaluate_qa_pair<br/>→ evaluation, score]
     F --> G[FilterByValueBlock<br/>filter_qa_pair<br/>score >= 2.0]
     G --> H[Generated Data]
```

### Prompts

Prompts in SDG Hub are defined using structured YAML files that specify
how the model should behave during each stage of the generation process.
These prompt configurations are designed to guide the LLM with clarity,
precision, and stylistic consistency.

#### Prompt Structure:

-   system: Defines the assistant\'s role or identity to help maintain
    consistent tone and persona.
-   introduction: Provides task-level context to the LLM, including
    placeholders like {{num_samples}} or {{task_description}} which are
    replaced dynamically at runtime.
-   principles: A list of quality and formatting guidelines the model
    should follow while generating responses.
-   examples: A representative input-output sample that sets the pattern
    and style the LLM should emulate.
-   generation: The actual prompt instruction, combining context and
    principles to invoke generation.
-   start_tags / end_tags: Used to extract or delimit generated outputs
    cleanly

This format makes prompt engineering modular, readable, and
reproducible, which is essential for reliable synthetic data generation
workflows.

#### Example Prompt:

``` yaml
system: You are a highly intelligent and helpful AI Assistant specializing in generating well-crafted questions tailored to specific tasks.

introduction: |
  Your task is to generate {{num_samples}} diverse and well-structured questions for the following task:
  "{{task_description}}"

principles: |
  Please follow these strict guidelines when generating each question:
  * Each question must be directly related to the task description.
  * Use correct grammar, spelling, and punctuation.
  * Questions must be clear, natural-sounding, and human-like.
  * Do **not** include answers, explanations, or commentary—only the question.
  * Ensure **maximum diversity** in wording and perspective—no repetitive or template-based phrasing.
  * Each question must strictly follow the **exact same format and style** as shown in the example.
  * Do **not deviate** from the example structure in any way.
  * Wrap each question between `[Start of Question]` and `[End of Question]` tags.

examples: |
  To guide you, here is an example of a correctly formatted question:

  [Start of Question]
  {{seed_question}}
  [End of Question]

generation: |
  Now generate {{num_samples}} such questions.
  Ensure that each one is:
  - Fully relevant to the task description.
  - Consistent with the example format.
  - Clearly enclosed between [Start of Question] and [End of Question] tags.
  Do not include any content outside these tags.

start_tags: ["[Start of Question]"]
end_tags: ["[End of Question]"]
```



``` python
# Lets look at the skills flow we will be using

import yaml

# flows/grounded_summary_extraction.yaml
with open("../../../src/sdg_hub/flows/generation/skills/synth_skills.yaml", "r") as f:
    flow = yaml.safe_load(f)
print(yaml.dump(flow, indent=2))
```



``` yaml
- block_config:
    batch_kwargs:
      num_samples: 30
    block_name: gen_questions
    config_path: configs/skills/freeform_questions.yaml
    model_id: meta-llama/Llama-3.3-70B-Instruct
    output_cols:
    - question
  block_type: LLMBlock
  drop_duplicates:
  - question
- block_config:
    block_name: eval_questions
    config_path: configs/skills/evaluate_freeform_questions.yaml
    model_id: meta-llama/Llama-3.3-70B-Instruct
    output_cols:
    - evaluation
    - score
  block_type: LLMBlock
- block_config:
    batch_kwargs:
      num_procs: 8
    block_name: filter_questions
    convert_dtype: float
    filter_column: score
    filter_value: 1.0
    operation: operator.eq
  block_type: FilterByValueBlock
  drop_columns:
  - evaluation
  - score
  - num_samples
- block_config:
    block_name: gen_responses
    config_path: configs/skills/freeform_responses.yaml
    model_id: meta-llama/Llama-3.3-70B-Instruct
    output_cols:
    - response
  block_type: LLMBlock
- block_config:
    block_name: evaluate_qa_pair
    config_path: configs/skills/evaluate_freeform_pair.yaml
    model_id: meta-llama/Llama-3.3-70B-Instruct
    output_cols:
    - evaluation
    - score
  block_type: LLMBlock
- block_config:
    batch_kwargs:
      num_procs: 8
    block_name: filter_qa_pair
    convert_dtype: float
    filter_column: score
    filter_value: 2.0
    operation: operator.ge
  block_type: FilterByValueBlock
  drop_columns:
  - evaluation
  - score
```


``` python
# Load the flow
flow = Flow(client).get_flow_from_file("flows/generation/skills/synth_skills.yaml")

# Initialize the synthetic data generator
generator = SDG(
    flows=[flow],
)
```


``` python
generated_data = generator.generate(seed_data)
```







## 🔍 Step 4: Explore and Validate the Synthetically Generated Data {#-step-4-explore-and-validate-the-synthetically-generated-data}

Once the skill generation pipeline has been executed, the output is a
set of **synthetically generated examples** --- new
context-question-response triples that follow the same structure as the
seed data but are expanded and refined by the teacher model.

Below is an example of one generated entry:


``` python
import random
from rich.panel import Panel
from rich.console import Console

console = Console()
rand_idx = random.choice(range(len(generated_data)))

# Pretty print the generated examples using rich
example = generated_data[rand_idx]
console.print(Panel.fit(
    f"[bold orange1]Question:[/bold orange1]\n{example['question']}\n\n" 
    f"[bold green]Response:[/bold green]\n{example['response']}"
))
console.rule(style="bright_white")
```

```{=html}
╭─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ <span style="color: #ffaf00; text-decoration-color: #ffaf00; font-weight: bold">Question:</span>                                                                                                       │
│ **Transaction History**                                                                                         │
│ | Transaction ID | Customer ID | Transaction Date | Amount |                                                    │
│ |----------------|-------------|------------------|--------|                                                    │
│ | T001           | C001        | 2022-01-01       | 100.00 |                                                    │
│ | T002           | C002        | 2022-01-15       | 200.00 |                                                    │
│ | T003           | C003        | 2022-02-01       | 50.00  |                                                    │
│ | T004           | C004        | 2022-03-01       | 150.00 |                                                    │
│ | T005           | C005        | 2022-04-01       | 250.00 |                                                    │
│                                                                                                                 │
│ **Customer Information**                                                                                        │
│ | Customer ID | Name         | Email                | Country      |                                            │
│ |-------------|--------------|----------------------|--------------|                                            │
│ | C001        | John Smith   | john.smith@example   | United Kingdom|                                           │
│ | C002        | Emily Lee    | emily.lee@example    | USA          |                                            │
│ | C003        | David Kim    | david.kim@example    | South Korea  |                                            │
│ | C004        | Sophia Patel | sophia.patel@example | India        |                                            │
│ | C006        | Michael Brown| michael.brown@example| Australia    |                                            │
│                                                                                                                 │
│ Infer missing customer names based on a predefined list of common names, and calculate the total transaction    │
│ amount for each customer. Then, join the tables on Customer ID, and compute a new column for the customer's     │
│ transaction frequency based on a predefined formula. Drop any rows with missing customer information, and       │
│ normalize the country names by standardizing them (e.g., United Kingdom → UK).                                  │
│                                                                                                                 │
│ <span style="color: #008000; text-decoration-color: #008000; font-weight: bold">Response:</span>                                                                                                       │
│ | Customer ID | Name         | Email                | Country | Transaction ID | Transaction Date | Amount |    │
│ Total Amount | Transaction Frequency |                                                                          │
│ |-------------|--------------|----------------------|---------|----------------|------------------|--------|--- │
│ -----------|------------------------|                                                                           │
│ | C001        | John Smith   | john.smith@example   | UK      | T001           | 2022-01-01       | 100.00 |    │
│ 100.00       | 1                      |                                                                         │
│ | C002        | Emily Lee    | emily.lee@example    | US      | T002           | 2022-01-15       | 200.00 |    │
│ 200.00       | 1                      |                                                                         │
│ | C003        | David Kim    | david.kim@example    | South Korea | T003 | 2022-02-01       | 50.00  | 50.00    │
│ | 1                      |                                                                                      │
│ | C004        | Sophia Patel | sophia.patel@example | India    | T004           | 2022-03-01       | 150.00 |   │
│ 150.00       | 1                      |                                                                         │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```






## 💾 Save the generated data {#-save-the-generated-data}

``` python
generated_data.to_json("llama_generated_table_manipulation.jsonl", orient="records", lines=True)
```



## 🏁 Conclusion {#-conclusion}

In this notebook, we demonstrated how to teach a custom skill to a
language model using the InstructLab Skill Data Generator (SDG).
Starting from a small set of seed examples, we walked through the full
synthetic data generation pipeline --- including context creation,
question generation, response synthesis, evaluation, and filtering.

We explored a real-world use case: Manipulating Markdown Tables, and
showed how the LAB framework can automate the generation of
high-quality, instructional training data at scale.

This approach is especially powerful for procedural or domain-specific
tasks where labeled data is scarce but consistent task logic can be
modeled. With just a few carefully curated seed examples, you can unlock
scalable skill creation and push new capabilities into LLMs with minimal
manual effort.

You're now ready to use these synthetic examples for Fine-tuning small
models!

Next steps?

-   Try changing the parameters of the flow to see how the generated
    data changes (e.g. change the num_samples or try generating with
    different temperature)
-   Try adapting this pipeline to your own task, domain, or format ---
    whether it's triaging support tickets, extracting structured data,
    or following domain-specific workflows. The skills are yours to
    create.

