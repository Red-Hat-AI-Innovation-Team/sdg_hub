```python
%load_ext autoreload
%autoreload 2
```

# Markdown Table Manipulation Skills

Modern enterprises rely on structured data to drive decisions across operations, HR, product, and sales. But real-world data is rarely clean. Tables are often inconsistent, incomplete, or split across sources. Analysts and engineers spend countless hours fixing formatting issues, merging data, and applying business logic manually.

This project teaches a language model how to understand, clean, manipulate, and reason over markdown tables—turning messy or fragmented tabular inputs into clean, analysis-ready markdown outputs that can be dropped into dashboards, reports, or downstream systems.

We do this using InstructLab, by providing examples of real-world table tasks that require reasoning, formatting precision, and consistency.


These tasks develop a model’s capabilities in:
* Cleaning: Normalize inconsistent entries (e.g., “USA”, “U.S.”, “United States” → “US”)
* Filtering: Apply multi-column conditions (e.g., Progress < 60% and Budget < 100k)
* Computation: Derive new columns from formulas (e.g., Adjusted Revenue = Revenue × Multiplier)
* Joining: Merge data from multiple markdown tables using a shared key
* Classification: Infer labels like “Seniority” from unstructured title strings
* Standardization: Enforce markdown formatting, column consistency, and data integrity


Task Examples Include:
1.	Applying Rules Across Columns

    Derive new columns by applying conditional logic to existing data. Examples include assigning statuses, flags, or labels based on thresholds, categories, or rule-based formulas.

2.  Cleaning and Normalizing Tabular Data

    Standardize inconsistent entries such as location names, department labels, or text casing to ensure consistency across rows—essential for reliable analysis or joins.

3. 	Inferring Categorical Labels from Text
    
    Extract or classify values (e.g., seniority, department type, status) from semi-structured strings using pattern recognition or keyword-based inference.

4. 	Merging and Enriching Data Across Tables
    
    Perform relational joins using keys like ID or Region, and enhance the dataset by combining fields from multiple sources.

5.  Retrieval and Filtering From the Table

    Retrieve specific rows or columns based on conditions or patterns, useful for ad-hoc queries or filtering out irrelevant data.


## 🧑‍🏫 Step 1: Set Up the Teacher Model

This demo expects an openai compatible endpoint. You can use your favorite inference server like vLLM, HFInferenceServer, LlamaStack, etc. For more details on how to setup an inference server using vLLM, please refer to the [README](README.md).

For this demo we will use Llama-3.3-70B-Instruct as our teacher model.

#### Let's test the connection


```python
from openai import OpenAI

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

    Connection successful! meta-llama/Llama-3.3-70B-Instruct: Hello. How can I help you today?


## ✍️ Step 2: Provide Custom Examples

As outlined in the LAB paper, the first step is to provide a small number of **seed examples** (typically 5) to bootstrap the skill. These examples are passed into the generation pipeline as input and are stored in a `.jsonl` file.

For this demo, we’ll use the pre-populated seed file located at: [mdtable_manipulation_seeds.jsonl](examples/instructlab/skills/sample_data/mdtable_manipulation_seeds.jsonl)

Lets open the file and explore a row: 


```python
from datasets import load_dataset

# Load the seed dataset
seed_data = load_dataset("json", data_files="sample_data/mdtable_manipulation_seeds.jsonl", split="train")

# Display the first example
seed_data[0]
```

#### Expected Output

```
{'task_description': 'Perform advanced table manipulation, including cleaning, joining, inferring values, and computing derived columns based on complex rules.',
 'seed_question': '| Project | Budget (USD) | Progress (%) | Phase     |\n|---------|--------------|--------------|-----------|\n| Mercury | 120000       | 85           | Alpha     |\n| Venus   | 95000        | 78           | Alpha     |\n| Earth   | 87000        | 52           | Beta      |\n| Mars    | 110000       | 45           | Beta      |\n| Jupiter | 78000        | 66           | Gamma     |\n\nQuestion: Add a new column \'Status\' using these rules:\n- If Budget > 100k and Progress ≥ 80%, mark as "On Track"\n- If Budget < 100k but Progress ≥ 60%, mark as "Risk: Underfunded"\n- If Progress < 60%, mark as "Behind"',
 'seed_response': '| Project | Budget (USD) | Progress (%) | Phase | Status            |\n|---------|--------------|--------------|--------|-------------------|\n| Mercury | 120000       | 85           | Alpha | On Track          |\n| Venus   | 95000        | 78           | Alpha | Risk: Underfunded |\n| Earth   | 87000        | 52           | Beta  | Behind            |\n| Mars    | 110000       | 45           | Beta  | Behind            |\n| Jupiter | 78000        | 66           | Gamma | Risk: Underfunded |'}
 ```


```python
print(seed_data[0]["seed_question"])
```

#### Expected Output

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

## 🚀 Step 3: Generate Synthetic Data

Now that we have our seed data ready, we can use LAB’s Skill Data Generator to create **high-quality synthetic training examples** for our custom skill.

This step leverages a predefined **flow configuration** that encodes how seed examples are expanded — by generating new contexts, questions, and responses, and filtering them for quality.

In this demo, we'll use the `synth_skills.yaml` flow, which follows LAB's grounded generation pattern (question → response).


```python
from sdg_hub.flow import Flow
from sdg_hub.pipeline import Pipeline
from sdg_hub.sdg import SDG

# Path to the skill generation flow configuration
flow_path = "flows/synth_skills.yaml"

# Load the flow
flow = Flow(client).get_flow_from_file(flow_path)

# Initialize the synthetic data generator
generator = SDG(
    [Pipeline(flow)],
)
```

At this point, the generator is ready to run the full pipeline — including context generation, question/response generation, evaluation, and filtering — to produce a synthetic dataset that can be used for fine-tuning or skill bootstrapping.

In the next step, we’ll run this pipeline and inspect the generated outputs. 

> ⚠️ Note: This would take a variable amount of time depending on the hardware used.


```python
generated_data = generator.generate(seed_data)
```

## 🔍 Step 4: Explore and Validate the Synthetically Generated Data

Once the skill generation pipeline has been executed, the output is a set of **synthetically generated examples** — new context-question-response triples that follow the same structure as the seed data but are expanded and refined by the teacher model.

Below is an example of one generated entry:

> ⚠️ Note: You might not get the same expected output as the one below because the generation is random.


```python
import random 

rand_idx = random.choice(range(len(generated_data)))
generated_data[rand_idx]
```

#### Expected Output


```
{'task_description': 'Perform advanced table manipulation, including cleaning, joining, inferring values, and computing derived columns based on complex rules.',
 'seed_question': '| Name         | Role Title                  |\n|--------------|-----------------------------|\n| Nia Kapoor   | Lead Software Engineer      |\n| Omar Ghali   | UX Designer                 |\n| Lin Zhu      | Intern - AI Research        |\n| Carlos Pena  | Data Specialist             |\n| Tessa Morgan | Principal Product Manager   |\n\nQuestion: Add a column called \'Seniority\' where:\n- Titles with \'Lead\', \'Principal\', or \'Head\' → "Senior"\n- Titles with \'Engineer\', \'Specialist\', \'Designer\', or \'Analyst\' → "Mid"\n- Titles with \'Intern\' or \'Trainee\' → "Junior"',
 'seed_response': '| Name         | Role Title                  | Seniority |\n|--------------|-----------------------------|-----------|\n| Nia Kapoor   | Lead Software Engineer      | Senior    |\n| Omar Ghali   | UX Designer                 | Mid       |\n| Lin Zhu      | Intern - AI Research        | Junior    |\n| Carlos Pena  | Data Specialist             | Mid       |\n| Tessa Morgan | Principal Product Manager   | Senior    |',
 'question': '| Student ID | Name         | Grade | GPA |\n|------------|--------------|-------|-----|\n| 1          | Emily Chen   | 10    | 3.8 |\n| 2          | David Lee    | 11    | 3.5 |\n| 3          | Sophia Patel | 12    | 3.9 |\n| 4          | Jackson Kim  | 10    | 3.2 |\n| 5          | Olivia Brown | 11    | 3.6 |\n\nQuestion: Create a new column called \'Academic Status\' where:\n- GPA greater than 3.7 → "Honors"\n- GPA between 3.3 and 3.7 → "Passing"\n- GPA less than 3.3 → "Probation"',
 'response': '| Student ID | Name         | Grade | GPA | Academic Status |\n|------------|--------------|-------|-----|-----------------|\n| 1          | Emily Chen   | 10    | 3.8 | Honors          |\n| 2          | David Lee    | 11    | 3.5 | Passing         |\n| 3          | Sophia Patel | 12    | 3.9 | Honors          |\n| 4          | Jackson Kim  | 10    | 3.2 | Probation       |\n| 5          | Olivia Brown | 11    | 3.6 | Passing         |'}
 ```


```python
print(generated_data[rand_idx]['question'])
```

#### Expected Output

| Student ID | Name         | Grade | GPA |
|------------|--------------|-------|-----|
| 1          | Emily Chen   | 10    | 3.8 |
| 2          | David Lee    | 11    | 3.5 |
| 3          | Sophia Patel | 12    | 3.9 |
| 4          | Jackson Kim  | 10    | 3.2 |
| 5          | Olivia Brown | 11    | 3.6 |

Question: Create a new column called 'Academic Status' where:
- GPA greater than 3.7 → "Honors"
- GPA between 3.3 and 3.7 → "Passing"
- GPA less than 3.3 → "Probation"


```python
print(generated_data[rand_idx]['response'])
```

#### Expected Output


| Student ID | Name         | Grade | GPA | Academic Status |
|------------|--------------|-------|-----|-----------------|
| 1          | Emily Chen   | 10    | 3.8 | Honors          |
| 2          | David Lee    | 11    | 3.5 | Passing         |
| 3          | Sophia Patel | 12    | 3.9 | Honors          |
| 4          | Jackson Kim  | 10    | 3.2 | Probation       |
| 5          | Olivia Brown | 11    | 3.6 | Passing         |

## 🏁 Conclusion

In this notebook, we demonstrated how to teach a custom skill to a language model using the InstructLab Skill Data Generator (SDG). Starting from a small set of seed examples, we walked through the full synthetic data generation pipeline — including context creation, question generation, response synthesis, evaluation, and filtering.

We explored a real-world use case: **Manipulating Markdown Tables**, and showed how the LAB framework can automate the generation of high-quality, instructional training data at scale.

This approach is especially powerful for procedural or domain-specific tasks where labeled data is scarce but consistent task logic can be modeled. With just a few carefully curated seed examples, you can unlock scalable skill creation and push new capabilities into LLMs with minimal manual effort.

You’re now ready to use these synthetic examples for Fine-tuning small models! 

Next steps? 

* Try changing the parameters of the flow to see how the generated data changes (e.g. change the `num_samples` or try generating with different temperature)
* Try adapting this pipeline to your own task, domain, or format — whether it’s triaging support tickets, extracting structured data, or following domain-specific workflows. The skills are yours to create.
