
``` python
%load_ext autoreload
%autoreload 2
```


``` python
# Standard
import random

# Third Party
from datasets import Dataset, load_dataset
from openai import OpenAI
from rich import print
from rich.console import Console
from rich.panel import Panel
from sklearn.metrics import classification_report
import yaml

# First Party
from sdg_hub.flow import Flow
from sdg_hub.sdg import SDG
from blocks import *
```



# Annotation with Yahoo Answers

In this tutorial, you'll learn how to create your own custom data
generation pipeline using SDG Hub. Our goal is to build a skill that
teaches a language model how to **annotate user-generated text** with
topic labels --- specifically using the [Yahoo Answers Topics
dataset](https://huggingface.co/datasets/yahoo_answers_topics) from
Hugging Face.

We'll go step by step through a progressively improving pipeline. Each
stage builds on the previous one, giving you a practical sense of how
synthetic labeling can evolve from simple heuristics to highly
customized and reliable data generation.

### 🔍 Understand the Task {#-understand-the-task}

Before we write any prompts or code, we'll take time to understand what
we want the model to learn. For this exercise, the task is **topic
annotation** --- assigning one of ten possible categories (e.g.,
\"Science & Mathematics\", \"Sports\", \"Politics & Government\") to a
user-submitted question or paragraph.

### 🛠️ Build a Basic Annotation Pipeline {#️-build-a-basic-annotation-pipeline}

We'll start by creating a minimal pipeline that takes a small number of
seed examples and uses them to generate topic labels on the unlabeled
Yahoo Answers data. This will use default prompts and simple scoring
logic to simulate how annotation works.

### 🎯 Improve with Better Examples {#-improve-with-better-examples}

Next, we'll refine the pipeline by enhancing the **seed examples**.
Better examples = better generations. You'll see how even a small change
in phrasing, structure, or label clarity can dramatically improve output
quality.

### ✏️ Customize with Your Own Prompts {#️-customize-with-your-own-prompts}

Finally, we'll show you how to take full control by writing your own
prompts from scratch. This allows you to inject task-specific
instructions, formatting rules, or even domain tone --- enabling the
model to generalize better and reduce noise in the generated labels.

Let's get started by loading a sample of the Yahoo dataset and
identifying what task we want the model to learn.


``` python
dataset = load_dataset("fancyzhx/ag_news")

train_data = dataset["train"].shuffle(seed=42).select(range(500))
test_data = dataset["test"].shuffle(seed=42).select(range(100))

# map the labels to the category names
label_map = train_data.features['label'].names

train_data = train_data.map(lambda x: {"category": label_map[x["label"]]})
test_data = test_data.map(lambda x: {"category": label_map[x["label"]]})
```


``` python
# Group examples by category
examples_by_category = {}
for item in train_data:
    category = item['category']
    if category not in examples_by_category:
        examples_by_category[category] = []
    examples_by_category[category].append(item['text'])

# Print one example from each category in a panel
for category, examples in examples_by_category.items():
    print(Panel(examples[0], title=f"Category: {category}", expand=False))
```

```{=html}
╭──────────────────────────────────────────────── Category: World ────────────────────────────────────────────────╮
│ Bangladesh paralysed by strikes Opposition activists have brought many towns and cities in Bangladesh to a      │
│ halt, the day after 18 people died in explosions at a political rally.                                          │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
</pre>
```


```{=html}
╭─────────────────────────────────────────────── Category: Sports ────────────────────────────────────────────────╮
│ Desiring Stability Redskins coach Joe Gibbs expects few major personnel changes in the offseason and wants to   │
│ instill a culture of stability in Washington.                                                                   │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
</pre>
```


```{=html}
╭────────────────────────────────────────────── Category: Sci/Tech ───────────────────────────────────────────────╮
│ U2 pitches for Apple New iTunes ads airing during baseball games Tuesday will feature the advertising-shy Irish │
│ rockers.                                                                                                        │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
</pre>
```


```{=html}
╭────────────────────────────────────────────── Category: Business ───────────────────────────────────────────────╮
│ Economy builds steam in KC Fed district The economy continued to strengthen in September and early October in   │
│ the Great Plains and Rocky Mountain regions covered by the Tenth Federal Reserve District, the Federal Reserve  │
│ Bank of Kansas City said Wednesday.                                                                             │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
</pre>
```




## Simple Data Annotation Pipeline

In this section, we'll create our **first working pipeline** to perform
annotation using a language model. The goal is to simulate how the model
can annotate raw user queries with topic labels using a minimal
configuration.

### Recap: How Pipelines Work

``` mermaid
flowchart LR
    A[Flows] --> B[Blocks] --> C[Prompts]
    C --> D[Synthetic Data!]
```



### Flow

Below is a minimal flow that uses a single LLMBlock to annotate raw
questions. We're using guided decoding with a fixed label set to keep
model outputs controlled and consistent.

``` yaml
- block_type: LLMBlock
  block_config:
    block_name: simple_annotation
    config_path: ../prompts/simple_annotation.yaml
    model_id: meta-llama/Llama-3.3-70B-Instruct
    output_cols:
      - output
  gen_kwargs:
    temperature: 0
    max_tokens: 5
    extra_body:
      guided_choice:
        - World
        - Sports
        - Business
        - Sci/Tech
```

### Prompt

This prompt teaches the model to take in a freeform query and return a
single topic label. Since we're using guided decoding, we're keeping the
format minimal and relying on constrained sampling to enforce label
consistency.

``` yaml
system: null
introduction: "Task Description: Data Annotation"
principles: null
examples: null
generation: |
  Here is the query for annotation:
  {{text}}
start_tags: [""]
end_tags: [""]
```

This prompt passes the raw text to the model with minimal guidance ---
think of it as a baseline to test how much the model already understands
the task when constrained to a limited label set.

### What This Does

-   Loads a batch of input text (e.g., Yahoo Answers questions)
-   Passes each query into the prompt under the {{text}} template
-   Uses guided decoding (via xgrammar) to force the output to be one of
    the specified topic labels
-   Outputs predictions in the output column

This is the simplest version of an annotation pipeline --- no examples,
no complex prompting --- just a structured flow powered by modular
blocks.

Let\'s test it out!



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



### Run the Simple Annotation Pipeline


``` python
# Load the flow
flow = Flow(client).get_flow_from_file("flows/simple_annotation.yaml")

# Initialize the synthetic data generator
simple_annotation_generator = SDG(
    [Pipeline(flow)],
)
```



``` python
generated_data = simple_annotation_generator.generate(test_data)
```



### Evaluation

Now that we've generated synthetic topic labels using our annotation
pipeline, it's time to evaluate how well the model performed. The goal
is to compare the predicted labels against the **true labels** from the
Yahoo Answers dataset using standard classification metrics.

We'll use `sklearn.metrics.classification_report`, which provides
precision, recall, F1-score, and support for each class.


``` python
print(classification_report(generated_data["category"], generated_data["output"]))
```

```{=html}
              precision    recall  f1-score   support

    Business       0.59      0.72      0.65        32
    Sci/Tech       1.00      0.05      0.10        19
      Sports       0.62      0.93      0.75        27
       World       0.40      0.36      0.38        22

    accuracy                           0.57       100
   macro avg       0.65      0.52      0.47       100
weighted avg       0.64      0.57      0.51       100
```




## Improving Results with Examples and Custom Prompts

Our initial pipeline used a **zero-shot approach** --- the model was
given the task, a fixed label set, and some input text, but **no
examples of how to perform the task**. While this baseline gives us a
useful starting point, it has clear limitations:

-   The model may rely on generic heuristics or surface patterns that
    don't generalize well.
-   It can confuse similar categories (e.g., \"World\" vs. \"Business\")
    without knowing how they\'re typically used.
-   Without guidance, the model may underperform on edge cases or
    ambiguous queries.

### Why Examples Matter

In-context examples act as **training demonstrations** --- they teach
the model how to think, how to respond, and how to structure its output.

With even a few high-quality seed examples, we can:

-   **Disambiguate confusing labels** by showing contrasting cases
-   **Guide tone and formatting**, especially for structured tasks
-   **Bias the model toward higher precision** by anchoring it to gold
    examples

Think of examples as the foundation for aligning the model to your task
--- they provide **task intent**, **style**, and **semantic anchors**
for generation.

### What We'll Do Next

We'll now enhance our prompt by adding **4 examples** that cover a
variety of labels from the Yahoo dataset. These examples will be
inserted into the prompt file used by the `LLMBlock`.

You'll then rerun the same pipeline and compare the results --- and see
how a few carefully chosen examples can dramatically improve both
**accuracy** and **label consistency**.

``` yaml
system: You are an expert text classifier trained to label questions from online forums. 
introduction: "Task Description: You will be given a text and you need to annotate it with one of the following categories: World, Sports, Business, Sci/Tech, Economy"
principles: |
  Please follow these rules when performing the classification:
  - Focus on the main topic, not peripheral mentions
  - Choose the most specific applicable category
  - Only choose category label per question
examples: |
  Text: Bangladesh paralysed by strikes Opposition activists have brought many towns and cities in Bangladesh to a halt, the day after 18 people died in explosions at a political rally.
  Category: World

  Text: Desiring Stability Redskins coach Joe Gibbs expects few major personnel changes in the offseason and wants to instill a culture of stability in Washington.
  Category: Sports

  Text: A Cosmic Storm: When Galaxy Clusters Collide Astronomers have found what they are calling the perfect cosmic storm, a galaxy cluster pile-up so powerful its energy output is second only to the Big Bang.
  Category: Sci/Tech

  Text: Economy builds steam in KC Fed district The economy continued to strengthen in September and early October in the Great Plains and Rocky Mountain regions covered by the Tenth Federal Reserve District, the Federal Reserve Bank of Kansas City said Wednesday.
  Category: Economy

generation: |
  Here is the query for annotation:
  
  Text: {{text}}
  Category: 
  
start_tags: [""]
end_tags: [""]
```

### Run the Pipeline with Examples and Custom Prompts


``` python
# Load the flow
flow = Flow(client).get_flow_from_file("flows/detailed_annotation.yaml")

# Initialize the synthetic data generator
detailed_annotation_generator = SDG( 
    flows=[flow],
)
```



``` python
generated_data = detailed_annotation_generator.generate(test_data)
```



### Evaluation {#evaluation}


``` python
print(classification_report(generated_data["category"], generated_data["output"]))
```

```text
              precision    recall  f1-score   support

    Business       0.76      0.81      0.79        32
    Sci/Tech       0.67      0.74      0.70        19
      Sports       0.90      1.00      0.95        27
       World       1.00      0.68      0.81        22

    accuracy                           0.82       100
   macro avg       0.83      0.81      0.81       100
weighted avg       0.83      0.82      0.82       100
```




## ✅ Summary: What You've Learned {#-summary-what-youve-learned}

In this tutorial, you built a complete data annotation pipeline ---
starting from scratch and evolving into a robust, high-accuracy system.
Along the way, you explored the core principles of skill-building with
large language models.

### 🚀 What's Next? {#-whats-next}

-   Extend the pipeline further! - Add an evaluation step
-   Try it out on your own data!

