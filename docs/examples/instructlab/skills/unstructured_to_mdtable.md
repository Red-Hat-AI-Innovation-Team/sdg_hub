```python
%load_ext autoreload
%autoreload 2
```

## Teaching a Language Model the Skill: Unstructured Text → Markdown Table

Company X receives large volumes of user feedback through support emails, in-app surveys, and app store reviews. These messages often contain valuable product insights, but the content is unstructured and difficult to analyze at scale.

To streamline internal workflows, an AI team at Company X wants to teach a language model how to convert raw user feedback into structured markdown tables. These tables summarize key topics, user sentiment, and issues in a format that’s easy to scan, report, or push into dashboards and tracking systems.

We can do this using InstructLab!

#### 🧾 Example Input and Output

📥 Input (Unstructured Feedback)
```
Hey team — I’ve been using the new update for about a week now.

Couple of things:
- The dark mode is awesome, great job!
- But the loading time after login feels slower than before. Not a deal breaker but noticeable.
- I also noticed that the calendar widget doesn’t update properly if I change time zones.

Overall, I love where this is going. Just needs a few tweaks.
```
📤 Output (Markdown Table)

| Feature           | Feedback                                                               | Sentiment |
|------------------|------------------------------------------------------------------------|-----------|
| Dark Mode        | Works well, user is satisfied.                                          | Positive  |
| Login Performance| Loading time after login is slower than previous version.               | Negative  |
| Calendar Widget  | Doesn't update correctly when time zones change.                        | Negative  |
| Overall          | User is happy with the direction of the product, but suggests tweaks.   | Positive  |

## 🧑‍🏫 Step 1: Serving Teacher Model

This demo expects an openai compatible endpoint. You can use your favorite inference server like vLLM, HFInferenceServer, LlamaStack, etc. For more details on how to setup an inference server using vLLM, please refer to the [README](README.md).

For this demo we will use meta-llama/Llama-3.3-70B-Instruct as our teacher model.

#### Let's test the connection


```python
from openai import OpenAI

openai_api_key = "EMPTY"
openai_api_base = "http://0.0.0.0:8000/v1"


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

For this demo, we’ll use the pre-populated seed file located at: [mdtable_seeds.jsonl](examples/instructlab/skills/sample_data/mdtable_seeds.jsonl)

Lets open the file and explore a row: 


```python
from datasets import load_dataset

# Load the seed dataset
seed_data = load_dataset("json", data_files="sample_data/unstructured_to_mdtable_seeds.jsonl", split="train")

# Display the first example
seed_data[0]
```

#### Expected Output

```
{'task_description': 'Convert the following unstructured user feedback into a structured markdown table.',
 'seed_question': "Been using the new dashboard for a few days. It's way faster than the previous one, really appreciate the snappy filters. But export to CSV seems broken — nothing happens when I click it. Also, dark mode resets every time I log in.\n\nI would like to convert the above feedback into a markdown table with columns for Feature, Feedback and Sentiment.",
 'seed_response': "| Feature           | Feedback                                                           | Sentiment |\n|------------------|--------------------------------------------------------------------|-----------|\n| Dashboard        | Much faster than previous version, filters are responsive.         | Positive  |\n| Export to CSV    | Clicking the export button doesn't trigger a download.             | Negative  |\n| Dark Mode        | Resets to light mode on login.                                     | Negative  |"}
 ```

## 🚀 Step 3: Generate Synthetic Data

Now that we have our seed data ready, we can use LAB’s Skill Data Generator to create **high-quality synthetic training examples** for our custom skill.

This step leverages a predefined **flow configuration** that encodes how seed examples are expanded — by generating new contexts, questions, and responses, and filtering them for quality.

In this demo, we'll use the `synth_grounded_skills.yaml` flow, which follows LAB's grounded generation pattern (context → question → response).


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


```python
import random 

rand_idx = random.choice(range(len(generated_data)))
generated_data[rand_idx]
```




    {'task_description': 'Convert the following unstructured user feedback into a structured markdown table.',
     'seed_question': 'Really love the new calendar UI. The drag-and-drop is intuitive. One issue: reminders don’t always sync between desktop and mobile. Also noticed tooltips sometimes cover buttons.\n\nPlease convert the above feedback into a markdown table with columns for Feature, Feedback and Sentiment.',
     'seed_response': '| Feature           | Feedback                                                             | Sentiment |\n|------------------|----------------------------------------------------------------------|-----------|\n| Calendar UI      | Drag-and-drop is intuitive and easy to use.                         | Positive  |\n| Reminders Sync   | Inconsistent between desktop and mobile devices.                   | Negative  |\n| Tooltips         | Occasionally block button access.                                   | Negative  |',
     'question': 'The customer service representative I spoke to was very helpful and resolved my issue quickly. However, the wait time to speak to someone was over an hour, which is unacceptable. I also noticed that the website has a lot of useful resources and tutorials, but the search function could be improved.\n\nPlease convert the above feedback into a markdown table with columns for Feature, Feedback, and Sentiment.',
     'response': '| Feature           | Feedback                                                             | Sentiment |\n|------------------|----------------------------------------------------------------------|-----------|\n| Customer Service | Representative was very helpful and resolved issue quickly.         | Positive  |\n| Wait Time        | Excessive wait time of over an hour to speak to someone.            | Negative  |\n| Website Resources| Many useful resources and tutorials available.                      | Positive  |\n| Search Function  | Could be improved for better user experience.                       | Negative  |'}



#### Expected Output

```
{'task_description': 'Convert the following unstructured user feedback into a structured markdown table.',
 'seed_question': 'Really love the new calendar UI. The drag-and-drop is intuitive. One issue: reminders don’t always sync between desktop and mobile. Also noticed tooltips sometimes cover buttons.\n\nPlease convert the above feedback into a markdown table with columns for Feature, Feedback and Sentiment.',
 'seed_response': '| Feature           | Feedback                                                             | Sentiment |\n|------------------|----------------------------------------------------------------------|-----------|\n| Calendar UI      | Drag-and-drop is intuitive and easy to use.                         | Positive  |\n| Reminders Sync   | Inconsistent between desktop and mobile devices.                   | Negative  |\n| Tooltips         | Occasionally block button access.                                   | Negative  |',
 'question': 'The customer service representative I spoke to was very helpful and resolved my issue quickly. However, the wait time to speak to someone was over an hour, which is unacceptable. I also noticed that the website has a lot of useful resources and tutorials, but the search function could be improved.\n\nPlease convert the above feedback into a markdown table with columns for Feature, Feedback, and Sentiment.',
 'response': '| Feature           | Feedback                                                             | Sentiment |\n|------------------|----------------------------------------------------------------------|-----------|\n| Customer Service | Representative was very helpful and resolved issue quickly.         | Positive  |\n| Wait Time        | Excessive wait time of over an hour to speak to someone.            | Negative  |\n| Website Resources| Many useful resources and tutorials available.                      | Positive  |\n| Search Function  | Could be improved for better user experience.                       | Negative  |'}
 ```

 Note: Since the generated data is random, the output will be different.


```python
print(generated_data[rand_idx]['question']), print(generated_data[rand_idx]['response'])
```

#### Expected Output

```
The customer service representative I spoke to was very helpful and resolved my issue quickly. However, the wait time to speak to someone was over an hour, which is unacceptable. I also noticed that the website has a lot of useful resources and tutorials, but the search function could be improved.

Please convert the above feedback into a markdown table with columns for Feature, Feedback, and Sentiment.


| Feature           | Feedback                                                             | Sentiment |
|------------------|----------------------------------------------------------------------|-----------|
| Customer Service | Representative was very helpful and resolved issue quickly.         | Positive  |
| Wait Time        | Excessive wait time of over an hour to speak to someone.            | Negative  |
| Website Resources| Many useful resources and tutorials available.                      | Positive  |
| Search Function  | Could be improved for better user experience.                       | Negative  |

```

## 🏁 Conclusion

In this notebook, we demonstrated how to teach a custom skill to a language model using the InstructLab Skill Data Generator (SDG). Starting from a small set of seed examples, we walked through the full synthetic data generation pipeline — including context creation, question generation, response synthesis, evaluation, and filtering.

We explored a real-world use case: **transforming unstructured user feedback into structured markdown tables**, and showed how the LAB framework can automate the generation of high-quality, instructional training data at scale.

This approach is especially powerful for procedural or domain-specific tasks where labeled data is scarce but consistent task logic can be modeled. With just a few carefully curated seed examples, you can unlock scalable skill creation and push new capabilities into LLMs with minimal manual effort.

You’re now ready to use these synthetic examples for Fine-tuning small models! 

Next steps? 

* Try changing the parameters of the flow to see how the generated data changes (e.g. change the `num_samples` or try generating with different temperature)
* Try adapting this pipeline to your own task, domain, or format — whether it’s triaging support tickets, extracting structured data, or following domain-specific workflows. The skills are yours to create.
