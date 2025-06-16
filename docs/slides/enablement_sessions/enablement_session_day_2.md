Slide-1
------------------
Agenda                                       

What we’ll
discuss today                                

► Quick Recap of Day 1
► Deep dive into various skills pipelines
► Building a pipeline for Structured Summary
► Hands-on Demo

2                                             [bottom-left page number]

[bottom-right]  ⬢ (stylized red fedora hat icon)  Red Hat


Speaker notes:

* **Opening Context**
  “To set the stage for today, here’s a quick agenda.”

* **Quick Recap of Day 1**
  “I’ll start with a brief refresher on yesterday’s key takeaways—just enough to make sure everyone is on the same page.”

* **Deep Dive into Skills Pipelines**
  “Next, we’ll zoom in on the different skills-generation pipelines introduced yesterday, looking at their individual components and decision points.”

* **Building a Pipeline for Structured Summary**
  “After that, we’ll walk through how to assemble a pipeline specifically targeted at producing structured summaries—highlighting the data flow, models used, and quality controls.”

* **Hands-on Demo**
  “Finally, we’ll get our hands dirty: I’ll demonstrate the pipeline live so you can see the tooling and workflow in action.”

* **Transition**
  “Please jot down any questions as we go; we’ll pause briefly after each section and have extra Q\&A time following the demo.”


Slide-2
------------------

Day 1

Use Case 1: Markdown Table Manipulation

[Left Table]
Student ID | Name         | Grade | GPA
----------------------------------------
1          | Emily Chen   | 10    | 3.8
2          | David Lee    | 11    | 3.5
3          | Sophia Patel | 12    | 3.9
4          | Jackson Kim  | 10    | 3.2
5          | Olivia Brown | 11    | 3.6

[Large gray arrow pointing right]

[Right Table]
Student ID | Name         | Grade | GPA | Academic Status
----------------------------------------------------------
1          | Emily Chen   | 10    | 3.8 | Honors
2          | David Lee    | 11    | 3.5 | Passing
3          | Sophia Patel | 12    | 3.9 | Honors
4          | Jackson Kim  | 10    | 3.2 | Probation
5          | Olivia Brown | 11    | 3.6 | Passing

Q: Create a new column called 'Academic Status'

Where:
  ● GPA greater than 3.7 → "Honors"
  ● GPA between 3.3 and 3.7 → "Passing"
  ● GPA less than 3.3 → "Probation"

Works with 1.5!

[Bottom-left] 3
[Bottom-right] Red Hat logo (red fedora hat icon + "Red Hat")


Speaker notes:

Alright, quick recap from Day 1!

This was one of the hands-on use cases we explored—Markdown Table Manipulation.

Here, we started with a basic student GPA table. The task was to add a new column called ‘Academic Status’, based on GPA thresholds:

GPAs over 3.7 were tagged 'Honors'

GPAs between 3.3 and 3.7 were tagged 'Passing'

GPAs below 3.3 were tagged 'Probation'

This example illustrates how to apply conditional logic to manipulate tabular data. It’s simple but powerful.

We used this as one of five table-related data generation tasks, and importantly — this already works out-of-the-box with version 1.5 of the pipeline, using a prebuilt configuration.

If you had time to try this out yesterday, great! If not, no worries—we’ll take a moment to answer any questions before diving into today’s content.


Slide-3
------------------

Day 1

Use Case #2: Unstructured to Structured

[Left box: Unstructured Document]
Hey team — I’ve been using the new update for about a week now.

Couple of things:
- The dark mode is awesome, great job!
- But the loading time after login feels slower than before. Not a deal breaker but noticeable.
- I also noticed that the calendar widget doesn’t update properly if I change time zones.

Overall, I love where this is going. Just needs a few tweaks.

[Large gray arrow pointing right]

[Right table]

Feature            | Feedback                                                         | Sentiment
------------------|------------------------------------------------------------------|-----------
Dark Mode         | Works well, user is satisfied.                                   | Positive
Login Performance | Loading time after login is slower than previous version.        | Negative
Calendar Widget   | Doesn't update correctly when time zones change.                 | Negative
Overall           | User is happy with the direction of the product, but suggests tweaks. | Positive

Does not work with 1.5!

[Bottom-left] 4
[Bottom-right] Red Hat logo (red fedora hat icon + "Red Hat")


Speaker notes:

Now let’s look at Use Case 2 — transforming unstructured text into a structured table.

This task is a bit more complex than the previous one. On the left side, we have a raw piece of user feedback—completely unstructured. The goal is to extract key features, summarize the relevant feedback, and assign a sentiment for each point.

On the right, you can see the desired structured format. Each row corresponds to a distinct feature mentioned in the unstructured text.

Here’s the key point: this task *does not work out of the box* in version 1.5. The current release only supports predefined pipelines, and none of those are designed to handle this type of format conversion.

So what did we do? We built a custom pipeline to support this. That includes specific parsing logic and data labeling strategies to map freeform text into a table like this one.

We’ll talk more about how we designed that customization later in the session.


Slide-4
------------------


Day 1

Going beyond 1.5

[Top row - simple workflow]
Collect Seed Examples → Setup iLab SDG → Generate

[Bottom row - expanded customizable workflow]
Collect Seed Examples → Define your Flow → Add Custom Blocks → Customize your Prompts → Generate

[The box "Setup iLab SDG" is enclosed in a red rounded rectangle, indicating it’s a key block in the default flow]

[The sequence: Define your Flow → Add Custom Blocks → Customize your Prompts is enclosed in a red dashed outline, showing an advanced or alternative path]

[Bottom-left] 5
[Bottom-right] Red Hat logo (red fedora hat icon + "Red Hat")



Speaker notes:

This diagram shows how we go beyond the default capabilities in version 1.5.

In the standard flow, you simply:
1. Collect some seed examples,
2. Plug them into the default pipeline (labeled here as “Setup iLab SDG”),
3. And generate data.

That’s fast and convenient for predefined use cases—but also limited in flexibility.

Now look at the expanded path on the bottom. This is how you unlock full customization:

- First, you define your own flow. That means building the exact step-by-step logic your use case requires.
- Then, you can add custom blocks. These let you chain stages, add logic, or clean up outputs with precision.
- Finally, you customize your prompts—tailor instructions, add specific examples, and tune how the LLM behaves.

This customizable workflow is how we handled more advanced tasks, like unstructured-to-structured transformation. It gives us the control and precision needed to go beyond what 1.5 can do out-of-the-box.


Slide-5
------------------

Understanding Contexts

Different Skills Pipeline

[Left Column – Basic Pipeline]
┌────────────┐
│ Question   │
├────────────┤
│ Answer     │
└────────────┘

[Middle Column – Grounded Context Pipeline]
┌────────────────────┐
│ Grounded Context   │
├────────────────────┤
│ Question           │
├────────────────────┤
│ Answer             │
└────────────────────┘

[Right Column – Generated Context Pipeline]
┌────────────────────┐
│ Generated Context  │
├────────────────────┤
│ Question           │
├────────────────────┤
│ Answer             │
└────────────────────┘

[Bottom-left] 6
[Bottom-right] Red Hat logo (red fedora hat icon + "Red Hat")


Speaker notes:

This slide illustrates three different types of skills pipelines we support.

The first one is the simplest — it only requires a question and an answer. You can think of this as a pure QA dataset with no context involved. It's straightforward to generate, and a good starting point for generic tasks.

The second pipeline introduces a grounded context — for example, customer-facing documentation or help center articles. From this context, we generate relevant questions and answers. This is more realistic for enterprise use cases, where responses need to align with source material.

The third version takes things one step further: instead of using real documents, we use model-generated context. That’s useful when we don’t have high-quality input documents, or when we want to explore new scenarios.

Each of these pipelines serves different goals, and your choice depends on what kind of skill data you need to generate.


Slide-6
------------------

Structured Summary

Use Case 3: Structured Summary

Conversation ID: c47a92e006b54d014a79b447528c55a7

Good morning. My name is Natalia, and I will be your conference operator today.

Also, please note that we will discuss certain non-GAAP financial measures in this call. Reconciliations on a GAAP basis for these measures are included in today’s press release.

In January, we made major progress in completing our transformation into a meaningfully less leveraged and much more focused consumer products company.

As the results of these -- as a result of these steps early this year, we are on track to achieve our leverage target of approximately 3.5x at the end of this fiscal year.

... and we began to see the effects of that late this November. We now must work closely with our retail customers to drive this business forward, despite the recent headwinds in the new housing markets, which, again, we began to see in November.

Now if we look at the full year fiscal '19 outlook and you can turn to Slide 8 for that.

delivering operational excellence in our manufacturing facilities and supply chain and providing all of this with exceptional customer service.

To be very blunt, the only thing we saw in the quarter was a little bit of weakness in HHI. So we expect a little bit more sales there.

These statements are based upon management’s current expectations, projections and assumptions and are by nature uncertain. Actual results may differ materially.

with materially increased financial strength and flexibility to drive our long-term growth ambitions.

[Extracted Summary box shown on the right, in JSON format:]
{
  "summary": "Spectrum Brands reported Q1 results in line with expectations, driven by progress in its transformation.",
  "keywords": [
    "Spectrum Brands",
    "deleveraging",
    "GAAP",
    "non-GAAP",
    "fiscal year guidance",
    "HHI",
    "Wi-Fi-enabled Halo Smart Locks",
    "advertising investments",
    "balance sheet strength",
    "free cash flow"
  ],
  "named_entities": {
    "organizations": [
      "Spectrum Brands",
      "Consumer Electronics Show"
    ],
    "people": [
      "Natalia"
    ],
    "locations": null,
    "dates": [
      "January",
      "November",
      "June",
      "September",
      "Q1",
      "fiscal '19"
    ]
  },
  "sentiment": "Positive"
}

[Bottom-left] 7
[Bottom-right] Red Hat logo (red fedora hat icon + "Red Hat")


Speaker notes:

This slide shows another common skills pipeline use case: structured summarization.

On the left is an excerpt from a customer-facing conversation—like a financial earnings call or a customer service transcript. It's long, dense, and mostly unstructured.

The task here is to extract a clean, structured summary from that content. On the right, you can see the result as a JSON object.

The model has:
- Generated a one-line summary of the call
- Extracted key terms and phrases into a "keywords" list
- Identified named entities, including people, organizations, and dates
- Tagged the overall sentiment as positive

This type of pipeline is especially valuable when you need to analyze or index large volumes of unstructured documents or calls.

We’ll explore how to build and customize this pipeline later in the session.


Slide-7
------------------

Structured Summary

This is your Structured Summary Flow

[Left side vertical flow – outlined red boxes]
Parse PDFs
Summary
Keywords
Named Entities
Sentiment
JSON Format

[Right side – Extracted Summary (JSON content)]

{
  "summary": "Spectrum Brands reported Q1 results in line with expectations, driven by progress in its transformation.",
  "keywords": [
    "Spectrum Brands",
    "deleveraging",
    "GAAP",
    "non-GAAP",
    "fiscal year guidance",
    "HHI",
    "Wi-Fi-enabled Halo Smart Locks",
    "advertising investments",
    "balance sheet strength",
    "free cash flow"
  ],
  "named_entities": {
    "organizations": [
      "Spectrum Brands",
      "Consumer Electronics Show"
    ],
    "people": [
      "Natalia"
    ],
    "locations": null,
    "dates": [
      "January",
      "November",
      "June",
      "September",
      "Q1",
      "fiscal '19"
    ]
  },
  "sentiment": "Positive"
}

[Bottom-left] 8
[Bottom-right] Red Hat logo (red fedora hat icon + "Red Hat")


Speaker notes:
This is the full structured summary flow—step by step.

First, we begin by parsing PDFs. This could be earnings reports, meeting transcripts, or any customer documentation.

Next, we extract a concise summary from the unstructured text.

Third, we identify and extract the most important keywords—terms that are central to the content.

Then, we perform named entity recognition—pulling out names of people, organizations, and dates.

After that, we analyze the overall sentiment conveyed in the text.

And finally, we assemble all the extracted information into a clean, structured JSON object.

This end-to-end flow enables powerful downstream applications: search indexing, analytics dashboards, or customer insight generation.


Slide-8
------------------

Structured Summary

[Left box – red outline]
Parse PDFs

[Center – Cartoon mascot]
Cartoon duck character holding a sheet of paper. This is the Docling mascot.

[Right – code snippet]

@BlockRegistry.register("DoclingParsePDF")
class DoclingParsePDF(Block):
    def __init__(self, ctx, pipe, block_name, pdf_path_column: str, output_column: str):
        super().__init__(ctx, pipe, block_name)
        self.pdf_path_column = pdf_path_column
        self.output_column = output_column
        self.converter = DocumentConverter()

    @staticmethod
    def _map_parse_pdf(samples, pdf_path_column, output_column, converter, num_proc=1):
        def parse_pdf(sample):
            pdf_path = sample[pdf_path_column]
            result = converter.convert(pdf_path)
            sample[output_column] = result.document.export_to_markdown()
            return sample

        return samples.map(parse_pdf, num_proc=num_proc)

    def generate(self, self, samples: Dataset) -> Dataset:
        samples = self._map_parse_pdf(
            samples, self.pdf_path_column, self.output_column, self.converter
        )
        return samples

[Bottom-left] 9
[Bottom-right] Red Hat logo (red fedor


Speaker notes:
To kick off the structured summary flow, we begin by parsing the source documents—typically PDFs.

We use a block called `DoclingParsePDF`. This is a custom wrapper around the open-source Docling tool, designed to work inside the InstructLab SDG pipeline.

On the right, you can see the implementation: 
- It initializes a converter using the `DocumentConverter()` class.
- The `_map_parse_pdf` method applies the converter to each input file, and exports the result to Markdown.
- Finally, the `generate` method hooks this into the pipeline so the processed data can flow downstream.

The duck mascot here represents Docling. It's a helpful visual cue that this is a tool focused on document understanding.

Once parsed, the text content from the PDF becomes the foundation for all subsequent steps—summarization, keyword extraction, and so on.


Slide-8
------------------

Structured Summary

[Left box – red outline]
Summary

[Right – YAML-style prompt content]

system: You are a highly capable AI Assistant that specializes in summarizing financial call transcripts with...

introduction: |
  Your task is to write a concise, well-structured **summary** of the following financial call transcript:
  Transcript Input: {{conversation}}

principles: |
  Please follow these guiding principles when generating the summary:
  * Focus on key financial results, strategic priorities, forward-looking statements, and market sentiment.
  * Avoid copying lines directly from the transcript; paraphrase into a fluent narrative.
  * Keep the summary within 4–5 sentences.
  * Use a neutral and professional tone appropriate for financial reporting.
  * Do not include any greetings, explanations, or comments about the summary itself.
  * Wrap the output between the tags: [Start of Summary] and [End of Summary].

examples: |
  Here is an example of a high-quality financial summary:

  [Start of Summary]
  In its Q2 earnings call, Nexora Technologies reported a 12% year-over-year revenue increase driven by stron...
  [End of Summary]

generation: |
  Now generate a new summary following the same structure and principles.
  Begin your output with [Start of Summary] and end with [End of Summary].
  Do not include any additional text outside these tags.

start_tags: ["[Start of Summary]"]
end_tags: ["[End of Summary]"]

[Bottom-left] 10
[Bottom-right] Red Hat logo (red fedora hat icon + "Red Hat")


Speaker notes:

This is an example of a prompt definition used for generating financial call summaries.

This file would typically be called `prompt.yaml`, and it's used to configure the structured summary generation pipeline.

It includes:

- A **system prompt** that defines the role of the assistant.
- An **introduction**, clearly stating the task and referencing the transcript input.
- A set of **principles** that ensure consistency and quality — such as focusing on financial results, avoiding verbatim copying, and maintaining a professional tone.
- A real **example** of what a good summary should look like.
- And finally, a **generation section** that instructs the model to follow the format and use tags like `[Start of Summary]` and `[End of Summary]`.

These structured prompts are critical to achieving consistent outputs, especially in enterprise or compliance-sensitive contexts.


Slide - 9
------------

Structured Summary

[Left box – red outline]
Keywords

[Right – YAML-style prompt content]

system: You are a highly capable AI Assistant that specializes in extracting precise and relevant keywords f...

introduction: |
  Your task is to extract a list of clear, relevant **keywords** from the following financial call transcript:
  Transcript Input: {{conversation}}

principles: |
  Please follow these guiding principles when generating the keywords:
  * Select keywords that reflect core themes: financial performance, business strategy, products/services, m...
  * Include company names, product lines, financial terms, and strategic actions (e.g., acquisitions, divest...
  * Avoid generic words like “company,” “business,” or “earnings” unless paired with specific qualifiers.
  * Return keywords as a flat list (not nested, no extra formatting).
  * Only return the top 10 keywords.
  * Wrap the list between the tags: [Start of Keywords] and [End of Keywords].

examples: |
  Here is an example of a good keyword list:

  [Start of Keywords]
  Q2 earnings, revenue growth, cloud services, operating margin, international expansion, M&A strategy, cost...
  [End of Keywords]

generation: |
  Now extract keywords following the same structure and principles.
  Begin your output with [Start of Keywords] and end with [End of Keywords].
  Do not include any additional text outside these tags.

start_tags: ["[Start of Keywords]"]
end_tags: ["[End of Keywords]"]

[Bottom-left] 11
[Bottom-right] Red Hat logo (red fedora hat icon + "Red Hat")


Speaker notes:

This slide shows the configuration file for generating keywords from transcripts—another core step in the structured summary pipeline.

As you can see, this `prompt.yaml` file contains:

- A **system prompt** establishing the assistant’s role in keyword extraction.
- An **introduction** clearly defining the input and expected task.
- A detailed list of **principles**:
  - Focus on high-signal financial and strategic terms.
  - Avoid vague or generic phrases.
  - Return exactly 10 concise, flat keywords.
- A concrete **example** of what the expected output should look like.
- A **generation block** that instructs the model to follow the same logic and wrap the results in tags.

These tags—`[Start of Keywords]` and `[End of Keywords]`—are crucial for downstream parsing and validation.

This format ensures consistency and precision in how keyword data is extracted from financial conversations.



Slide - 10
------------

Structured Summary

[Left box – red outline]
Named Entities

[Right – YAML-style prompt content]

system: You are a highly capable AI Assistant that specializes in extracting named entities from financial an...

introduction: |
  Your task is to extract key **named entities** from the following financial call transcript:
  Transcript Input: {{conversation}}

principles: |
  Please follow these guiding principles when extracting the named entities:
  * Organize entities into four categories: organizations, people, locations, and dates.
  * Include only clearly identifiable and real entities – do not infer or hallucinate.
  * Group the entities under appropriate headings in a clean, flat list.
  * Do not include duplicates or generic terms (e.g., "the company").
  * Wrap the output between the tags: [Start of Named Entities] and [End of Named Entities].

examples: |
  Here is an example of well-structured named entity output:

  [Start of Named Entities]
  organizations:
    – Nexora Technologies
    – Goldman Sachs
    – AWS
  people:
    – Maria Chen
    – Jonathan Reyes
  locations:
    – San Francisco
    – Southeast Asia
  dates:
    – Q2 2025
    – July 18, 2025
  [End of Named Entities]

generation: |
  Now extract named entities following the same structure and principles.
  Begin your output with [Start of Named Entities] and end with [End of Named Entities].
  Do not include any additional text outside these tags.

start_tags: ["[Start of Named Entities]"]
end_tags: ["[End of Named Entities]"]

[Bottom-left] 12
[Bottom-right] Red Hat logo (red fedora hat icon + "Red Hat")


Speaker notes:

This slide shows the configuration used to extract named entities from a financial call transcript.

The goal here is to identify real, non-generic named entities and organize them into four categories:

- Organizations (e.g., companies or institutions)
- People (e.g., executives or mentioned individuals)
- Locations (e.g., markets, cities, or regions)
- Dates (e.g., fiscal quarters, specific report dates)

This YAML-based prompt includes:
- A clear task introduction
- Strict principles to avoid hallucinated or vague entries
- An example demonstrating the expected structure
- And clearly marked start and end tags for downstream processing

Using structured formatting like this helps make entity extraction outputs reliable and machine-readable, which is critical when building pipelines that feed into analytics, reporting, or search systems.


Slide - 11
--------

Structured Summary

[Left box – red outline]
Sentiment

[Right – YAML-style prompt content]

system: You are a highly capable AI Assistant that specializes in analyzing and summarizing sentiment from f...

introduction: |
  Your task is to assess and extract the overall **sentiment** expressed in the following financial call transcript:
  Transcript Input: {{conversation}}

principles: |
  Please follow these guiding principles when analyzing sentiment:
  * Provide one of the following sentiment labels: Positive, Neutral, or Negative.
  * Base your judgment on the overall tone, outlook, and performance discussed in the transcript.
  * Consider indicators like revenue growth, forward-looking guidance, confidence in execution, or expression...
  * Do not explain your answer or include additional commentary.
  * Wrap the sentiment label between the tags: [Start of Sentiment] and [End of Sentiment].

examples: |
  Here is an example of sentiment extraction:

  [Start of Sentiment]
  Positive
  [End of Sentiment]

generation: |
  Now analyze the input and return a sentiment label following the same structure and principles.
  Begin your output with [Start of Sentiment] and end with [End of Sentiment].
  Do not include any additional text outside these tags.

start_tags: ["[Start of Sentiment]"]
end_tags: ["[End of Sentiment]"]

[Bottom-left] 13
[Bottom-right] Red Hat logo (red fedora hat icon + "Red Hat")


Speaker notes:

This slide covers the sentiment analysis prompt used in the structured summary pipeline.

The model’s task is to assign a single overall sentiment label — Positive, Neutral, or Negative — to a financial call transcript.

The prompt includes:
- An intro that defines the task
- A clear list of principles:
  - Focus on tone, outlook, and key financial signals
  - Use judgment based on overall direction, not isolated statements
  - No reasoning or extra commentary — just the label
- An example illustrating expected output format
- Tag instructions for [Start of Sentiment] and [End of Sentiment]

This structure ensures that sentiment annotations are concise, uniform, and easily parsable by downstream tools.



Slide - 12
--------

Structured Summary

[Left box – red outline]
JSON Format

[Right – code block titled "🧱 JSONFormat Block"]

# SPDX-License-Identifier: Apache-2.0

# First Party
from instructlab.sdg.registry import BlockRegistry
from instructlab.sdg.blocks.block import Block
from datasets import Dataset
import yaml
import json

@BlockRegistry.register("JSONFormat")
class JSONFormat(Block):
    def __init__(self, ctx, pipe, block_name, output_column: str):
        super().__init__(ctx, pipe, block_name)
        self.output_column = output_column

    @staticmethod
    def _parse_named_entities(raw_text):
        try:
            parsed = yaml.safe_load(raw_text)
            return {
                "organizations": parsed.get("organizations", []) if isinstance(parsed, dict) else [],
                "people": parsed.get("people", []) if isinstance(parsed, dict) else [],
                "locations": parsed.get("locations", []) if isinstance(parsed, dict) else [],
                "dates": parsed.get("dates", []) if isinstance(parsed, dict) else [],
            }
        except Exception:
            return {
                "organizations": None,
                "people": None,
                "locations": None,
                "dates": None,
            }

    @staticmethod
    def _map_format_json(samples, output_column, num_proc=1):
        def format_json(sample):
            json_output = {
                "summary": sample.get("summary", None),
                "keywords": None,
                "named_entities": {
                    "organizations": None,
                    "people": None,
                    "locations": None,
                    "dates": None,
                },
                "sentiment": sample.get("sentiment", None),
            }
        return samples.map(format_json, num_proc=num_proc)

[Bottom-left] 14
[Bottom-right] Red Hat logo (red fedora hat icon + "Red Hat")


Speaker notes:


This is the final step in the structured summary pipeline—assembling all components into a clean JSON format.

This custom block, called `JSONFormat`, is registered with InstructLab and implemented in Python. It consolidates the outputs from previous blocks: summary, keywords, named entities, and sentiment.

Key parts:
- The `_parse_named_entities()` method safely extracts and validates YAML-based entity structures.
- The `format_json()` function organizes all fields into a structured JSON schema.
- If any field is missing or malformed, it defaults to `None` to avoid breaking the pipeline.

This block ensures that downstream systems—like dashboards, APIs, or search indexes—receive consistent, machine-readable output.

This final output is what drives automation, insight generation, and traceable analysis across many enterprise workflows.



Slide - 13
--------


Structured Summary

[Left-side vertical stack – red outlined boxes]
Parse PDFs  
Summary  
Keywords  
Named Entities  
Sentiment  
JSON Format

[Right – code block titled “🛠️ SDG Pipeline (YAML)”]

version: "1.0"
blocks:
  - name: parse_transcript
    type: DoclingParsePDF
    config:
      pdf_path_column: pdf_path
      output_column: conversation

  - name: add_question
    type: AddStaticValue
    config:
      column_name: question
      static_value: >
        Extract summary, keywords, named entities, and sentiment from the transcript and return in JSON fo...

  - name: gen_summary
    type: LLMBlock
    config:
      config_path: ../prompts/summary.yaml
    output_cols:
      - summary

  - name: gen_keywords
    type: LLMBlock
    config:
      config_path: ../prompts/keywords.yaml
    output_cols:
      - keywords

  - name: gen_named_entities
    type: LLMBlock
    config:
      config_path: ../prompts/named_entities.yaml
    output_cols:
      - named_entities

  - name: gen_sentiment
    type: LLMBlock
    config:
      config_path: ../prompts/sentiment.yaml
    output_cols:
      - sentiment

  - name: format_json
    type: JSONFormat
    config:
      output_column: json_output
    drop_columns:
      - summary
      - keywords
      - named_entities
      - sentiment

[Bottom-left] 15
[Bottom-right] Red Hat logo (red fedora hat icon + "Red Hat")



Speaker notes:

This slide brings everything together. It shows the complete YAML pipeline for structured summary generation using InstructLab SDG.

The flow is modular, with each block performing one specific task:

1. **parse_transcript**: Uses the DoclingParsePDF block to extract text from the PDF and store it as a “conversation.”
2. **add_question**: Adds a guiding static instruction that tells the LLM what to extract—summary, keywords, entities, sentiment.
3. **gen_summary**: Uses a prompt file (summary.yaml) to generate a concise narrative.
4. **gen_keywords**: Applies a separate prompt (keywords.yaml) to pull out high-signal terms.
5. **gen_named_entities**: Extracts structured entities like people, organizations, dates using named_entities.yaml.
6. **gen_sentiment**: Classifies the overall tone using sentiment.yaml.
7. **format_json**: Collects all the generated components and formats them into a single JSON output for downstream use.

Each block reads from or writes to clearly defined columns. The final step drops intermediate columns and emits a clean `json_output`.

This flow is reproducible, extensible, and aligns with enterprise-grade structured generation requirements.



## **Skills Training Enablement \- Day 2 \- 2025/05/13 07:46 CDT \- Transcript 2**


# **Transcript**

# **Day 2 Presentation Transcript \- Part 1 (Cleaned)**

## **Opening Discussion**

**AI Innovation Team:** We're saying you have to wait for RHEL AI 3.0 \- that's not really true. If you have a RHEL AI instance, you can clone iLab SDG and run everything. Everything will run. Just so that we're on the same page, these notebooks will execute fantastically well on our AI instance.

And so to Jeff's point yesterday, right, we do want to have this conversation. We can have more than a conversation \- we can actually create examples. And if you create something nice, please contribute back. That'll be great. But it will run on RHEL AI instances. Basically, you can use the Python module and run all of them instead of using it through the InstructLab CLI. So just clone the repo and execute the notebook. You're good to go.

Okay, I think with that we should be great. So as Akash mentioned, today we're going to look into this particular pipeline of how we can go from grounded context and generate some question-answers based on it. So let's look at the use case.

## **Structured Summary Use Case**

**AI Innovation Team:** So we have some transcripts, and this is coming from a dataset. Basically what it does is it's from S\&P 500, and it's basically the quarterly meetings they have discussing some companies. So there are notes on it, and what we're looking for is to sort of get this particular way of structured summary.

If you look at it, it has a couple of fields. So we have summary, which is a succinct representation of the meeting transcript. Then we are looking for certain keywords from the meeting, and we're looking for some named entities. So we want to know if there were any names of prominent people mentioned, or any locations, dates.

And then finally, we're trying to get the sentiment of the overall meeting. Was it negative? Something like that. Why do we want this? Maybe we want to enable some sort of a following workflow or a tool which will send out automated emails or some other enterprise workflows which will do this.

So if you ask a language model to generate this exact structure, even a large teacher model could have some issues with it. Let's maybe break this down separately. So you can generate summary, keywords, named entities, sentiment \- all of this independently.

## **Building the Pipeline Flow**

**AI Innovation Team:** And we would need to somehow parse the PDFs and then finally format everything to JSON. If you realize what we did right now on the left is basically creating a flow. This is just a pictorial representation of it. So let's dive deeper into how we can sort of implement individual blocks of this flow.

So to begin with, to parse the PDFs, you're going to use the help of our favorite \- Docling. This is already supported and built with RHEL AI, so you can start using it. What we're going to do is introduce a block using a decorator where it takes the PDFs, converts it \- this file might be a good place to open and sort of say in iLab SDG this is a block.

So I hope that is clear, right? Of course you can just parse it, but maybe this will help with the intuition of why we call it a flow. As long as you can wrap any Pythonic function in this particular fashion and have it register with the block registry decorator, it becomes part of your flow.

So you can generalize the hell out of this and have any arbitrary transformation. The benefit is that the whole pipeline becomes automated. So you won't need to do sort of the hand-holding of converting the PDFs to markdown and then starting a generation pipeline, but rather you can go from PDFs to your final synthetic data in just one seamless workflow. Just to make sure that we are on the same page.

## **Advanced Pipeline Considerations**

**AI Innovation Team:** \[At 5:00\] Totally optional. You can do it separately, or you can in fact \- now if you're thinking about it ahead \- let's say certain things that after parsing gets from Docling gets messed up, and you want to create a pipeline that takes care of it, runs certain kind of checks, right? Or let's say you want to do parsing and also want to run the document through a vision language model to do a cross-checking.

So the PDF goes into Docling. The PDF also goes into a VLM, and then you want the language model to compare the output and reconcile \- like I'm just talking about very complicated pipelines.

I know Docling has been a pain point. I know parsing doesn't work perfectly, but the reason we are showing iLab SDG and things like this is to show you how you can create your own automated pipeline that can use any open source model. So you can use BLIP-2's vision model or anything else, have it also parse a thing, write a logic that compares them, and that becomes your data parsing pipeline. And you're not just restricted to Docling \- you can pretty much, if you find a library which works better on your particular set of documents, you can use that library. All you need to do is, as Akash mentioned, wrap it as a block and register it.

That way it becomes part of your flow. Great. So in this session we'll be using Docling.

## **Summary Generation Block**

**AI Innovation Team:** All right. So that's the first block in our flow. Next block is basically generating the summaries. For this we're going to just use our existing LLM blocks because this is just going to be a model-generated summary. All we need to do is configure the prompts for that block. So it's structured in this classical way of system, principles, examples, and generation. What we did was we went ahead and sort of systematized every single aspect of the structure.

So in system, you mention that this language model specializes in generating these succinct summaries for a financial call transcript, and you give it some principles. You say, "Okay, keep the summary short \- only four to five sentences. Focus on financial results, strategic priorities, guidance statements, market sentiment." You're asking it to use a neutral tone to do the summarization and to not include any sort of greetings or explanation or comments in the summary.

And you can also provide an example of what kind of summary you're looking for.

## **Keywords Extraction Block**

**AI Innovation Team:** Next, we're going to look into the block for extracting keywords from the transcript. Similarly, this is going to be model-generated. So, this will also be an LLM block. All you need to do is configure your prompt. So in this case, we are looking for only the top five to 10 keywords from the meeting transcript, and we're asking it to avoid generic terms \- "business," "earnings," all of these \- because for sure these are going to be mentioned quite a lot in the transcript.

So you can simply add as little or as much principle as you want to make sure that the model behaves exactly how you want.

## **Named Entities Extraction Block**

**AI Innovation Team:** Next we're going to extract named entities \- the same ritual. This is going to be an LLM block. So all you need to do is configure the prompt. But if you look at it, we are looking for some sort of structure in this named entity extraction. What we are saying is we're asking the model to extract categories. So if it mentioned any organization, people, location or dates, and we also ask it to sort of structure it in a YAML fashion.

So we provide an example here on how the model should respond and how it should structure itself.

## **Sentiment Analysis Block**

**AI Innovation Team:** Finally, let's say we want to generate sentiment of the meeting transcript \- also going to be an LLM block because it's going to be model-generated. All you need to do is customize your prompt.

**AI Innovation Team:** \[At 10:00\] And you mention that you're looking for a sentiment from the transcript \- whether it's going to be positive or negative or neutral. So you can think of it as some sort of a classification task.

## **JSON Formatting Block**

**AI Innovation Team:** Great. Finally, we want to format all of this into JSON to look something like this. So instead of letting a model do this where it might hallucinate, since you already have individual components, you can basically write a Python function to format everything into JSON. And in order to use Python functions in our flows, all you need to do is wrap it with a block decorator and it becomes part of your flow.

So maybe this one is another example \- because I think somebody was asking, "Hey, could you use a dedicated NER system?" Absolutely. Anything that is Pythonic and you want to make it part of a scalable pipeline that flows in a linear fashion, the answer is yes. And so if you want to use spaCy or NLTK, any of the libraries like that, you can. Yesterday we were asking, "Hey, could we use classifiers and small models?" Absolutely. If it's written in scikit-learn or whatever your favorite predictive AI packages are, if it's Pythonic, you can use it.

## **Q\&A Discussion**

**Participant:** I think the number of few-shot examples is still five \- that's what we recommend, yeah.

**audience 3:** And for folks that \- if we happen to talk to data science customers \- what would be the reasoning? What guidance could we offer them in terms of why that's kind of the canonical number? Because there are a few kinds...

**AI Innovation Team:** Yeah, I think it's a good question. The true answer is it depends on the use case, but if the average case is five, language models are good at interpolation. So five is generally a good number, and the number comes from \- if you look at, I think, six months ago when we still used to do five-shot evaluation of models, that's kind of where it comes from. People found that at five examples, language models tend to get the pattern right. So they would give five examples and then let's say you do have a setup in which you just like have too much diversity and you have to capture all that, right? All those different use cases \- again, in the iLab SDG, you can go beyond that. There isn't really a lot of these limits that you heard of \- they were specific implementation limitations in the platform. One of the reasons why we are decoupling these things is that there were never a limitation in the actual SDG mechanism, right? So you can provide one or you can provide 15\. The guidance is around five, but as the diversity of your task increases, you should provide representative examples of the different possible types.

And like we mentioned yesterday, the constraint we have on the 250 to 500 tokens in the context \- that is also sort of an artificial one which we have in RHEL AI 1.5. In theory, you can give longer context as well.

## **Technical Dependencies Question**

**Speaker:** Hey, this is a very good one. So Max asked \- and I think he had to drop off \- how would we provide third-party dependencies if we declare additional dependencies in the custom block? Right, so currently the way it works is you can install any libraries for your use case, but if you want to package this flow and sort of productionize it, you would have to contribute that back into iLab SDG.

# **Day 2 Presentation Transcript \- Part 2 (Cleaned)**

## **Seed Data and Pipeline Overview**

**AI Innovation Team:** \[At 30:00\] So in terms of seed data, what we have is basically a dataset which shows the conversation ID \- something like this \- and a path to the actual PDF. This is what we're going to feed into a data generation pipeline, and the pipeline's going to take care of parsing the PDF, converting it to markdown using that as the grounded context, and then generating summaries, keywords, all of that step by step, and finally formatting everything into a neat JSON and giving it back to you.

So pictorially, this is what the flow would look like. You're starting from a PDF transcript. We're using the new block which we added using Docling where we are parsing the PDFs to convert from PDF to markdown text, and then you're adding the static question which we use. So the question is going to be fixed. You're saying the question is going to be something like "convert this into a summary and these are the fields I'm looking for in my summary and they all need to be formatted in JSON." This is a fixed question. So all your data samples will have this. Then we have a couple of LLM blocks. One for summary, one to generate keywords, named entities.

Basically what we did was instead of giving this as a huge task to the teacher model, we've broken it down into simpler tasks. That way we can get finer controls and also better performance. Finally, we have another block. This is purely Pythonic where you're taking all these individual components and formatting them to a JSON, which will give you the final JSON output.

This is what the flow would look like in YAML.

## **Q\&A: Data Generation vs Processing Pipeline**

**audience 1:** You've got a hand raised. Sorry, no. Go ahead.

**audience 7:** So look at \- from what I understand, this is a processing pipeline. It's not actually generating anything, is it? Other than the output JSON, right?

**AI Innovation Team:** It is generating \- so it's processing the PDF. So that's only the first block in this pipeline.

**audience 7:** Yeah.

**AI Innovation Team:** Your mouse so people can see where \- no, is it just a block? I don't think you should \- okay, so no, that's the only parsing part is a block, but then you see-

**audience 7:** No, no, no. What I mean? Hold on. I think you understand my question. So, I get all this. You're getting the summary, you're getting the keywords, you're getting the named entities, you're getting the sentiment, and you're outputting to JSON. And the output is JSON output. What I don't understand is you're not generating any synthetic data to train a model with this. This is just a processing pipeline.

**AI Innovation Team:** We are, we are. So when you say getting the summary, that is a generation \- getting a summary from a language model. Generate a summary.

**audience 7:** Right. Right.

**AI Innovation Team:** A language model.

**audience 7:** No, no. I know. But basically, it's not going to generate samples previously where you gave it Q\&A and it prompted that. Do you know what I mean? There. So, I see this extracting information from the documents you've given.

**AI Innovation Team:** Yes, I mean I see your question.

**audience 7:** Is that correct?

**AI Innovation Team:** So I wouldn't say it's not generating. I would say this is essentially creating or transforming the document into a summary-

**audience 7:** Perfect. That's what I mean. Yeah. Yeah. Yeah. So, yeah.

**AI Innovation Team:** \-because that transformation requires generation. But I get your point.

**audience 7:** Yeah... we're not going to train \- we're not going to train a model to do this. We're showing how we can put a process- Yes.

**AI Innovation Team:** This is still going to give you a dataset-

**audience 7:** Yes. Get that.

**AI Innovation Team:** And that dataset we will use to train a model. Think about what is happening.

**audience 7:** Right. Right. I see what you mean.

**AI Innovation Team:** Your intuition is actually great because you could say "hey, catch \- if I had to write a model or create a model to do this task-"

**audience 7:** \[At 35:00\] Okay.

**AI Innovation Team:** "Why don't I just use this pipeline and put it directly?" This is great\! I'm glad that you're making that connection because in some sense, if we just would have done this, it would have worked. But let's say for some strange reason you want to do this in one shot.

So you're saying "I'm only here \- I'm going to give you a summary \- sorry, a document \- and what I want in a single shot is a JSON output structured this particular way," right? Then you can use the data that comes out here because it's paired data. On one hand you have the document, on the other hand you have the JSON output, and you can train the model to do that. But great, great point because I think what we haven't \- why we were pushing so well at iLab SDG \- this is the right way to train the model or enable the model to do these complicated tasks, multifaceted tasks, right? Or I mean, one could call an agentic pipeline, which it is. We're still going to call it agentic. In this particular case, it is an agentic pipeline where in fact you could provide tools \- the JSON conversion.

It's just a tool which I think somebody recognized in the panel \- you can register Pythonic tools in a linear flow like that. Once you move to the graph, that becomes an entire orchestration piece, right?

But yeah, great point. I hope other people realize what this point was, but yeah, for this exercise, we will \- because I think it's just to make the connection that you can use this pipeline to create a dataset to do a one-shot conversion.

**audience 7:** Okay.

**audience 7:** Cheers. Appreciate it.

## **Use Cases for One-Shot Generation**

**AI Innovation Team:** Basically, if you have some constraints on how many inference calls you can do in your production workflow. So let's say you're looking for this single shot generation.

This way you can generate your data on, let's say, meeting transcripts from the past years to 2024, and then from here on you want to train, let's say, a small language model to do this one shot \- you can do it this way. I think the other thing to maybe keep in mind is you wanted a very particular type of summary, right? Then you will have to teach. So you have a very specific ask for how the summary should be done. Just a summary block, and your student model is not able to do that \- you can use it to force a student model to generate it that way, right?

Yeah, I think \- so let's walk through the flow block by block.

## **Flow Block Breakdown**

**AI Innovation Team:** The first one \- we have introduced this new Docling Parse PDF block. All it takes is a PDF path, and then it parses the PDF and outputs it in a markdown fashion in this particular column which we are calling "conversation." And then we're going to add our static question to this. Our question is going to be "extract summary, keywords, named entities, and sentiment from the transcript and return it in the JSON format." So this is going to be the fixed question in your dataset. All the rows will have the same question. You're not asking the model to generate the question.

Next we have a couple of LLM blocks. First one is going to be generate summary, and then followed by the keywords, named entity extraction, and sentiment.

All of these is going to use the parsed conversations you have in your dataset and then generate. Finally, you have your format JSON where this is just a Pythonic function where you basically combine all the generated columns in your dataset into this particular JSON format which we want. So you're taking the summary, keywords, named entities, and sentiment and wrapping it as a JSON and then saving it. This is going to be the pipeline or flow we're going to use to generate data.

You're going to use similarly the Llama model as a teacher to generate data and set up the pipeline here from this flow. And then for this demo, I'm selecting the first 10 samples from the dataset so that it just runs faster. The dataset will have a few more samples for you to test out later.

## **Pipeline Context Configuration**

**AI Innovation Team:** Now, I think one thing to maybe get into a bit more detail is the two pipelines and how you're putting them together. Just above this block, I think it'd be worth explaining what is happening here. Right. So the first part is basically you're setting up a pipeline. Can you highlight this because I think people have difficulty understanding what we're talking about? Yeah. Yeah.

**AI Innovation Team:** \[At 40:00\] Zoom in. So the first part you're setting up is a pipeline context. This is where you specify what's going to be your language model endpoint. So in this case we have already created an OpenAI compatible endpoint client, and then which family to use. We're specifying here that we're going to be using the Llama model, and then finally sort of like a batch \- in this case I've specified it to zero so it runs in a linear fashion, one after the other. And then I'm initializing a skills pipeline from the YAML which we just saw above. So if we change the batch size to non-zero, it does parallel processing.

Okay, just to be \- yeah, just that's why I wanted to touch upon this, because this is something that you currently don't have access to because you would run it through the pre-made stuff, but if you're running the notebook, then actually I should stop saying "if" \- you do have access to it if you can run this. Yeah, but just to understand the different parts will be helpful.

**audience 3:** Okay.

**AI Innovation Team:** And the other thing to just \- in case you're not yet confused enough, I'll add a little bit, a bit more \- which is we call it "pipeline context," we should have just called it "flows" or just to be consistent. So yeah, that's on us. We should have called it flows. All right, there was a question somewhere. Right.

## **Batch Size Clarification**

**audience 3:** I was going to ask you to repeat what you said about batch size.

**AI Innovation Team:** So in this case we have set the batch size to zero. So basically what it does is it processes the PDFs one after the other. But if you set it to a non-zero number, what happens is you start parallelizing it.

So it can start processing the PDFs in parallel and generating data based on those PDFs all in parallel. So these are enhancements which you can add to improve your generation depending on how much your endpoint can handle.

**audience 3:** So this is not training kind batch size.

**audience 3:** This is a separate parameter from training.

**AI Innovation Team:** Yeah, there's no training here. It's just a-

**audience 3:** So it's not that type of thing. It's not like a hyper parameter. But then when you say it's parallelized, could you explain what parallel \- like is it taking advantage of whatever might be there? What is the parallelization that's happening?

**AI Innovation Team:** So you have an endpoint \- so an endpoint is how you have access to the teacher model. Just stay there. I think it's easier to like \- and the teacher model can take a single request. So teacher model is wrapped around in vLLM and that's wrapped around an OpenAI endpoint. You can give the teacher model a single question, right? Or you can give it five questions at the same time, right? Depending on how vLLM will manage the batching, you will get performance improvement.

**audience 3:** Mm-hm. Okay.

**audience 3:** That's all it is. Okay.

**AI Innovation Team:** Correct, correct \- how many requests you're sending concurrently. No, it's just to like \- we just want to talk about these things because if you want to play with it, you should know what it is, not otherwise.

**audience 3:** It's just like the inference \- like what you're sending per request type thing. Okay. Okay.

**audience 3:** It's really not as complicated as I was making it.

## **Demo Setup**

**AI Innovation Team:** Great. So we are selecting the first 10 samples just for this demo, but you can run it on the entire dataset. So the 10 samples here would be 10 PDFs or the 10 meeting transcripts. And if you run this pipeline, the first run would actually take a couple of minutes because you're initializing Docling new on your environment. It will try to pull some models which it will use to do the parsing. So that might take a minute or so.

# **Day 2 Presentation Transcript \- Part 3 (Cleaned)**

## **Generated Data Results**

**AI Innovation Team:** \[At 45:00\] So just a note on that, right? So if you're able to \- maybe we can run this at your own leisure and at your own pace, and then you can hit us back with questions in the Slack channel. So I'll just go ahead and look at the generated data what we have on my end. So as you've seen before, can keep an eye on those questions. If you look at the JSON output column which we have from generated data, it will look something like this. It's a JSON dictionary.

It has a bunch of keywords just like we expected and a succinct summary, followed by \- we asked the model to only generate anywhere between five to 10 keywords. Seems like it's respecting that as well. And then in this particular case it wasn't able to pull any location mentioned from the meeting transcript. So that gets filled with a null object, and then the overall sentiment for this particular \[transcript is positive\].

## **Pre-built Blocks and Custom Components**

**Pete Davis:** So everything we're using here is a pre-built block that you've already provided through, and then we can create our... Okay.

**AI Innovation Team:** So if you go \- yeah. Yeah. So I've shared the link to all the available pre- all the custom blocks we have introduced in this demo would be housed here under the blocks. You can go ahead and check them out.

**Pete Davis:** Okay, thanks.

**AI Innovation Team:** And likewise the custom prompts we use will also be here. We have one for keywords, named entity, sentiment \- all those.

## **Model Performance Discussion**

**Pete Davis:** So, do you find that this works better with certain models as opposed to other certain sizes, certain flavors? We're using Llama 70B. How does it perform the same sort of task with a smaller Granite model?

**AI Innovation Team:** I mean, so for generation we've always used Mixtral and now we're also using Llama. We never used Granite because Granite is typically the student model \- it doesn't have enough... to be honest, right? It's not the best model out there in terms of instruction following capability. The reason we have to teach it things is because it doesn't know \- it's small and hence that's where the value is. But actually, most small models will struggle with doing this kind of task, especially in a single shot. So what we do here is we use a bigger model that is specifically good at instruction following, and so you can look up IfEval numbers for your teacher model if you're interested.

Some small models like Phi-3.5 or Phi-4 \- they're only 14B but they're very, very good at instruction following. These models are very good at instruction following but despite their small size. But if you're interested in playing with small models, those are the two I would recommend. If you want to run them locally, they have very, very high IfEval numbers, which means they can follow your instructions or the guidance here much more precisely than Llamas and smaller Llamas and Granites.

## **Hands-on Exercise Suggestions**

**AI Innovation Team:** So that brings us to basically the end of this particular use case. So if you want to play with it, we recommend you try \- a simple thing which you could do is like I mentioned, for this demo we ran this pipeline only on 10 PDFs. You can try running it on the whole dataset. If you want to go a bit more further, you can try adding a new block or a new part in your structured summary.

**AI Innovation Team:** \[At 50:00\] So let's say you want to extract if there were any risk factors mentioned \- it's simple. You just introduce an LLM block, write a simple prompt, even in your prompts, and then add it to your flow and try to see if the model is able to now generate that as well. So maybe I'll say it in the following way. The easiest thing you can do is \- and I will say everybody should \- go to the summarization block and look up its corresponding prompt and try to add. Can you bring up that prompt?

So try changing or adding a new principle or removing a principle just to get an idea of what would happen. Let's say you want the summary to look for something specific \- if changing this is doing it, or play with it to see until you get the results. How do you change a prompt? Because that's kind of where most of the work really goes, right? In making sure that you're... maybe change the example. Maybe add a few more examples. For instance, in this case, we're asking for a four to five sentence summary. You can ask for a single sentence summary and see how that differs. And again, example here is just one. You can add a bit more if you like.

And then the next step would be you can introduce another pre-made block which is \- let's say you want to add an LLM block. So can we go to the block definition again? So let's say you want to add a new block. This is literally what you have to do. You just create, copy and paste maybe this \- go a little bit up \- the summary thing. So the first block there, right? "gen\_summary" \- you can literally copy and paste. The only thing you will have to choose \- sorry, change \- is now you will have to go in your prompts folder and add... you can in fact copy the summary.yaml and then change it there. Then it basically takes a summary of the first one and does something else with it, right?

And then finally, if you're feeling really, really courageously here, maybe try to write an evaluation block-type block, right? What you would want to do is give your language model a bunch of keywords and ask it to filter samples if they have those keywords there. For example, you can use the use case from yesterday where Shiv showed you how to write that block \- the filter by value. What's the name? Eval block. Sorry. Let me show you how to write the eval block and see if you can prompt your language model that if it detects a certain keyword, it removes that entire sample. Nice.

But I hope you can see it's very, very simple once you get the basic building blocks in your mind. So again, if you want to write your own flow, this is what you do. You go on the GitHub, load the repo, create a flow folder, create a blocks folder, create a prompt folder, and then define a YAML in which you would \- within the flow folder \- write your flow, which is basically a collection of blocks.

For each block, they provide mostly two things in the block: an LLM and a prompt. That's it \- you have a new flow pipeline. Anyway, I think from a time \- so we have 20 minutes perfect. Let's open for questions. Maybe we can go back to the...

## **Q\&A Session**

**audience 1:** Looks like we had some activity in the Slack channel, but I think you guys answered. Hunter?

**audience 2:** All right, I will take a stab at this. I am trying to wrap my head around everything, and so I'm going to kind of regurgitate back a high-level summary and if it's correct or if it misses the mark \- and if not, then I'll have to adjust my flow, I guess, to better summarize.

**audience 2:** \[At 55:00\] So what I basically heard today is you have defined what I see on screen, right? These series of blocks into a flow, and somewhere in your history you have chosen that these are the right blocks to go in this order, and they will give you some output that is going to be used to train a model at some point \- most likely. It doesn't have to be, right? It could be \- I mean, now that we're decoupled, we can use this however we want.

But effectively, this will give you higher quality synthetic data to train your model as opposed to \- I think what you said earlier was "one-shotting," right? Where we would have a Q\&A YAML and just give it to the model and have it generate all the synthetic data, and it may be good, it may be bad, and it will train a model. Am I missing something, or is that an overall okay summary?

**AI Innovation Team:** It's a good summary. I think the one thing I would like to clarify is when you give a Q\&A or YAML in the current workflow, there is actually a whole flow behind it that you saw yesterday. It's just not exposed to you, right? So what happens in that case is that when you give a Q\&A YAML, it goes and uses an LLM block to generate the questions, then evaluate those questions, filter those questions, then for the remaining filtered questions it starts generating the answer, then evaluates the question and answer pair, and then gives you the final answer. So in some sense, there is no...

It's a difference of how you interact with the system. So because of the way we wanted to do this in the product, the ask was "hey, the user is only writing the examples." Fair enough. If you are in that box, you can't do much. What has happened is if you look inside the blocks, you will realize that "hey, no, I can change anything I want, and if I can change anything I want, then I don't need to restrict to provide those five examples as my only interaction, right? I can directly manipulate the prompt. I can provide principles. I can..." So I think that's kind of where it is \- it is differing. Earlier, the amount of control you had on what gets generated was limited to the examples only. And as we have realized in quite a lot of use cases, that falls short.

**audience 2:** So, I heard that \- and I saw the link. There's a ton of different blocks that are already provided, right? We can compose those as we need. Are there also flows that are provided? Because it seems like there should be a standard set of flows and that...

**AI Innovation Team:** So let me now make that clear. So the knowledge pipeline that you worked with earlier \- that's a flow. If you go to the repo, you can see that flow today, right? And in fact, now you can edit that. You can go in the prompt and hey, "where is Akash putting my examples in? Can I go and change a principle there instead of just the example?" Right? So that's the big sort of like jump that you're getting in terms of the control you have over generation. Principle is the same \- it's like \- so you provide an example which you will supervise. It's just now you have a lot more levers you can pull.

**audience 2:** So would it be reasonable to think maybe long term that we have different flows? Obviously, we have knowledge skills. Would it be maybe reasonable to say that we have different flows for the different types of model use cases that we might want?

**AI Innovation Team:** Absolutely. That is the goal. So we already have an annotation flow. We already have a reasoning model customization flow, and a bunch of other things that we use internally for research.

For example, what you will see \- there's something coming up in InferenceScaling hub. I think the first release was made, but essentially it's a way to unroll your language model guided by a reward function in real time. So you're not doing it for training \- you're just generating again and again and again until it gets it right. Think about agentic workflow. That's what you will need there. Until you make the right function call, I'm not going to let you go, and I have a way to check it.

**audience 2:** \[At 1:00:00\] Okay, awesome. Great.

**AI Innovation Team:** So absolutely, the success looks like everybody in this group contributes a flow. So, the reason we're asking you to go change this flow is you can contribute it back, and then that stays there as a pre-made pipeline that you can use again.

## **Process Discussion for Customer Pipelines**

**audience 3:** Maybe to piggyback on that. So, if we do have customers that we found tend to \- possibly they could potentially benefit from certain pre-canned pipelines. Yes, we can go back and create some of these, but there are some off the bat that are top of mind that would be nice to include in our canned arsenal. What's the process to sync with your team to get any of those codified? And how would you approve or prioritize those use case pipelines?

**AI Innovation Team:** Great point, and the answer is even better. You don't have to just create those and contribute the fact. We got a release every week. We're going to move to hopefully a nightly release. And then the PMs on the side will work on having those certain releases sort of come baked into... but any release the way it's set up will be compatible with the platform.

So anything that you contribute the moment it becomes part of the release... Taylor, I'm gonna maybe stop you there.

**audience 4:** So, I just want to chime in, Akash, because that's a valid ask from Tola. I think that'll be more for the InstructLab engineers and not Akash's research team to have a process where we request, "hey, we need this built out," because it's not necessarily realistic. Sure. I'm just saying I think it would be more... however they're organized on engineering to help us contribute because what I want the... Well, that's fine. The point then I want to make is that I don't think it's realistic to expect the field to build out everything they're going to need. So there does need to be some kind of Jira request process to get some of this built out potentially that we need to discuss.

**AI Innovation Team:** I think there's going to be some changes in the engineering of how it will happen, so let's just wait for that to happen. That's what I'm trying to say. Maybe there's some confusion. My team is part of the engineering.

**audience 4:** So that's what Tola was asking about.

**AI Innovation Team:** Yeah, absolutely. Absolutely.

## **Enterprise Readiness Considerations**

**audience 3:** Yeah, because it's one thing for us to build certain things especially for the field, it's going to be missing certain pieces like some things that would be essential if you're going to be enterprise ready. For instance, I mean \- and to be fair, I was kind of following but I was multitasking too. So you might have covered this, but I would imagine that some of these \- it would be necessary for you to have an evaluation step codified or however your error checking happens. Like that process would need to be codified and a few other things. The fact that we don't have cycles yet, but we probably need something that would stand in for that. All of those things in terms of enterprise readiness would be great, and sometimes I guess the field won't always have that.

So if we could formalize a process, that would be great. And the other reason I bring up the use cases \- we've had actual customer engagements where we've thought it would be nice. There are particular use cases off the top of my head I could even rattle off and that others possibly have also run into that I think would make us competitive if we could point and say we had these out of the box for them. So that's the reason I'm bringing it up.

**audience 1:** No, I think you...

**AI Innovation Team:** Yeah. Yeah, I mean I think I just want to make sure we're not getting into the point where some of it sounds like a consulting gig, which is different than what gets into the product will be different than... it's work in progress. That's why I'm being kind of very restrictive about saying too much about it, right? If we're getting to a point where every customer has a new need, right? That is not something that will come as part of the product then, right? So I think maybe we could table this for now. We'll come back to it when we have a little bit more definition on exactly how this thing will... Yeah.

**audience 3:** Yeah. Yeah. I think, right, we probably need a little bit less generalization than what we have now, I guess, is what I'm saying. It's good to have the flexibility to be able to configure pipelines as much as you need, and we're never going to be able to fit every use case and we shouldn't. But then there might be some repeatable things that we've been encountering in the field \- that's what I'm talking about \- where it would be nice to be able to say we have canned solutions for these that you can still extend to your heart's content, but something a bit more granular than...

# **Day 2 Presentation Transcript \- Part 4 (Final)**

## **Product Strategy Discussion**

**audience 3:** \[At 1:05:00\] What we have today.

**audience 5:** But I think wouldn't those actually be just templates or patterns that will actually be for specific use cases that's sitting outside of the product? Because those are actually patterns that are built using the features of the product. It gives you the tools and you actually consume them either as solutions or patterns that the field needs for customer use cases.

**audience 3:** Perhaps like I'm...

**AI Innovation Team:** Where this is going towards. I think the point is a little bit \- if I'm reading this correct \- is on very repeatable things that we might find. The knowledge pipeline, which by the way already exists in the product, certain version of it potentially could come as part of the product.

**audience 3:** Okay. Right. Or even the summarization...

**AI Innovation Team:** I mean the point that Vijay is making is why \- and it's sort of why I'm a little hesitant right now \- is we're trying to figure out how often you get to a point where that pre-made thing just works as it is without any customization. What you all have told us is that that's not really the case often. But then a level of abstraction that we want to work at would be that these tools exist in the product. You do create these things or templates as Vijay said, right? And this modification of template is still a task.

**audience 3:** Yes. Yes.

**audience 5:** But I think, yeah, let us table this offline because how do we consume those templates? How do we manage that repository of use cases is something that will have to come out maybe not this way but at some point, so everybody contributes to that repository and then learns from it.

## **Support and Model Compatibility Concerns**

**audience 6:** I'm sorry, I'm sorry not to jump ahead of you for this, but then how exactly is this going to be supported if people are using, let's say, different models, right? Because I mean, if anyone can contribute, I get the idea of a software development where you can develop all of this and it's very nicely created \- the structure and everything \- but at the end of the day, if you want people to contribute back with new modules, and these modules do a lot of things, who gets to choose what exactly is going to be part of the product and what is going to be supported at the enterprise level and on which model? Because let's say I created a pipeline that I...

**AI Innovation Team:** Arthur, it's a great point, and why the reason I'm currently saying let's come back to it \- because these are the things that are being flushed out. We don't have an answer on it. There's a third-party model validation workflow as you're aware, so some compatibility definition come from there. But 3.0 will answer all those questions. At a high level, I mean, what Vijay said is where we will potentially land, which at which point it becomes... certain third-party models that we already support in our inference server and overall OpenShift AI products will be the ones that we support.

We may not even support particular workflows at all, right? So I think probably it'd be best to table these discussions and use probably this time, let's say, for questions on the pipeline itself. I don't know if Katie, Jeff, you guys want to do a separate session once we get a little bit more clarity on how this is prioritized, we can...

**audience 6:** Okay, then I have a really short question then \- 10 seconds. Can we say that for customers that this SDK \- it is basically supported in terms of the code? If there is an error, let's say, for whatever the library does and it's a problem, then this is supported but not the custom-built pipeline? Okay.

**AI Innovation Team:** It is supported. Let's just go with the community support for now because this will run on the platform. Whether we will provide you support in terms of debugging it...

**audience 6:** Okay. Yeah.

**AI Innovation Team:** No. \[Someone's\] going to kill somebody.

**audience 6:** Okay. Cool. Thank you.

## **Field Scope and Use Case Discussion**

**audience 1:** All right. No, you've been patient.

**audience 2:** Actually, can I jump in front of Noel real quick?

**audience 7:** So I have a list of 13 questions because \[you're\] not letting me off. Good news, Akash \- basically this has got nothing to do with you, which is perfect. Thanks for everything you've done. It's been really good. So from a perspective in front of customers, what's in scope for SE and what isn't when it comes to stuff like this? Because the tech is there. I can see how it all fits together. But where do we kind of draw the line with the InstructLab when we're doing documents? That was pretty discrete, but this seems to be evolving into bigger and bigger \- or potentially bigger and bigger use cases as we go forward.

**audience 7:** \[At 1:10:00\] So, how does... I think it's more a conversation or a question for Jeff \- basically how do we say that's perfect.

**AI Innovation Team:** I was going to say that this is the reason this is happening has a lot to do with Jeff, and I see him on the screen so he should take this question.

## **Next Steps and Use Case Curation**

**audience 8:** So if you guys remember in Boston, we had what \- five to seven use cases that we all focused on for knowledge. This one is very open-ended. I want to see what the capabilities were, where the training hub was, etc. And then curate another set of use cases that we can focus on. Next week is going to be a busy week for a lot of folks, but I think there'll be a lot of quiet time for some folks that won't be at summit. So digest this over the next week.

I'd like to ask Katie to have a homework assignment for the folks that are not attending summit to think about use cases that you've seen that you couldn't attack with knowledge in the field, and let's curate \- I don't know \- three to four use cases that we all want to come together and work on over the next couple of months and then publish them throughout our playbook and the enterprise to make sure that we can at least have a surface area of use cases we can publish to the field, blog about. I'd like to get our team blogging with the content team on our Red Hat AI site so you can show examples in the field. But I think three to four use cases would be the target for what those are. I don't know. I'm seeing this in real time with you guys.

**AI Innovation Team:** Maybe just to add to Jeff's thing, we should think a little bit about agents, and something that would be great is to maybe work towards creating an agentic system that explains what our product is with this OpenShift AI. Apparently, Lightspeed folks do have some MCP servers that they made for basic stuff. There's product documentation, so it could be a good target to have a small model do all that as a Clippy version of the thing.

Yeah, thinking about other use cases will be great.

## **Session Wrap-up and Tomorrow's Preview**

**audience 1:** All right, with two minutes left, and so now we covered a lot.

**audience 7:** I've not finished my questions. So, I'm not doing it.

**audience 1:** Gosh. Okay. Right.

**AI Innovation Team:** Remaining 11 questions.

**audience 1:** Again, we'll be hanging out again tomorrow. So, just hold that. Similar to what we did today, I think it will make sense maybe to start tomorrow with a Q\&A session a little bit before we dive into the next topic. But similar to what we did yesterday, I'll put everything in the Slack channel and I will be putting it in a lot faster. I already have the first recording already done, so we'll get it a lot faster than we did yesterday. But thanks for your attention. I know it's a lot to be continued, and we'll put the links to not only the recordings, but also any additional kind of prep for tomorrow in that Slack channel as well. So, bye.

**AI Innovation Team:** Like a hype video. For prep material tomorrow, we will walk you through how to customize a reasoning model. So, how to make it think about your... Fun fact, it's going to be a demo at summit. So, maybe we can send you the short two-minute video of what we made. You want to take a look, get excited about it. Yeah, and it'd be interesting because it'll be knowledge all over again.

But now from a point of view of iLab SDG, hopefully we can start seeing those changes and how easy or difficult it is to work with it. All right, we should share the video with...

**audience 1:** All right. Thanks everybody. We'll see you tomorrow. Bye.

**audience 6:** Thank you.

**audience 9:** Bye.


# **Actual Q\&A from iLab SDG Session**

## **Q\&A Exchange 1: Data Generation vs Processing Pipeline**

**audience 7:** From what I understand, this is a processing pipeline. It's not actually generating anything, is it? Other than the output JSON, right?

**AI Innovation Team:** It is generating \- so it's processing the PDF. So that's only the first block in this pipeline.

**audience 7:** No, no, no. What I mean? Hold on. I think you understand my question. So, I get all this. You're getting the summary, you're getting the keywords, you're getting the named entities, you're getting the sentiment, and you're outputting to JSON. And the output is JSON output. What I don't understand is you're not generating any synthetic data to train a model with this. This is just a processing pipeline.

**AI Innovation Team:** We are, we are. So when you say getting the summary, that is a generation \- getting a summary from a language model. Generate a summary.

**audience 7:** No, no. I know. But basically, it's not going to generate samples previously where you gave it Q\&A and it prompted that. Do you know what I mean? There. So, I see this extracting information from the documents you've given.

**AI Innovation Team:** So I wouldn't say it's not generating. I would say this is essentially creating or transforming the document into a summary because that transformation requires generation. But I get your point.

**audience 7:** Perfect. That's what I mean. Yeah. Yeah. Yeah. So, yeah... we're not going to train \- we're not going to train a model to do this. We're showing how we can put a process- Yes.

**AI Innovation Team:** This is still going to give you a dataset, and that dataset we will use to train a model. Think about what is happening. Your intuition is actually great because you could say "hey, catch \- if I had to write a model or create a model to do this task \- why don't I just use this pipeline and put it directly?" This is great\! I'm glad that you're making that connection because in some sense, if we just would have done this, it would have worked. But let's say for some strange reason you want to do this in one shot.

So you're saying "I'm only here \- I'm going to give you a summary \- sorry, a document \- and what I want in a single shot is a JSON output structured this particular way," right? Then you can use the data that comes out here because it's paired data. On one hand you have the document, on the other hand you have the JSON output, and you can train the model to do that.

---

## **Q\&A Exchange 2: Pre-built Blocks**

**Pete Davis:** So everything we're using here is a pre-built block that you've already provided through, and then we can create our... Okay.

**AI Innovation Team:** So I've shared the link to all the available pre- all the custom blocks we have introduced in this demo would be housed here under the blocks. You can go ahead and check them out. And likewise the custom prompts we use will also be here. We have one for keywords, named entity, sentiment \- all those.

---

## **Q\&A Exchange 3: Model Performance**

**Pete Davis:** So, do you find that this works better with certain models as opposed to other certain sizes, certain flavors? We're using Llama 70B. How does it perform the same sort of task with a smaller Granite model?

**AI Innovation Team:** I mean, so for generation we've always used Mixtral and now we're also using Llama. We never used Granite because Granite is typically the student model \- it doesn't have enough... to be honest, right? It's not the best model out there in terms of instruction following capability. The reason we have to teach it things is because it doesn't know \- it's small and hence that's where the value is. But actually, most small models will struggle with doing this kind of task, especially in a single shot. So what we do here is we use a bigger model that is specifically good at instruction following, and so you can look up IfEval numbers for your teacher model if you're interested.

Some small models like Phi-3.5 or Phi-4 \- they're only 14B but they're very, very good at instruction following. These models are very good at instruction following but despite their small size. But if you're interested in playing with small models, those are the two I would recommend. If you want to run them locally, they have very, very high IfEval numbers, which means they can follow your instructions or the guidance here much more precisely than Llamas and smaller Llamas and Granites.

---

## **Q\&A Exchange 4: Few-shot Examples**

**audience 3:** And for folks that \- if we happen to talk to data science customers \- what would be the reasoning? What guidance could we offer them in terms of why that's kind of the canonical number? Because there are a few kinds...

**AI Innovation Team:** Yeah, I think it's a good question. The true answer is it depends on the use case, but if the average case is five, language models are good at interpolation. So five is generally a good number, and the number comes from \- if you look at, I think, six months ago when we still used to do five-shot evaluation of models, that's kind of where it comes from. People found that at five examples, language models tend to get the pattern right. So they would give five examples and then let's say you do have a setup in which you just like have too much diversity and you have to capture all that, right? All those different use cases \- again, in the iLab SDG, you can go beyond that. There isn't really a lot of these limits that you heard of \- they were specific implementation limitations in the platform. One of the reasons why we are decoupling these things is that there were never a limitation in the actual SDG mechanism, right? So you can provide one or you can provide 15\. The guidance is around five, but as the diversity of your task increases, you should provide representative examples of the different possible types.

---

## **Q\&A Exchange 5: Batch Size Clarification**

**audience 3:** I was going to ask you to repeat what you said about batch size.

**AI Innovation Team:** So in this case we have set the batch size to zero. So basically what it does is it processes the PDFs one after the other. But if you set it to a non-zero number, what happens is you start parallelizing it. So it can start processing the PDFs in parallel and generating data based on those PDFs all in parallel. So these are enhancements which you can add to improve your generation depending on how much your endpoint can handle.

**audience 3:** So this is not training kind batch size. This is a separate parameter from training.

**AI Innovation Team:** Yeah, there's no training here. It's just a-

**audience 3:** So it's not that type of thing. It's not like a hyper parameter. But then when you say it's parallelized, could you explain what parallel \- like is it taking advantage of whatever might be there? What is the parallelization that's happening?

**AI Innovation Team:** So you have an endpoint \- so an endpoint is how you have access to the teacher model. Just stay there. I think it's easier to like \- and the teacher model can take a single request. So teacher model is wrapped around in vLLM and that's wrapped around an OpenAI endpoint. You can give the teacher model a single question, right? Or you can give it five questions at the same time, right? Depending on how vLLM will manage the batching, you will get performance improvement.

**audience 3:** It's just like the inference \- like what you're sending per request type thing. Okay. Okay. It's really not as complicated as I was making it.

---

## **Q\&A Exchange 6: Hunter's Summary Question**

**audience 2:** All right, I will take a stab at this. I am trying to wrap my head around everything, and so I'm going to kind of regurgitate back a high-level summary and if it's correct or if it misses the mark \- and if not, then I'll have to adjust my flow, I guess, to better summarize.

So what I basically heard today is you have defined what I see on screen, right? These series of blocks into a flow, and somewhere in your history you have chosen that these are the right blocks to go in this order, and they will give you some output that is going to be used to train a model at some point \- most likely. It doesn't have to be, right? It could be \- I mean, now that we're decoupled, we can use this however we want.

But effectively, this will give you higher quality synthetic data to train your model as opposed to \- I think what you said earlier was "one-shotting," right? Where we would have a Q\&A YAML and just give it to the model and have it generate all the synthetic data, and it may be good, it may be bad, and it will train a model. Am I missing something, or is that an overall okay summary?

**AI Innovation Team:** It's a good summary. I think the one thing I would like to clarify is when you give a Q\&A or YAML in the current workflow, there is actually a whole flow behind it that you saw yesterday. It's just not exposed to you, right? So what happens in that case is that when you give a Q\&A YAML, it goes and uses an LLM block to generate the questions, then evaluate those questions, filter those questions, then for the remaining filtered questions it starts generating the answer, then evaluates the question and answer pair, and then gives you the final answer. So in some sense, there is no...

It's a difference of how you interact with the system. So because of the way we wanted to do this in the product, the ask was "hey, the user is only writing the examples." Fair enough. If you are in that box, you can't do much. What has happened is if you look inside the blocks, you will realize that "hey, no, I can change anything I want, and if I can change anything I want, then I don't need to restrict to provide those five examples as my only interaction, right? I can directly manipulate the prompt. I can provide principles. I can..." So I think that's kind of where it is \- it is differing. Earlier, the amount of control you had on what gets generated was limited to the examples only. And as we have realized in quite a lot of use cases, that falls short.

---

## **Q\&A Exchange 7: Available Flows**

**audience 2:** So, I heard that \- and I saw the link. There's a ton of different blocks that are already provided, right? We can compose those as we need. Are there also flows that are provided? Because it seems like there should be a standard set of flows and that...

**AI Innovation Team:** So let me now make that clear. So the knowledge pipeline that you worked with earlier \- that's a flow. If you go to the repo, you can see that flow today, right? And in fact, now you can edit that. You can go in the prompt and hey, "where is Akash putting my examples in? Can I go and change a principle there instead of just the example?" Right? So that's the big sort of like jump that you're getting in terms of the control you have over generation. Principle is the same \- it's like \- so you provide an example which you will supervise. It's just now you have a lot more levers you can pull.

**audience 2:** So would it be reasonable to think maybe long term that we have different flows? Obviously, we have knowledge skills. Would it be maybe reasonable to say that we have different flows for the different types of model use cases that we might want?

**AI Innovation Team:** Absolutely. That is the goal. So we already have an annotation flow. We already have a reasoning model customization flow, and a bunch of other things that we use internally for research.

---

## **Q\&A Exchange 8: Support and Model Compatibility**

**audience 6:** I'm sorry, I'm sorry not to jump ahead of you for this, but then how exactly is this going to be supported if people are using, let's say, different models, right? Because I mean, if anyone can contribute, I get the idea of a software development where you can develop all of this and it's very nicely created \- the structure and everything \- but at the end of the day, if you want people to contribute back with new modules, and these modules do a lot of things, who gets to choose what exactly is going to be part of the product and what is going to be supported at the enterprise level and on which model? Because let's say I created a pipeline that I...

**AI Innovation Team:** Arthur, it's a great point, and why the reason I'm currently saying let's come back to it \- because these are the things that are being flushed out. We don't have an answer on it. There's a third-party model validation workflow as you're aware, so some compatibility definition come from there. But 3.0 will answer all those questions. At a high level, I mean, what Vijay said is where we will potentially land, which at which point it becomes... certain third-party models that we already support in our inference server and overall OpenShift AI products will be the ones that we support.

**audience 6:** Okay, then I have a really short question then \- 10 seconds. Can we say that for customers that this SDK \- it is basically supported in terms of the code? If there is an error, let's say, for whatever the library does and it's a problem, then this is supported but not the custom-built pipeline? Okay.

**AI Innovation Team:** It is supported. Let's just go with the community support for now because this will run on the platform. Whether we will provide you support in terms of debugging it... No.

---

## **Q\&A Exchange 9: Customer Pipeline Process**

**audience 3:** Maybe to piggyback on that. So, if we do have customers that we found tend to \- possibly they could potentially benefit from certain pre-canned pipelines. Yes, we can go back and create some of these, but there are some off the bat that are top of mind that would be nice to include in our canned arsenal. What's the process to sync with your team to get any of those codified? And how would you approve or prioritize those use case pipelines?

**AI Innovation Team:** Great point, and the answer is even better. You don't have to just create those and contribute the fact. We got a release every week. We're going to move to hopefully a nightly release. And then the PMs on the side will work on having those certain releases sort of come baked into... but any release the way it's set up will be compatible with the platform.

---

## **Q\&A Exchange 10: Field Scope Question**

**audience 7:** So I have a list of 13 questions because \[you're\] not letting me off. Good news, Akash \- basically this has got nothing to do with you, which is perfect. Thanks for everything you've done. It's been really good. So from a perspective in front of customers, what's in scope for SE and what isn't when it comes to stuff like this? Because the tech is there. I can see how it all fits together. But where do we kind of draw the line with the InstructLab when we're doing documents? That was pretty discrete, but this seems to be evolving into bigger and bigger \- or potentially bigger and bigger use cases as we go forward.

**audience 8:** So if you guys remember in Boston, we had what \- five to seven use cases that we all focused on for knowledge. This one is very open-ended. I want to see what the capabilities were, where the training hub was, etc. And then curate another set of use cases that we can focus on. Next week is going to be a busy week for a lot of folks, but I think there'll be a lot of quiet time for some folks that won't be at summit. So digest this over the next week.

I'd like to ask Katie to have a homework assignment for the folks that are not attending summit to think about use cases that you've seen that you couldn't attack with knowledge in the field, and let's curate \- I don't know \- three to four use cases that we all want to come together and work on over the next couple of months and then publish them throughout our playbook and the enterprise to make sure that we can at least have a surface area of use cases we can publish to the field, blog about.