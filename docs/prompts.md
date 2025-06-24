# Prompt Configuration Guide

This guide explains how to configure LLM prompts in SDG Hub using the declarative YAML structure. SDG Hub uses a standardized approach to prompt engineering that enables consistent, high-quality data generation across different domains and tasks.

## Overview

SDG Hub prompts are defined in YAML configuration files that follow a structured format. These configurations support Jinja2 templating, multi-stage parsing, and sophisticated output formatting to enable complex data generation workflows.

## Basic Structure

All prompt configurations follow this standard structure:

```yaml
system: <system_message>
introduction: <task_introduction>
principles: <task_principles_and_guidelines>
examples: <few_shot_examples>
generation: <generation_prompt>
start_tags: [<parsing_start_tags>]
end_tags: [<parsing_end_tags>]
```

## Core Components

### System Message (`system`)

Sets the AI assistant's role and behavior context.

**Format Options:**
- String: Defines the assistant's persona
- `null`: No system message
- Empty string: Minimal system context

**Examples:**
```yaml
# Knowledge assistant
system: You are a very knowledgeable AI Assistant that will faithfully assist the user with their task.

# Specialized classifier
system: You are an expert text classifier trained to label questions from online forums.

# No system message
system: null
```

### Introduction (`introduction`)

Describes the high-level task to be performed. Often includes context and objectives.

**Examples:**
```yaml
# Educational content generation
introduction: Develop a series of educational question and answer pairs from a chapter in a {{domain}} textbook.

# Data annotation task
introduction: "Task Description: Data Annotation"

# Dynamic task with variables
introduction: |
  Your task is to generate {{num_samples}} diverse and well-structured questions for the following task:
  "{{task_description}}"
```

### Principles (`principles`)

Provides detailed guidelines, constraints, and requirements for the task. This is where you specify quality criteria, formatting rules, and step-by-step instructions.

**Examples:**
```yaml
# Diversity requirements
principles: |
  Here are the requirements:
  1. Try not to repeat the verb for each instruction to maximize diversity.
  2. The language used for the instruction also should be diverse.
  3. The instructions should be in English.
  4. The instructions should be 1 to 2 sentences long.

# Quality criteria
principles: |
  Quality Requirements:
  - Questions should be answerable from the provided document
  - Answers must be factually accurate and well-supported
  - Use clear, concise language appropriate for the target audience
  - Avoid ambiguous or leading questions
```

### Examples (`examples`)

Provides few-shot learning examples to demonstrate the expected behavior and output format.

**Examples:**
```yaml
# Simple example with variables
examples: |
  Here is an example to help you understand:
  {{seed_question}}
  {{seed_response}}

# Multi-example format
examples: |
  Example 1:
  Question: {{icl_query_1}}
  Answer: {{icl_response_1}}
  
  Example 2:
  Question: {{icl_query_2}}
  Answer: {{icl_response_2}}

# No examples
examples: ""
```

### Generation (`generation`)

Contains the actual prompt that will be sent to the LLM, including variable placeholders and final instructions.

**Examples:**
```yaml
# Simple generation
generation: |
  Here is the query for annotation:
  {{text}}

# Structured input format
generation: |
  Here is the document:
  [Start of Document]
  {{document}}
  [End of Document]
  
  Generate a question and answer pair based on this content.

# Multi-step generation
generation: |
  Document: {{document}}
  Task: {{task_description}}
  
  Please provide your response in the following format:
  [QUESTION] Your question here [/QUESTION]
  [ANSWER] Your answer here [/ANSWER]
```

### Parsing Tags (`start_tags` and `end_tags`)

Define how to extract structured output from LLM responses. These tags mark the beginning and end of extractable content sections.

**Examples:**
```yaml
# Question and answer extraction
start_tags: ["[QUESTION]", "[ANSWER]"]
end_tags: ["[/QUESTION]", "[/ANSWER]"]

# Multi-stage evaluation
start_tags: ["[Start of Judgement]", "[Start of Verdict]"]
end_tags: ["[End of Judgement]", "[End of Verdict]"]

# No structured parsing
start_tags: [""]
end_tags: [""]
```

## Variable Templating with Jinja2

SDG Hub uses Jinja2 templating for dynamic prompt generation. Variables are enclosed in double curly braces `{ { variable_name } }`.

### Common Variable Patterns

**Content Variables:**
```yaml
generation: |
  Document: {{document}}
  Text: {{text}}
  Query: {{prompt}}
```

**Control Variables:**
```yaml
introduction: Generate {{num_samples}} examples for {{domain}} domain
```

**Example Variables:**
```yaml
examples: |
  Example: {{seed_question}}
  Response: {{seed_response}}
```

### Advanced Templating

**Conditional Content:**
```yaml
system: |
  You are an AI assistant.
  {% if task_description %}
  Your current task: {{ task_description }}
  {% endif %}
```

**Loops and Lists:**
```yaml
examples: |
  {% for i in range(3) %}
  Example {{ i+1 }}: {{ icl_query[i] }}
  Response {{ i+1 }}: {{ icl_response[i] }}
  {% endfor %}
```

## Advanced Configuration Patterns

### Multi-Stage Parsing

For complex outputs requiring multiple extraction phases:

```yaml
start_tags: ["[Start of Analysis]", "[Start of Recommendation]", "[Start of Implementation]"]
end_tags: ["[End of Analysis]", "[End of Recommendation]", "[End of Implementation]"]
```

### Conditional Logic

Using Jinja2 conditionals for dynamic content:

```yaml
generation: |
  { % if difficulty_level == "beginner" % }
  Please provide a simple explanation suitable for beginners.
  { % elif difficulty_level == "advanced" % }
  Please provide a detailed technical explanation.
  { % endif % }
  
  Topic: { {topic} }
```

### ICL (In-Context Learning) Patterns

For sophisticated few-shot learning:

```yaml
examples: |
  Here are detailed examples:
  
  Query: {{icl_query_1}}
  Analysis: {{icl_analysis_1}}
  Response: {{icl_response_1}}
  Critique: {{icl_critique_1}}
  Improved Response: {{icl_revised_response_1}}
  
  Query: {{icl_query_2}}
  Analysis: {{icl_analysis_2}}
  Response: {{icl_response_2}}
  Critique: {{icl_critique_2}}
  Improved Response: {{icl_revised_response_2}}
```

## Integration with Blocks

Prompt configurations are used by LLM blocks in flows:

```yaml
- block_type: LLMBlock
  block_config:
    block_name: generate_qa
    config_path: configs/knowledge/simple_generate_qa.yaml
    model_id: llama3.1-70b
    output_cols: [question, answer]
  gen_kwargs:
    temperature: 0.7
    max_tokens: 2048
```

The system automatically:
1. Loads the YAML configuration
2. Renders Jinja2 templates with sample data
3. Applies model-specific chat templates
4. Parses responses using the defined tags

## Best Practices

### 1. Clear Structure
- Use descriptive section headers
- Separate concerns between different sections
- Keep principles specific and actionable

### 2. Effective Examples
- Provide diverse, high-quality examples
- Show the exact format you want
- Include edge cases when relevant

### 3. Robust Parsing
- Use distinctive parsing tags
- Avoid tags that might appear in natural text
- Test parsing with various response formats

### 4. Variable Design
- Use descriptive variable names
- Document expected variable types
- Provide fallback content for optional variables

### 5. Quality Control
- Include quality criteria in principles
- Specify output format requirements clearly
- Add validation rules when possible

### 6. Testing and Iteration
- Test prompts with different models
- Validate output quality and format
- Iterate based on actual performance

## Common Patterns Reference

### Basic Q&A Generation
```yaml
system: You are an educational AI assistant.
introduction: Generate question-answer pairs from the provided content.
principles: |
  1. Questions should be clear and specific
  2. Answers should be comprehensive but concise
  3. Focus on key concepts and important details
examples: |
  [QUESTION] What is the main topic of this section? [/QUESTION]
  [ANSWER] The main topic is... [/ANSWER]
generation: |
  Content: {{document}}
  Generate a Q&A pair:
start_tags: ["[QUESTION]", "[ANSWER]"]
end_tags: ["[/QUESTION]", "[/ANSWER]"]
```

### Text Classification
```yaml
system: You are a text classification expert.
introduction: Classify the given text into predefined categories.
principles: |
  1. Read the text carefully
  2. Consider all categories
  3. Choose the most appropriate match
  4. Provide reasoning for your choice
examples: ""
generation: |
  Text: {{text}}
  Categories: {{categories}}
  
  Classification: {{category}}
  Reasoning: {{reasoning}}
start_tags: ["Classification:", "Reasoning:"]
end_tags: ["Reasoning:", ""]
```

### Evaluation and Judgment
```yaml
system: You are an impartial evaluator.
introduction: Evaluate and compare the quality of the provided responses.
principles: |
  1. Be objective and unbiased
  2. Consider multiple quality dimensions
  3. Provide specific feedback
  4. Justify your evaluation
examples: |
  [Example evaluation format]
generation: |
  Question: {{question}}
  Response A: {{response_a}}
  Response B: {{response_b}}
  
  Please evaluate:
start_tags: ["[Start of Evaluation]", "[Start of Verdict]"]
end_tags: ["[End of Evaluation]", "[End of Verdict]"]
```

This declarative approach to prompt configuration enables consistent, maintainable, and scalable LLM data generation across diverse domains and use cases in SDG Hub.