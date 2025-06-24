# Notebook Overview: Synthetic Data Generation for Document-Grounded Reasoning

This notebook demonstrates how to build a **synthetic dataset
specifically designed to train reasoning-capable language models**,
using the `sdg_hub` pipeline. It focuses on helping a student model
learn *how to reason* over complex documents by mimicking the structured
thinking patterns of a large teacher LLM.

### Key Highlights:

-   **Reasoning-Centric Pipeline**: The entire flow is optimized to
    generate examples that reflect *step-by-step reasoning*, rather than
    simple factual recall or shallow Q&A.
-   **Hands-On, Step-by-Step Construction**: We walk through building a
    reasoning-focused synthetic data pipeline---from swapping the
    teacher model to progressively modifying the knowledge flow for
    generating \"thinking data.\"
-   **Pipeline Extension with New Blocks**: The notebook shows how to
    extend an existing flow by injecting new functional blocks---such as
    post-processing to clean \"think\" prompts or regex parsers to
    structure model outputs.
-   **Grounded and Faithful Outputs**: Reasoning outputs remain aligned
    with the source documents, minimizing hallucinations and ensuring
    factual consistency.

This notebook is ideal for researchers and engineers using `sdg_hub` to
train or fine-tune models that reason---rather than simply
respond---when tackling document-grounded tasks.



``` python
# Let first instantiate teacher model using OpenAI client. We assume the teacher model is already deployed on vllm server.
# In this case we will be using a Nemotron Super model.
from datasets import load_dataset
from openai import OpenAI

endpoint = f"http://localhost:8000/v1"
openai_api_key = "EMPTY"
openai_api_base = endpoint

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)
teacher_model = client.models.list().data[0].id

teacher_model
```



We start by first importing all components we will need for running SDG:
the Flow, and the SDG class

``` python
from sdg_hub.flow import Flow
from sdg_hub.sdg import SDG
```



Our goal in this notebook is to perform **knowledge tuning using a
reasoning model**. In the previous knowledge tuning notebook, we saw how
to fine-tune a model on a user's document to improve its factual recall.
Here, we extend that approach by training a *reasoning model* on the
same document---so the model doesn\'t just remember facts, but also
learns to **reason through the content**.

We begin by **swapping the teacher model** to `Nemotron Super`, which
will be used to generate high-quality reasoning data.

If this model isn\'t already registered in `sdg_hub`, we'll need to add
it to the **Prompt Registry**. You can check the currently supported
templates in `src/sdg_hub/prompts.py`.

To register a new prompt template:

-   Import the `PromptRegistry` class.
-   Create a function that returns the chat template for your model
    (obtained via `tokenizer.chat_template`).
-   Use the `@PromptRegistry.register(...)` decorator to register it, as
    shown below.

In this example, we instantiate the tokenizer for Nemotron Super and
extract its chat formatting logic for integration into the pipeline.



``` python
from sdg_hub.prompts import PromptRegistry
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("nvidia/Llama-3_3-Nemotron-Super-49B-v1")
nemotron_chat_template = tokenizer.chat_template
nemotron_chat_template
```



``` jinja
'{{- bos_token }}{%- if messages[0][\'role\'] == \'system\' %}{%- set system_message = messages[0][\'content\']|trim %}{%- set messages = messages[1:] %}{%- else %}{%- set system_message = "" %}{%- endif %}{{- "<|start_header_id|>system<|end_header_id|>\\n\\n" }}{{- system_message }}{{- "<|eot_id|>" }}{%- for message in messages %}{%- if message[\'role\'] == \'assistant\' and \'</think>\' in message[\'content\'] %}{%- set content = message[\'content\'].split(\'</think>\')[-1].lstrip() %}{%- else %}{%- set content = message[\'content\'] %}{%- endif %}{{- \'<|start_header_id|>\' + message[\'role\'] + \'<|end_header_id|>\\n\\n\' + content | trim + \'<|eot_id|>\' }}{%- endfor %}{%- if add_generation_prompt %}{{- \'<|start_header_id|>assistant<|end_header_id|>\\n\\n\' }}{%- endif %}'
```



The `Nemotron Super` model supports two reasoning modes controlled via
the system message:

-   **Detailed Thinking ON**: The model is instructed to *think
    step-by-step before answering* (i.e., \"think before you answer\").
-   **Detailed Thinking OFF**: The model is prompted to *respond
    directly without internal reasoning*.

For our use case, we want to **always enable detailed thinking** to
ensure that the teacher model generates rich, multi-step reasoning
traces. To achieve this, we will **modify the chat template's Jinja
structure** so that the system message consistently sets
`detailed thinking: on`.

This updated template will then be returned in our registered chat
template function and integrated into the prompt registry, ensuring
every prompt sent to Nemotron Super reflects our reasoning-first
configuration.


``` python
@PromptRegistry.register("nvidia/Llama-3_3-Nemotron-Super-49B-v1")
def nemotron_chat_template():
    return """{{- bos_token }}
{{- "<|start_header_id|>system<|end_header_id|>\n\n" }}detailed thinking on{{- "<|eot_id|>" }}
{%- for message in messages %}
  {%- if message['role'] == 'assistant' and '</think>' in message['content'] %}
    {%- set content = message['content'].split('</think>')[-1].lstrip() %}
  {%- else %}
    {%- set content = message['content'] %}
  {%- endif %}
  {{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' + content | trim + '<|eot_id|>' }}
{%- endfor %}
{%- if add_generation_prompt %}
  {{- '<|start_header_id|>assistant<|end_header_id|>\n\n' }}
{%- endif %}"""
```



Next, we load the corpus and combine it with the seed examples to
construct the **final seed dataset**.

In `sdg_hub`, the seed data must adhere to a **strict column naming
convention**: Each column name must **exactly match** the variable names
expected in the prompt template of the first `LLMBlock` in the flow.

This alignment is crucial---every subsequent block in the pipeline
relies on the output columns of the previous block to match its own
input variable names. This ensures that prompts are populated correctly
and that the flow remains **fully compatible and composable** from block
to block.

Maintaining this column-to-variable consistency is essential for smooth
data propagation and accurate prompt construction across the entire
synthetic data generation pipeline.


``` python
# Create a simple dataset
from datasets import load_dataset
quality_corpus = load_dataset("zitongyang/entigraph-quality-corpus", split='train').remove_columns(['entity', 'entigraph']).rename_columns({'raw': 'document', 'uid': 'document_outline'})

# For knowledge tuning we need seed examples (teacher model will use this template to generate more data) to generate synthetic data. Write example QA/seed_examples
seed_examples = {
    "icl_document": (
      "The coastal town of Willow Creek, once renowned for its pristine beaches, now struggles with rampant pollution. Plastic debris and oil spills have devastated marine life, prompting a decline in tourism and fishing industries. Residents have organized weekly clean-up initiatives, but the scale of the problem overwhelms their efforts.",
      "Technologists at the local university have developed an AI-powered buoy system to combat this. The buoys, equipped with solar panels and filtration technology, can identify and absorb oil spills while collecting microplastics. Data from the buoys is shared publicly, raising awareness and pressuring corporations to adopt sustainable practices. Though costly, the project has sparked hope for revitalizing the ecosystem and economy."
    ),
    "icl_query_1": "How does the technological solution address the economic *and* environmental challenges highlighted in the document?",
    "icl_response_1": "The buoys directly mitigate environmental harm by absorbing spills and collecting plastics, which could restore marine life. Economically, this restoration might revive tourism and fishing, reversing their decline. Public data-sharing also pressures corporations, potentially reducing pollution and creating a sustainable economic cycle aligned with environmental health.",
    
    "icl_query_2": "What implicit values or priorities do the community’s actions (clean-up initiatives) and the technologists’ project reflect, and how do these align or contrast?",
    "icl_response_2": "Both reflect a prioritization of environmental stewardship and community engagement. The clean-ups emphasize collective responsibility, while the buoy project highlights innovation and systemic accountability (targeting corporations). They align in seeking solutions but contrast in scale and approach—manual vs. technological, local vs. systemic.",
    
    "icl_query_3": "Imagine the buoy project succeeds. What unintended consequences might arise from its impact, considering document's themes?",
    "icl_response_3": "Success could lead to reduced community participation in clean-up efforts if residents perceive the technology as a sufficient solution. Economically, revitalization might exacerbate existing resource strains (e.g., overwhelming local infrastructure) or create dependency on the buoy system, undermining long-term sustainability if the technology falters."

}

# Add seed examples to the our corpus
quality_corpus = quality_corpus.map(lambda x: seed_examples)
quality_corpus
```

 {.output .execute_result execution_count="14"}
    Dataset({
        features: ['document_outline', 'document', 'icl_document', 'icl_query_1', 'icl_response_1', 'icl_query_2', 'icl_response_2', 'icl_query_3', 'icl_response_3'],
        num_rows: 263
    })




Next we will take our knowledge flow and update the model-id to work
with new registered model



``` python
import yaml
# Load the exiting knowledge flow
with open("../../src/sdg_hub/flows/generation/knowledge/synth_knowledge1.5.yaml", "r") as f:
    flow_config = yaml.safe_load(f)

# Pretty print the flow
print(yaml.dump(flow_config, indent=4))
```



``` yaml
-   block_config:
        block_name: duplicate_document_col
        columns_map:
            document: base_document
    block_type: DuplicateColumns
-   block_config:
        block_name: gen_detailed_summary
        config_path: configs/knowledge/detailed_summary.yaml
        model_id: mistralai/Mixtral-8x7B-Instruct-v0.1
        output_cols:
        - summary_detailed
    block_type: LLMBlock
    gen_kwargs:
        max_tokens: 4096
        n: 50
        seed: 7452
        temperature: 0.7
-   block_config:
        block_name: gen_atomic_facts
        config_path: configs/knowledge/atomic_facts.yaml
        model_id: mistralai/Mixtral-8x7B-Instruct-v0.1
        output_cols:
        - summary_atomic_facts
    block_type: LLMBlock
    gen_kwargs:
        max_tokens: 4096
        seed: 7452
        temperature: 0.7
-   block_config:
        block_name: gen_extractive_summary
        config_path: configs/knowledge/extractive_summary.yaml
        model_id: mistralai/Mixtral-8x7B-Instruct-v0.1
        output_cols:
        - summary_extractive
    block_type: LLMBlock
    gen_kwargs:
        max_tokens: 4096
        seed: 7452
        temperature: 0.7
-   block_config:
        block_name: flatten_summary_columns
        value_name: summary
        var_cols:
        - summary_detailed
        - summary_extractive
        - summary_atomic_facts
        - base_document
        var_name: dataset_type
    block_type: FlattenColumnsBlock
-   block_config:
        block_name: rename_to_document_column
        columns_map:
            document: raw_document
            summary: document
    block_type: RenameColumns
-   block_config:
        block_name: knowledge generation
        config_path: configs/knowledge/generate_questions.yaml
        model_id: mistralai/Mixtral-8x7B-Instruct-v0.1
        output_cols:
        - question
        parser_kwargs:
            parser_name: custom
            parsing_pattern: \[(?:Question|QUESTION)\]\s*(.*?)\s*(?=\[(?:Question|QUESTION)\]|$)
    block_type: LLMBlock
    gen_kwargs:
        max_tokens: 100
        seed: 7452
        temperature: 0.7
-   block_config:
        block_name: knowledge generation
        config_path: configs/knowledge/generate_responses.yaml
        model_id: mistralai/Mixtral-8x7B-Instruct-v0.1
        output_cols:
        - response
    block_type: LLMBlock
    gen_kwargs:
        max_tokens: 2048
        seed: 7452
        temperature: 0.7
-   block_config:
        block_name: eval_faithfulness_qa_pair
        config_path: configs/knowledge/evaluate_faithfulness.yaml
        model_id: mistralai/Mixtral-8x7B-Instruct-v0.1
        output_cols:
        - explanation
        - judgment
    block_type: LLMBlock
    gen_kwargs:
        max_tokens: 2048
-   block_config:
        block_name: filter_faithfulness
        filter_column: judgment
        filter_value: 'YES'
        operation: operator.eq
    block_type: FilterByValueBlock
    drop_columns:
    - judgment
    - explanation
-   block_config:
        block_name: eval_relevancy_qa_pair
        config_path: configs/knowledge/evaluate_relevancy.yaml
        model_id: mistralai/Mixtral-8x7B-Instruct-v0.1
        output_cols:
        - feedback
        - score
    block_type: LLMBlock
    gen_kwargs:
        max_tokens: 2048
-   block_config:
        block_name: filter_relevancy
        convert_dtype: float
        filter_column: score
        filter_value: 2.0
        operation: operator.eq
    block_type: FilterByValueBlock
    drop_columns:
    - feedback
    - score
-   block_config:
        block_name: eval_verify_question
        config_path: configs/knowledge/evaluate_question.yaml
        model_id: mistralai/Mixtral-8x7B-Instruct-v0.1
        output_cols:
        - explanation
        - rating
    block_type: LLMBlock
    gen_kwargs:
        max_tokens: 2048
-   block_config:
        block_name: filter_verify_question
        convert_dtype: float
        filter_column: rating
        filter_value: 1.0
        operation: operator.eq
    block_type: FilterByValueBlock
    drop_columns:
    - explanation
    - rating
    - __index_level_0__
```



``` python
# Define the model ID to be used across blocks
MODEL_ID = "nvidia/Llama-3_3-Nemotron-Super-49B-v1"

# Print the first block
print("Original model-id:",flow_config[1])
# Update the model-id
for block in flow_config:
    if block['block_type'] == 'LLMBlock':
        block['block_config']['model_id'] = MODEL_ID

# Print the first block
print("Updated model-id:",flow_config[1])

# Save the updated flow
with open("flows/synth_knowledge1.5_updated.yaml", "w") as f:
    yaml.dump(flow_config, f)
```



``` python
# Create and run the flow
knowledge_1_5_flow = Flow(client).get_flow_from_file("flows/synth_knowledge1.5_updated.yaml")
knowledge_1_5_sdg = SDG(
    flows=[knowledge_1_5_flow],
    num_workers=1,
    batch_size=1
)

# Generate summaries
generated_summaries = knowledge_1_5_sdg.generate(quality_corpus.select([1]), checkpoint_dir="simple_output")
```



``` python
print("Generated Summary:", generated_summaries[0]['document'])
```



``` text
Generated Summary:
<think>
Okay, let's start by reading through the document carefully. The title is "Fight Clubbed" by David Plotz, from 1999. The main point seems to be comparing the movie Fight Club with the real Ultimate Fighting Championship (UFC) and how the ...</think>

**Detailed Summary of "Fight Clubbed" by David Plotz (1999)**
...
- Contrast with *Fight Club*’s cultural impact.  
- Underground survival of the sport.  

**No New Information Added; All Document Points Retained.**
```



### Understanding PostProcessThinkingBlock

The PostProcessThinkingBlock is a custom block that helps process the
output from models that use thinking/reasoning tags in their responses.
This block is particularly useful when working with models that show
their reasoning process using special tags like `<think>` and
`</think>`. This the text between the thinking tags

Here\'s how to create and use a PostProcessThinkingBlock:


``` python
from sdg_hub.blocks import BlockRegistry, Block
from datasets import Dataset

@BlockRegistry.register("PostProcessThinkingBlock")
class PostProcessThinkingBlock(Block):
    def __init__(self, block_name: str, column_name: str) -> None:
        super().__init__(block_name=block_name)  
        self.column_name = column_name
    
    def generate(self, samples: Dataset):
        def post_process_thinking(x):
            if '</think>' in x[self.column_name]:
                x[self.column_name] = x[self.column_name].split('</think>')[-1].lstrip()
            return x
        samples = samples.map(post_process_thinking)
        return samples

# Helper function to create a post-processing block configuration
def create_postprocess_block(block_name, column_name):
    return {
        "block_type": "PostProcessThinkingBlock",
        "block_config": {
            "block_name": block_name,
            "column_name": column_name
        }
    }
```


    /home/lab/.conda/envs/research_sdg/lib/python3.10/site-packages/tqdm/auto.py:21: TqdmWarning: IProgress not found. Please update jupyter and ipywidgets. See https://ipywidgets.readthedocs.io/en/stable/user_install.html
      from .autonotebook import tqdm as notebook_tqdm




To use this block in a flow, you would add it after an LLMBlock that
generates the thinking tags (`<think>`{=html}\...`</think>`{=html})
output:

``` yaml
- block_config:
    block_name: duplicate_document_col
    columns_map:
      document: base_document
  block_type: DuplicateColumns
- block_config:
    block_name: gen_detailed_summary
    config_path: configs/knowledge/detailed_summary.yaml
    model_id: nvidia/Llama-3_3-Nemotron-Super-49B-v1
    output_cols:
    - summary_detailed
  block_type: LLMBlock
  gen_kwargs:
    max_tokens: 4096
    n: 50
    seed: 7452
    temperature: 0.7

- block_type: PostProcessThinkingBlock
  block_config:
    block_name: process_thinking
    column_name: summary_detailed
```

The PostProcessThinkingBlock:

1.  Takes a column name as input that contains the thinking output
2.  Looks for the `</think>` tag in the text
3.  If found, removes everything before and including the `</think>` tag
4.  Returns the cleaned output

This is particularly useful when you want to:

-   Remove the model\'s thinking process from the final output
-   Clean up responses that include reasoning steps
-   Extract only the final answer from a chain-of-thought response

Next lets update the flow yaml to use the post-process thinking Block.

We want to remove all the thinking block from all responses except the
LLMBlock that generates the responses for the generated question. This
is important as we want to train a student reasoning model to always
output in this format:

``` text
Question:
...question...
Model Response:
<think>
...
</think>
...final answer
...
```


``` python
# Update the flow to use the PostProcessThinkingBlock
with open("flows/synth_knowledge1.5_updated.yaml", "r") as f:
    flow_config = yaml.safe_load(f)

# Add the PostProcessThinkingBlock after all LLMBlocks except the LLMBlock that generates the response
new_flow_config = []
for block in flow_config:
    
    # Add post-process thinking block after the LLMBlock except for the generate_response block
    new_flow_config.append(block)
    if block['block_type'] != 'LLMBlock' :
        continue
    if 'generate_questions.yaml' in block['block_config']['config_path']:
        new_flow_config[-1]['gen_kwargs']['max_tokens'] = 1024

    # Add post-process thinking block after the LLMBlock except for the generate_response block
    if 'generate_responses.yaml' not in block['block_config']['config_path']:
            new_flow_config.append(
                create_postprocess_block(
                    block_name="process_thinking",
                    column_name=block['block_config']['output_cols'][0]
                    )
                    )

# Save the updated flow
with open("flows/synth_knowledge1.5_updated_with_post_process_thinking.yaml", "w") as f:
    yaml.dump(new_flow_config, f)
```



``` python
# Create and run the flow
knowledge_1_5_flow = Flow(client).get_flow_from_file("flows/synth_knowledge1.5_updated_with_post_process_thinking.yaml")
knowledge_1_5_sdg = SDG(
    flows=[knowledge_1_5_flow],
    num_workers=1,
    batch_size=1
)

# Generate summaries
generated_summaries = knowledge_1_5_sdg.generate(quality_corpus.select([1]), checkpoint_dir="simple_output")
```



Next we need to update out pipeline to fix the parsing order of the
model response.

### Current Setup

Right now, the typical definition of our **LLMBlock** for generating
questions looks like this:

``` yaml
- block_config:
    block_name: knowledge generation
    config_path: configs/knowledge/generate_questions.yaml
    model_id: nvidia/Llama-3_3-Nemotron-Super-49B-v1
    output_cols:
      - question
    parser_kwargs:
      parser_name: custom
      parsing_pattern: \[(?:Question|QUESTION)\]\s*(.*?)\s*(?=\[(?:Question|QUESTION)\]|$)
  block_type: LLMBlock
  gen_kwargs:
    max_tokens: 100
    seed: 7452
    temperature: 0.7
- block_type: PostProcessThinkingBlock
  block_config:
    block_name: process_thinking
    column_name: question
```

This block is currently followed by a **post-processing block**, which
comes *after* regex parsing.

### Current Flow

    LLM Output (e.g., "<think>...</think>Q1 ... Q2 ... Q3 ...") 
    → Regex Parse (extract [Q1, Q2, Q3, ...]) 
    → Post-Process Thinking (e.g., clean or enrich Q1, Q2, Q3)

### The Problem

This ordering introduces ambiguity in the downstream output. The regex
may extract incomplete or malformed questions, especially if the
`<think>` section or question formatting isn\'t perfectly consistent.
Additionally, applying post-processing *after* parsing can cause
inconsistencies in the final output quality.

### Proposed Change

We want to **remove the post-processing block after regex parsing** and
instead **apply post-processing directly on the raw LLM output before
parsing**. This ensures better control and guarantees the integrity of
the final parsed questions.

### Updated Flow

    LLM Output (e.g., "<think>...</think>Q1 ... Q2 ... Q3 ...") 
    → Post-Process Thinking (clean/improve Q1, Q2, Q3 format in raw text) 
    → Regex Parse (extract [Q1, Q2, Q3, ...])

This adjustment improves reliability and consistency across generated
data samples.

How do we achieve this:

-   We will first remove all the regex parsing from the LLM Block by
    removing the `parser_kwargs` from the corresponding LLM Blocks
-   We will introduce new Block for regular expression and add it after
    the LLMBlock, PostProcessThinkingBlock pair

``` python
import re
from typing import List

@BlockRegistry.register("RegexParserBlock")
class RegexParserBlock(Block):
    """Block for parsing text using regular expressions and cleaning up tags.
    
    This block takes text input, applies regex pattern matching to extract structured data,
    and optionally cleans up specified tags from the output.
    """
    def __init__(self, block_name: str, column_name: str, parsing_pattern: str="", parser_cleanup_tags: List[str]=[], output_cols: List[str]=[]) -> None:
        """Initialize the RegexParserBlock.
        
        Args:
            block_name: Name identifier for this block
            column_name: Name of column containing text to parse
            parsing_pattern: Regex pattern to use for parsing
            parser_cleanup_tags: List of tags to remove from parsed output
            output_cols: Names of columns to store parsed outputs
        """
        super().__init__(block_name=block_name)
        self.column_name = column_name
        self.parsing_pattern = parsing_pattern
        self.parser_cleanup_tags = parser_cleanup_tags
        self.output_cols = output_cols

    def generate(self, samples: Dataset):
        """Process samples by parsing text and cleaning tags.
        
        Args:
            samples: Dataset containing samples to process
            
        Returns:
            Dataset with parsed and cleaned outputs
        """
        if self.parsing_pattern:
            # Parse text using regex pattern
            new_data = []
            for sample in samples:
                parsed_outputs = self._parse(sample[self.column_name])
                
                # Align all outputs to same length by taking max length
                max_length = max(len(value) for value in parsed_outputs.values())
                for values in zip(*(lst[:max_length] for lst in parsed_outputs.values())):
                    new_data.append({**sample, **dict(zip(parsed_outputs.keys(), values))})
            samples = Dataset.from_list(new_data)
            
        # Clean up any specified tags from output
        if self.parser_cleanup_tags:
            for clean_tag in self.parser_cleanup_tags:
               samples = samples.map(lambda x: {column_name: x[column_name].replace(clean_tag, "") for column_name in self.output_cols})
        return samples

    def _parse(self, generated_string):
        """Parse text using regex pattern to extract structured data.
        
        Args:
            generated_string: Text string to parse
            
        Returns:
            Dictionary mapping output column names to lists of parsed values
        """
        pattern = re.compile(self.parsing_pattern, re.DOTALL)
        all_matches = pattern.findall(generated_string)
        matches = {column_name: [] for column_name in self.output_cols}
        
        # Handle both single group and multiple capture group matches
        if all_matches and isinstance(all_matches[0], tuple):
            for match in all_matches:
                for column_name, value in zip(self.output_cols, match):
                    value = value.strip()
                    matches[column_name].append(value)
        else:
            matches[self.output_cols[0]] = (
                [match.strip() for match in all_matches] if all_matches else []
            )
        return matches

# Helper function to create a regex parser block configuration
def create_regex_parser_block(block_name, column_name, parsing_pattern, cleanup_tags, output_cols):
    return {
        "block_type": "RegexParserBlock",
        "block_config": {
            "block_name": block_name,
            "column_name": column_name,
            "parsing_pattern": parsing_pattern,
            "parser_cleanup_tags": cleanup_tags,
            "output_cols": copy.deepcopy(output_cols)
        }
    }
```

``` python
import copy

# Update the flow to use the new RegexParserBlock
with open("flows/synth_knowledge1.5_updated.yaml", "r") as f:
    flow_config = yaml.safe_load(f)

regex_patterns = {
    "configs/knowledge/generate_questions.yaml": ("\\[(?:Question|QUESTION)\\]\\s*(.*?)\\s*(?=\\[(?:Question|QUESTION)\\]|$)", ["[END]"]),
    "configs/knowledge/generate_answers.yaml": ("", ["[END]", "[ANSWER]", "assistant"]),
}
new_flow_config = []
# Update the flow to use the new RegexParserBlock
for block in flow_config:
    
    # Add post-process thinking block after the LLMBlock except for the generate_response block
    new_flow_config.append(block)
    if block['block_type'] != 'LLMBlock' :
        continue
    if 'generate_questions.yaml' in block['block_config']['config_path']:
        new_flow_config[-1]['gen_kwargs']['max_tokens'] = 1024
    # Remove parser_kwargs for generate_answers.yaml and generate_questions.yaml
    if 'generate_answers.yaml' in block['block_config']['config_path'] or 'generate_questions.yaml' in block['block_config']['config_path']:
        del block['block_config']['parser_kwargs']

    # Add post-process thinking block after the LLMBlock except for the generate_response block
    if 'generate_responses.yaml' not in block['block_config']['config_path']:
            # Create post-processing block for summaries
            new_flow_config.append(
                create_postprocess_block(
                    block_name="process_thinking",
                    column_name=block['block_config']['output_cols'][0]
                    )
                    )
    # Add regex parser block after the post-process thinking block
    if 'generate_answers.yaml' in block['block_config']['config_path'] or 'generate_questions.yaml' in block['block_config']['config_path']:
        new_flow_config.append(
            create_regex_parser_block(
                block_name="regex_parser",
                column_name=block['block_config']['output_cols'][0],
                parsing_pattern=regex_patterns[block['block_config']['config_path']][0],
                cleanup_tags=regex_patterns[block['block_config']['config_path']][1],
                output_cols=block['block_config']['output_cols']
            )
        )

# Save the updated flow
with open("flows/synth_knowledge1.5_updated_with_post_process_thinking_and_regex_parser.yaml", "w") as f:
    yaml.dump(new_flow_config, f, default_style='"')
            
```



You\'ll also notice that we added a `parser_cleanup_tags` parameter to
our `RegexParserBlock`. This works similarly to the
`parser_cleanup_tags` used in `LLMBlock`. It defines a list of tags or
strings that may appear in the model's response and should be **cleaned
up before parsing**. Internally, this is equivalent to applying:

``` python
response = response.replace(tag, "")
```

for each tag in the `parser_cleanup_tags` list. This ensures cleaner and
more consistent outputs, especially when the model appends artifacts
like `[END]` or `[THINK]`.

Additionally, you might observe that the `parsing_pattern` for the
`generate_answer` block is intentionally left **empty**. This is because
we want to **retain the raw output** of the model without applying any
regex-based extraction---useful when no structural parsing is required
or when the output is already in the desired format.



``` python
# Create and run the flow
knowledge_1_5_flow = Flow(client).get_flow_from_file("flows/synth_knowledge1.5_updated_with_post_process_thinking_and_regex_parser.yaml")
knowledge_1_5_sdg = SDG(
    flows=[knowledge_1_5_flow],
    num_workers=1,
    batch_size=1
)

# Generate summaries
generated_summaries = knowledge_1_5_sdg.generate(quality_corpus.select([1]), checkpoint_dir="simple_output")
```

``` python

print(generated_summaries['question'][1])
```

 {.output .stream .stdout}
    What implicit double standards in American cultural and political discourse are highlighted by the comparison between the UFC’s treatment and the acceptance of boxing, despite the UFC’s claimed safety advantages?  




### Extending the Pipeline with Instruction-Driven Summaries

Because a reasoning model can **brainstorm multiple viewpoints**, we'll
ask it to invent its own summarization *instructions* before it writes
any summaries. For instance, given a document on EVs and their
environmental impact, the model might propose:

1.  "Summarize as an academic essay."
2.  "Highlight key environmental implications."
3.  "Provide a comparative analysis."
4.  "Rewrite as an investor pitch."
5.  *...and so on.*

We'll implement this in two steps, each powered by its own prompt:

------------------------------------------------------------------------

#### 1. `generate_summary_inst.yaml` -- *Create 10 diverse instructions* {#1-generate_summary_instyaml--create-10-diverse-instructions}

``` yaml
system: You are an AI assistant that is expert at summarizing text.

introduction: |
  Given the document below, generate 10 short, distinct instructions for summarizing it.
  Each instruction should vary in perspective, tone, or purpose.

principles: ""

examples: |
  Example:
  1. Summarize the article in simple terms for a 10-year-old.
  2. Highlight the major arguments and counterarguments.
  3. Provide a summary focusing on implications for future research.

generation: |
  Document:
  {{document_outline}}

  {{document}}

  Now generate 10 diverse summarization instructions.

start_tags: [""]
end_tags: [""]
```

------------------------------------------------------------------------

#### 2. `generate_summary.yaml` -- *Write a summary for each instruction* {#2-generate_summaryyaml--write-a-summary-for-each-instruction}

``` yaml
system: You are an AI assistant that is expert at summarizing text.

introduction: |
  Using the instruction below, summarize the document:
  {{summary_instruction}}

principles:
  - Include as much of the document as possible to create a comprehensive summary.
  - If tables are present, include all table data.

examples: ""

generation: |
  Document:
  {{document_outline}}
  {{document}}

start_tags: [""]
end_tags: [""]
```

------------------------------------------------------------------------

### Flow Updates

To support these prompts, we add two new processing stages and remove
the old single-summary block:

  ------------------------------------------------------------------------------------------
  \#   Block                          Purpose
  ---- ------------------------------ ------------------------------------------------------
  1    **LLMBlock --                  Generate the 10-item instruction list.
       `gen_summary_instructions`**   

  2    **PostProcessThinkingBlock**   Capture the model's internal "thinking" for
                                      transparency/debugging.

  3    **RegexParserBlock**           Extract the numbered instructions into a clean
                                      `summary_instruction` column (`parser_cleanup_tags`
                                      handle stray tokens such as `[END]`).

  4    **LLMBlock --                  Produce one summary per instruction.
       `gen_detailed_summary`**       

  5    **PostProcessThinkingBlock**   Record the reasoning steps used to craft each summary.
  ------------------------------------------------------------------------------------------

With these two additional blocks---and by removing the original
one-size-fits-all summarization step---the pipeline now:

1.  **Brainstorms** multiple summarization angles.
2.  **Parses** and stores those angles cleanly.
3.  **Writes** a tailored summary for each angle, capturing the model's
    reasoning throughout.

This upgrade lets us generate richer, more varied training data that
showcases the model's reasoning abilities instead of a single static
summary.

``` python
# Import required libraries
import yaml
import copy

# Helper function to create an LLM block configuration
def create_llm_block(block_name, config_path, output_cols, model_id, gen_kwargs):
    return {
        "block_type": "LLMBlock",
        "block_config": {
            "block_name": block_name,
            "config_path": config_path,
            "model_id": model_id,
            "output_cols": output_cols,
            "gen_kwargs": gen_kwargs
        }
    }

# Create block for generating summary instructions
summary_inst_block = create_llm_block(
    block_name="gen_summary_instructions",
    config_path="prompts/generate_summary_inst.yaml",
    output_cols=["summary_instruction"],
    model_id=MODEL_ID,
    gen_kwargs={
        "max_tokens": 4096,
        "temperature": 0.6,
        "top_p": 0.95,
        "n": 2,
        "seed": 43146
    }
)

# Create post-processing block for summary instructions
thinking_block_inst = create_postprocess_block(
    block_name="process_thinking",
    column_name=summary_inst_block["block_config"]["output_cols"][0]
)

# Create regex parser block to extract numbered instructions
regex_parser_block = create_regex_parser_block(
    block_name="regex_parser",
    column_name=summary_inst_block["block_config"]["output_cols"][0],
    parsing_pattern="(?:^|\\n)\\s*\\d+[\\.\\)]\\s*([^\\n]+)",
    cleanup_tags=["[END]"],
    output_cols=summary_inst_block["block_config"]["output_cols"]
)

# Create block for generating detailed summaries
summary_block = create_llm_block(
    block_name="gen_detailed_summary",
    config_path="prompts/generate_summary.yaml",
    output_cols=["summary"],
    model_id=MODEL_ID,
    gen_kwargs={
        "max_tokens": 4096,
        "temperature": 0.6,
        "top_p": 0.95,
        "n": 1
    }
)

# Create post-processing block for summaries
thinking_block_summary = create_postprocess_block(
    block_name="process_thinking",
    column_name=summary_block["block_config"]["output_cols"][0]
)

# Define input and output YAML paths
input_yaml_path = "flows/synth_knowledge1.5_updated_with_post_process_thinking_and_regex_parser.yaml"
output_yaml_path = "flows/synth_knowledge1.5_updated_with_post_process_thinking_and_regex_parser_with_summary_instruction.yaml"

# Load existing flow configuration
with open(input_yaml_path, 'r') as f:
    flow_config = yaml.safe_load(f)

# Find the index of the flatten_summary_columns block
for idx, block in enumerate(flow_config):
    if 'block_config' in block and block['block_config'].get('block_name') == 'flatten_summary_columns':
        break

# Create list of new blocks to inject
injected_blocks = [
    summary_inst_block,
    thinking_block_inst,
    regex_parser_block,
    summary_block,
    thinking_block_summary
]

# Insert new blocks after the first block and before the flatten block
flow_config = flow_config[:1] + injected_blocks + flow_config[idx+1:]

# Save the modified flow configuration
with open(output_yaml_path, 'w') as f:
    yaml.dump(flow_config, f, default_style='"')
```



``` python
# Create and run the flow
knowledge_1_5_flow = Flow(client).get_flow_from_file("flows/synth_knowledge1.5_updated_with_post_process_thinking_and_regex_parser_with_summary_instruction.yaml")
knowledge_1_5_sdg = SDG(
    flows=[knowledge_1_5_flow],
    num_workers=1,
    batch_size=1
)

# Generate summaries
generated_summaries = knowledge_1_5_sdg.generate(quality_corpus.select([1]), checkpoint_dir="simple_output")
```



``` python
print("Generated Summary Instruction:\n", generated_summaries['summary_instruction'][0])
print("Corresponding generated summary:\n", generated_summaries['document'][0])
```



``` text
Generated Summary Instruction:
**Summarize for a debate competition**: Focus on the ethical arguments comparing UFC’s safety record to boxing and counterarguments from critics like Sen. McCain.

Corresponding generated summary:
**Summarized Debate Summary: UFC vs. Boxing Safety & Ethics**  

**Background**: The Ultimate Fighting Championship (UFC), founded in 1993, initially featured no-holds-barred cage fights, attracting millions with its brutality ("the damage"). However, beneath the 
...
**Conclusion**: The UFC’s demise reflects not ethical failings but political and cultural bias. Its underground persistence raises questions about America’s tolerance for innovation in combat sports versus entrenched preferences.  

*(No tables in the document; all key data and quotes incorporated.)*
```



``` python
print("Question:\n", generated_summaries['question'][0])
print("Answer:\n", generated_summaries['response'][0])
```



``` text
Question:
How do the ethical arguments for UFC’s safety (e.g., submission-focused endings, adaptive rules) contrast with critics’ moral objections, and what underlying values drive this divergence?  

Answer:
assistant<think>
Okay, let's tackle this question. The user wants to know how the ethical arguments for UFC's safety contrast with critics' moral objections and what underlying values cause this divergence.
...
Check the examples provided to ensure the format and style match. The answer should directly address both parts of the question: contrast in arguments and underlying values.
</think>

The ethical arguments for UFC’s safety emphasize empirical safety records and adaptive governance, contrasting sharply with critics’ moral objections rooted in cultural perception and tradition. UFC advocates highlight **submission-focused endings** (reducing prolonged brain 
...
whereas critics value **traditional moral norms** and societal comfort with established sports, even when safety records contradict these preferences.  
[END]
```



## Conclusion: Building Reasoning-Centric Synthetic Data with `sdg_hub`

In this notebook, we extended the traditional knowledge tuning workflow
to support **document-grounded reasoning tasks** using `sdg_hub`. By
swapping in a reasoning-capable teacher model (Nemotron Super) and
customizing its prompt template, we enabled richer, multi-step thinking
in our synthetic data generation pipeline.

Key contributions of this workflow include:

-   **Instruction-first summarization**: The model is prompted to
    brainstorm diverse ways to summarize a document, reflecting
    different tones, goals, and reasoning perspectives.
-   **Multi-block reasoning flow**: We composed the pipeline using
    `LLMBlock`, `PostProcessThinkingBlock`, and `RegexParserBlock` to
    simulate how a human might structure thoughts, write summaries, and
    clean up output.
-   **Faithful alignment between prompts and data**: Each block
    maintained strict variable alignment, enabling seamless chaining of
    tasks and clean YAML integration.

This reasoning-aware seed dataset is now ready to be used for
**fine-tuning student models**---training them not only to recall facts,
but to reason through complex text in diverse, instruction-driven ways.



## For reference: scripts for running SDG, starting student model and training

### Start Teacher Model

``` shell
export HUGGINGFACE_HUB_CACHE="/new_data/hf_cache"   
export HF_DATASETS_CACHE="/dev/shm/hf"
export HF_HOME="/new_data/hf_cache"
export HF_MODEL_CACHE="/new_data/hf_cache"

for i in $(seq 0 2 6); do
    port=$((8000 + i/2))
    CUDA_VISIBLE_DEVICES=$i,$((i+1)) python -m vllm.entrypoints.openai.api_server \
        --model nvidia/Llama-3_3-Nemotron-Super-49B-v1 \
        --dtype float16 \
        --tensor-parallel-size 2 \
        --port $port \
        --trust-remote-code > log_$((i/2+1)).log 2>&1 &
done
```

### Run SDG in parallel

``` shell
# Get dataset size and save into variable
dataset_size=$(wc -l seed_data.jsonl | awk '{print $1}')
number_of_processes=4
port=8000
for i in {0..4}; do
    # Continue until i=4
    if [ $i -eq 4 ]; then
        break
    fi
    dataset_start_index=$((i * dataset_size / number_of_processes))
    dataset_end_index=$((dataset_start_index + dataset_size / number_of_processes))
    python -m sdg_hub.flow_runner --ds_path  seed_data.jsonl \
        --bs 2 --num_workers 10 \
        --save_path data/knowledge/quality/synth_knowledge_reasoning/gen.jsonl \
        --flow flows/synth_knowledge1.5_updated_with_post_process_thinking_and_regex_parser_with_summary_instruction.yaml \
        --endpoint http://localhost:$port/v1 \
        --checkpoint_dir data/knowledge/quality/synth_knowledge_reasoning/data_checkpoints \
        --save_freq 1000 \
        --dataset_start_index $dataset_start_index \
        --dataset_end_index $dataset_end_index > run_sdg_$i.log 2>&1 &
    echo "Starting process $i with dataset from $dataset_start_index to $dataset_end_index on port $port"
    port=$((port + 1))
done
```




