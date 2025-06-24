# Documentation: src/sdg_hub/blocks/llmblock.py - LLM Interaction Blocks

This module provides core "Block" implementations for interacting with Large Language Models (LLMs). It includes the `LLMBlock` for standard text generation and the `ConditionalLLMBlock` for more dynamic, condition-based prompt selection.

---

## Module Overview

The `llmblock.py` module facilitates communication with LLMs, encompassing prompt formatting, making API calls, parsing responses, and structuring the generated data.

* **SPDX-License-Identifier**: `Apache-2.0`

---

## Imports

The module leverages several libraries and local components:

* **Standard Library**:
    * `typing`: For type hinting (`Any, Dict, List, Optional, Union`).
    * `json`: Though not directly used in the provided snippet, often relevant for LLM I/O.
    * `re`: For regular expression operations, crucial for parsing LLM outputs.
* **Third-Party**:
    * `datasets.Dataset`: From the Hugging Face `datasets` library, used for handling input and output data.
    * `jinja2.Template`: For rendering prompt templates.
    * `openai`: The official OpenAI Python client library for interacting with LLM APIs.
* **Local Project Modules**:
    * `.block.Block`: The base `Block` class from which LLM blocks inherit.
    * `..logger_config.setup_logger`: For creating a structured logger instance.
    * `..registry.BlockRegistry, ..registry.PromptRegistry`: For registering block types and accessing prompt rendering utilities.

---

## Logger Configuration

A logger is set up for this module:

```python
logger = setup_logger(__name__)
```

---

## Helper Function: `server_supports_batched`

This function checks if the connected LLM server supports batched requests and the `n` parameter (for generating multiple completions per prompt).

```python
def server_supports_batched(client: openai.OpenAI, model_id: str) -> bool:
    # ... (implementation details below)
    pass
```

* **Purpose**: To determine if the LLM server can efficiently process multiple prompts in one go and/or generate multiple responses for each prompt using the `n` parameter. This allows for optimized API usage.
* **Parameters**:
    * `client` (openai.OpenAI): The OpenAI client instance.
    * `model_id` (str): The identifier of the LLM model to be tested.
* **Logic**:
    1.  It first checks if a cached result `client.server_supports_batched` exists and returns it if available.
    2.  If not cached, it makes a test call to `client.completions.create` with two dummy prompts (`["test1", "test2"]`) and `n=3`.
    3.  It expects `2 * 3 = 6` choices in the response if batching and `n` are fully supported.
    4.  If an `openai.InternalServerError` occurs (some servers might not support this kind of batched request), it assumes batching is not supported.
    5.  The result (`True` or `False`) is cached on the `client` object as `server_supports_batched` and returned.
* **Returns**: `bool` - `True` if batched inputs and `n` parameter are supported, `False` otherwise.

---

## `LLMBlock` Class

This is the primary block for most LLM-based text generation tasks.

```python
@BlockRegistry.register("LLMBlock")
class LLMBlock(Block):
    # ... (implementation details below)
    pass
```

* **Inheritance**: Inherits from `Block`.
* **Registration**: Registered with `BlockRegistry` under the name "LLMBlock".
* **Purpose**: Encapsulates the logic for generating text using an LLM, including dynamic prompt formatting from a configuration, making calls to an LLM endpoint, and parsing the structured data from the LLM's raw output.

### `__init__(...)`

The constructor initializes the `LLMBlock`.

* **Parameters**:
    * `block_name` (str): Name of the block instance.
    * `config_path` (str): Path to the YAML configuration file containing prompt structure and other settings.
    * `client` (openai.OpenAI): An initialized OpenAI client instance.
    * `output_cols` (List[str]): A list of names for the output columns where parsed LLM generations will be stored.
    * `parser_kwargs` (Dict[str, Any], optional): Arguments for the output parser (e.g., `parser_name`, `parsing_pattern`, `parser_cleanup_tags`). Defaults to `{}`.
    * `model_prompt` (str, optional): A template string for the final model prompt, often a simple wrapper like `"{prompt}"`. Defaults to `"{prompt}"`.
    * `model_id` (Optional[str], optional): The specific model ID to use. If `None`, it defaults to the first model ID reported by the `client`.
    * `**batch_kwargs` (Dict[str, Any]): Additional keyword arguments, primarily for batch processing settings.
* **Initialization Steps**:
    1.  Calls `super().__init__(block_name)`.
    2.  Loads the YAML configuration from `config_path` into `self.block_config` using `_load_config()`.
    3.  Defines `self.prompt_struct`: a multi-line string template that combines `system`, `introduction`, `principles`, `examples`, and `generation` parts from the loaded configuration.
    4.  Creates `self.prompt_template`: a `jinja2.Template` object from `self.prompt_struct`, filling missing keys from `block_config` with empty strings.
    5.  Stores `client`, `model_id` (resolving default if needed), `model_prompt`, and `output_cols`.
    6.  Extracts batch processing parameters from `batch_kwargs.get("batch_kwargs", {})` into `self.batch_params`.
    7.  Extracts parser settings from `parser_kwargs`: `parser_name`, `parsing_pattern`, `parser_cleanup_tags`.
    8.  Sets `self.defaults`: a dictionary of default parameters for LLM generation (e.g., `model`, `temperature`, `max_tokens`).
    9.  Calls `server_supports_batched(client, self.model)` to determine and store `self.server_supports_batched`.

### `_extract_matches(...)`

A helper method to extract substrings using regular expressions based on start and end tags.

```python
    def _extract_matches(
        self, text: str, start_tag: Optional[str], end_tag: Optional[str]
    ) -> List[str]:
        # ...
        pass
```
* **Purpose**: To find and extract specific pieces of text from a larger string, typically an LLM's raw output, based on provided delimiter tags.
* **Parameters**:
    * `text` (str): The input string to search within.
    * `start_tag` (Optional[str]): The starting delimiter.
    * `end_tag` (Optional[str]): The ending delimiter.
* **Returns**: `List[str]` - A list of extracted substrings (stripped of whitespace). If no tags are provided, returns the entire input text as a single-item list.

### `_parse(...)`

Parses the raw generated string from the LLM into a structured dictionary.

```python
    def _parse(self, generated_string: str) -> dict:
        # ...
        pass
```

* **Purpose**: To transform the unstructured text output of an LLM into a dictionary where keys correspond to `self.output_cols`.
* **Logic**:
    * **Custom Parser**: If `self.parser_name` is "custom", it uses `self.parsing_pattern` (a regex) to find all matches.
        * If the regex has multiple capture groups, it zips these groups with `self.output_cols`.
        * Values are stripped and cleaned using `self.parser_cleanup_tags`.
        * Each key in the `matches` dictionary will have a list of corresponding extracted values.
    * **Tag-Based Parser**: Otherwise (default behavior), it iterates through `start_tags` and `end_tags` (expected to be lists in `self.block_config`, corresponding to each `output_col`) and uses `self._extract_matches()` for each pair to populate the `matches` dictionary.
* **Returns**: `dict` - A dictionary where keys are column names from `self.output_cols` and values are lists of parsed strings.

### `_format_prompt(...)`

Formats the input sample into the final prompt string to be sent to the LLM.

```python
    def _format_prompt(self, sample: Dict) -> str:
        # ...
        pass
```

* **Purpose**: To construct the complete prompt using the block's Jinja2 template and the input data.
* **Logic**:
    1.  Renders `self.prompt_template` (the detailed template from `config_path`) using the input `sample` data.
    2.  The result is then passed to `PromptRegistry.render_template` along with `self.model_prompt` (e.g., `"{prompt}"`). This suggests `PromptRegistry` might add further global formatting or instructions (like "generation prompts").
* **Returns**: `str` - The fully formatted prompt string.

### `_generate(...)`

Handles the actual API call(s) to the LLM to get text generations.

```python
    def _generate(self, samples: Dataset, **gen_kwargs: Dict[str, Any]) -> list:
        # ...
        pass
```

* **Purpose**: To send formatted prompts to the LLM and retrieve the generated text(s).
* **Parameters**:
    * `samples` (Dataset): A Hugging Face Dataset containing the input samples (used to create prompts).
    * `**gen_kwargs` (Dict[str, Any]): Keyword arguments for LLM generation (e.g., `temperature`, `max_tokens`, `n`).
* **Logic**:
    1.  Formats prompts for all input `samples` using `self._format_prompt()`.
    2.  Merges `self.defaults` with `gen_kwargs` to get the final generation arguments.
    3.  **Batched Execution**: If `self.server_supports_batched` is `True`, it makes a single API call to `self.client.completions.create()` with all prompts.
    4.  **Sequential Execution**: If not batched, it iterates through each prompt and makes an individual API call. If `n` (number of completions per prompt) is greater than 1, it makes `n` separate calls for that single prompt.
    5.  **Stop Token Handling**: If `stop` sequences are provided in `gen_kwargs`, the block appends these stop sequences to the generated text. This is because some LLM APIs (like OpenAI's) might not include the stop sequence itself in the output `text` field. Adding it back ensures consistency for downstream parsing.
* **Returns**: `list` - A list of raw generated text strings from the LLM.

### `generate(...)`

The main public method of the `LLMBlock` that orchestrates the entire generation process.

```python
    def generate(self, samples: Dataset, **gen_kwargs: Dict[str, Any]) -> Dataset:
        # ...
        pass
```

* **Purpose**: To take an input `Dataset`, process each sample through the LLM, parse the results, and return an augmented `Dataset`. This is the primary method called by the `Flow` runner.
* **Parameters**:
    * `samples` (Dataset): The input Hugging Face Dataset.
    * `**gen_kwargs` (Dict[str, Any]): Keyword arguments for LLM generation.
* **Logic**:
    1.  Optionally adds a `num_samples` column to the input `samples` if configured in `self.block_config`.
    2.  **Validation**: Iterates through each `sample` in the input `Dataset` and validates it against `self.prompt_template` using `self._validate()`. Invalid samples are logged and discarded. If no valid samples remain, an empty `Dataset` is returned.
    3.  **Generation**: Calls `self._generate()` with the valid samples and `gen_kwargs` to get a list of raw LLM outputs.
    4.  **Data Alignment**: If `n` (number of parallel samples per input, from `gen_kwargs`) is greater than 1, the original input samples are duplicated `n` times in `extended_samples` to match the number of generated outputs.
    5.  **Parsing and Structuring**:
        * Iterates through the `extended_samples` and the corresponding raw `outputs`.
        * For each `output`, calls `self._parse()` to get `parsed_outputs` (a dictionary where keys are output columns and values are lists of strings).
        * Calculates `max_length` as the maximum number of items found across all lists in `parsed_outputs.values()`. This handles cases where a single LLM output, after parsing, might yield multiple entries for one or more output columns (e.g., extracting multiple bullet points into a list for one output column).
        * It then `zip`s these lists (padded to `max_length`) to create new rows. Each new row combines the original sample data with one set of corresponding parsed values for the output columns. This means a single input sample could result in multiple output rows if parsing extracts multiple items.
    6.  **Output**: Returns a new `Dataset` created from `new_data`.

---

## `ConditionalLLMBlock` Class

This block extends `LLMBlock` to allow for the selection of different prompt configurations based on the value of a specific column in the input data.

```python
@BlockRegistry.register("ConditionalLLMBlock")
class ConditionalLLMBlock(LLMBlock):
    # ... (implementation details below)
    pass
```

* **Inheritance**: Inherits from `LLMBlock`.
* **Registration**: Registered with `BlockRegistry` under the name "ConditionalLLMBlock".
* **Purpose**: To provide flexibility in prompt engineering by dynamically choosing a prompt structure based on input data features.

### `__init__(...)`

The constructor initializes the `ConditionalLLMBlock`.

* **Parameters**:
    * `block_name` (str): Name of the block instance.
    * `config_paths` (Dict[str, str]): A dictionary where keys are possible values from the `selector_column_name` and values are the paths to the corresponding YAML config files for prompts.
    * `client` (openai.OpenAI): An initialized OpenAI client instance.
    * `model_id` (str): The specific model ID to use.
    * `output_cols` (List[str]): A list of names for the output columns.
    * `selector_column_name` (str): The name of the column in the input data whose value will determine which prompt configuration to use.
    * `model_prompt` (str, optional): Template string for the model prompt. Defaults to `"{prompt}"`.
    * `**batch_kwargs` (Dict[str, Any]): Additional keyword arguments for batch processing.
* **Initialization Steps**:
    1.  Calls `super().__init__()`, passing the `config_path` of the *first* entry in `config_paths` for initial setup of the parent `LLMBlock`.
    2.  Stores `selector_column_name`.
    3.  Initializes `self.prompt_template` as a dictionary.
    4.  If `config_paths` contains an "All" key, it uses the initially loaded configuration (from `super().__init__`) for all conditions.
    5.  Otherwise, it iterates through `config_paths`. For each `config_key` (selector value) and `config` (path to YAML), it loads the configuration using `self._load_config(config)` and creates a `jinja2.Template` object. This template is stored in `self.prompt_template[config_key]`.

### `_format_prompt(...)` (Override)

Overrides the parent method to select and render the correct prompt template.

```python
    def _format_prompt(self, sample: Dict[str, Any]) -> str:
        # ...
        pass
```

* **Purpose**: To dynamically select the prompt template based on the value in the `selector_column_name` of the input `sample` and then format the prompt.
* **Logic**:
    * If `self.prompt_template` is a dictionary (i.e., multiple configs were loaded), it retrieves the specific Jinja2 template using `sample[self.selector_column_name]` as the key.
    * It then renders this selected template with the `sample` data.
    * If `self.prompt_template` is not a dictionary (the "All" case), it directly renders the single template.
* **Returns**: `str` - The formatted prompt string using the conditionally selected template.

### `_validate(...)` (Override)

Overrides the parent method to ensure validation uses the conditionally selected prompt template.

```python
    def _validate(self, prompt_template: Union[str, Template], input_dict: Dict[str, Any]) -> bool:
        # ...
        pass
```
* **Purpose**: To select the correct prompt template based on the `input_dict` before performing validation.
* **Logic**:
    * The `prompt_template` argument here will be `self.prompt_template` from the `LLMBlock.generate` method.
    * If this `prompt_template` is a dictionary, it selects the specific template using `input_dict[self.selector_column_name]`.
    * It then calls `super()._validate()` with the correctly selected template and the `input_dict`.
* **Returns**: `bool` - Result of the parent's validation method.

This module provides powerful and flexible ways to integrate LLM generation capabilities into `sdg_hub` workflows.
```