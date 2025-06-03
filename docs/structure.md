# Documentation: sdg_hub Repository Structure

This document outlines the directory and file structure of the `sdg_hub` project. Understanding this structure is key to navigating the codebase, finding relevant examples, and comprehending how different components interact.

## Root Directory Layout

The main directories at the root of the project are:

* `.`: The root of the repository.
* `assets/`: Contains static assets used by the project, primarily images for documentation.
* `CONTRIBUTING.md`: Provides guidelines for developers who wish to contribute to the `sdg_hub` project.
* `examples/`: A crucial directory containing various examples demonstrating how to use `sdg_hub` for different synthetic data generation tasks.
* `LICENSE`: Contains the software license information for the project.
* `Makefile`: Defines a set of tasks to be executed, typically for building, testing, or deploying the project.
* `MANIFEST.in`: A file that specifies which files to include in a source distribution of the Python package.
* `pyproject.toml`: A standard Python project configuration file (PEP 518) that defines project metadata, build system requirements, and tool configurations (like linters or formatters).
* `README.md`: The main introductory document for the project, providing an overview, installation instructions, and basic usage (previously processed).
* `scripts/`: Contains utility scripts for development, automation, or operational tasks.
* `src/`: The primary directory containing the source code for the `sdg_hub` library.
* `tests/`: Contains all the tests (unit, integration, etc.) for the `sdg_hub` codebase.
* `tox.ini`: Configuration file for `tox`, an automation tool for Python testing across different environments.
* `web_interface/`: Contains the files for a web-based user interface for interacting with `sdg_hub`.

## Detailed Directory Breakdown

### 1. `assets/`

This directory stores static files, primarily images used in documentation (like the `README.md`).

* `imgs/`: Contains image files.
    * `fig-workflow.png`, `overview.png`: Diagrams illustrating the `sdg_hub` workflow or architecture.
    * `IL_skills_pipeline.png`, `instructlab-banner.png`: Images likely related to "InstructLab" examples or integrations.

### 2. `examples/`

This directory is vital for users and developers to understand the practical application of `sdg_hub`. It's organized by use case or specific demonstrations.

* `data-generation-with-llama-70b/`:
    * `data-generation-with-llama-70b.ipynb`: A Jupyter notebook demonstrating data generation using a Llama 70B model.
    * `synth_knowledge1.5_llama3.3.yaml`: A YAML flow configuration for this specific example.
* `instructlab/`: Examples related to the "InstructLab" methodology, focusing on generating data for teaching LLMs new skills or knowledge.
    * `annotation/`: Likely related to data annotation tasks within InstructLab.
        * `sample_data/emotion_classification.jsonl`: Sample data for an emotion classification task.
    * `skills/`: Focuses on skill-based synthetic data generation.
        * `configs/`: YAML prompt configurations for various skill-related generation and evaluation tasks (e.g., `freeform_questions.yaml`, `evaluate_grounded_pair.yaml`).
        * `flows/`: YAML flow configurations for skill generation (e.g., `synth_grounded_skills.yaml`).
        * `mdtable_manipulation.ipynb`, `unstructured_to_mdtable.ipynb`: Jupyter notebooks for specific skill demonstrations (Markdown table manipulation).
        * `README.md`: A specific README for the `instructlab/skills` examples.
        * `sample_data/`: Seed data for the skill examples.
* `knowledge_tuning/`: Examples focused on generating data to imbue LLMs with specific knowledge.
    * `instructlab/knowledge/`: InstructLab specific knowledge tuning.
        * `document_collection/ibm-annual-report/`: Example documents (PDF, MD, JSON) and a `qna.yaml` prompt configuration for Q&A generation based on an IBM annual report.
        * `document_pre_processing.ipynb`: Notebook for pre-processing documents.
        * `knowledge_generation_and_mixing.ipynb`: Notebook demonstrating the generation and mixing of knowledge data.
    * `knowledge_tuning_with_reasoning_model/`: Advanced knowledge tuning involving a reasoning model.
        * `assets/`: Images specific to this example.
        * `blocks/blocks.py`: Custom Python blocks for this specific example workflow.
        * `flows/`: Complex YAML flow configurations for knowledge generation with reasoning (e.g., `synth_knowledge_reasoning_nemotron_super_49b_rewrite_with_diversity.yaml`).
        * `generate.py`: A Python script likely used to run the generation defined in the flows.
        * `prompts/`: YAML prompt configurations tailored for reasoning-based generation (e.g., `generate_answers_cot.yaml`, `generate_doc_rewrite_inst.yaml`).
        * `README.md`: A specific README for this advanced example.
        * `reasoning_sdg_financebench.ipynb`, `reasoning_sdg_quality.ipynb`: Notebooks for analysis or specific tasks related to this example.
        * `utils.py`: Utility functions for this example.
    * `knowledge_utils.py`: Utility functions shared across knowledge tuning examples.

### 3. `scripts/`

Contains various helper and utility scripts.

* `docparser.py`, `docparser_v2.py`: Scripts likely used for parsing documents as part of a data preparation pipeline.
* `flow_runner.py`: A script to execute `sdg_hub` flows defined in YAML files. This is a key utility for running the framework.
* `__init__.py`: Makes the `scripts` directory a Python package if needed.
* `ruff.sh`: A shell script, possibly for running the Ruff linter/formatter.

### 4. `src/sdg_hub/`

This is the core of the `sdg_hub` library, containing all its Python source code.

* `blocks/`: Contains the Python implementations of the different "Blocks" that form the computational units of the framework.
    * `block.py`: Defines the base `Block` abstract class from which all other blocks inherit.
    * `__init__.py`: Makes the `blocks` directory a Python package and likely imports key block classes for easier access.
    * `llmblock.py`: Implements various `LLMBlock` types (e.g., `LLMBlock`, `ConditionalLLMBlock`, `LLMLogProbBlock`, `LLMMessagesBlock`).
    * `utilblocks.py`: Implements utility blocks (e.g., `FilterByValueBlock`, `CombineColumnsBlock`, `RenameColumns`).
* `checkpointer.py`: Implements the data checkpointing functionality, allowing flows to be resumed.
* `configs/`: Contains default or built-in YAML prompt configurations, organized by category. These serve as templates or starting points for users.
    * `annotations/`: Prompts related to data annotation tasks (e.g., `cot_reflection.yaml`, `simple.yaml`).
    * `knowledge/`: Prompts for knowledge-based generation (e.g., `atomic_facts.yaml`, `generate_questions_responses.yaml`, `evaluate_faithfulness.yaml`).
    * `reasoning/`: Prompts related to reasoning tasks (e.g., `dynamic_cot.yaml`).
    * `skills/`: Prompts for skill-based generation and evaluation (e.g., `analyzer.yaml`, `freeform_questions.yaml`, `icl_examples/`).
        * `icl_examples/`: In-context learning examples for various skill categories (coding, math, reasoning, etc.).
* `flow.py`: Contains the logic for parsing, validating, and executing a "Flow" defined in a YAML file. This class orchestrates the sequence of block executions.
* `flows/`: Contains default or built-in YAML flow configurations, organized by category. These showcase how to combine blocks for common tasks.
    * `annotation/emotion/`: Example flows for emotion annotation.
    * `generation/knowledge/`: Example flows for knowledge generation (e.g., `simple_knowledge.yaml`, `synth_knowledge.yaml`).
    * `generation/skills/`: Example flows for skill generation (e.g., `simple_freeform_skill.yaml`, `synth_skills.yaml`).
* `__init__.py`: Makes `sdg_hub` a Python package and often exports key classes and functions for public use.
* `logger_config.py`: Configures the logging behavior for the `sdg_hub` library.
* `pipeline.py`: May contain higher-level abstractions or entry points for running complex data generation pipelines, potentially wrapping `Flow` execution.
* `prompts.py`: Contains utility functions for loading, parsing, and formatting prompts from YAML configuration files.
* `py.typed`: A marker file (PEP 561) indicating that the package provides type information for static type checkers.
* `registry.py`: Implements a registry system for discovering and managing available `Block` types, allowing for dynamic block loading based on `block_type` in YAML.
* `sdg.py`: Could be a central script or module that ties together various components of the `sdg_hub` framework, possibly providing a main API or command-line interface.
* `utils/`: Contains utility modules and functions used across the `sdg_hub` library.
    * `datautils.py`: Utilities specifically for data manipulation, likely interacting with Hugging Face Datasets.

### 5. `tests/`

This directory houses all the automated tests for the `sdg_hub` project, ensuring code quality and correctness.

* `blocks/`: Tests for individual `Block` implementations.
    * `testdata/test_config.yaml`: Sample configuration data used in block tests.
    * `test_filterblock.py`, `test_llmblock.py`: Test scripts for specific blocks.
* `flows/`: Tests for the `Flow` execution logic and YAML parsing.
    * `testdata/`: Sample YAML configurations (`test_config_1.yaml`) and flow files (`test_flow_1.yaml`) for testing flow execution.
    * `test_flow.py`: Test script for the `flow.py` module.
* `__init__.py`: Makes the `tests` directory a Python package.
* `test_checkpointer.py`: Test script for the `checkpointer.py` module.

### 6. `web_interface/`

Contains the components for a web-based user interface, likely built using a framework like Flask or Streamlit.

* `app.py`: The main application file for the web interface (e.g., Flask app).
* `README.md`: Instructions specific to setting up and running the web interface.
* `requirements.txt`: Python dependencies required for the web interface.
* `static/`: Directory for static web assets.
    * `css/style.css`: Stylesheets for the web interface.
    * `js/app.js`: JavaScript for client-side interactivity.
* `templates/index.html`: HTML template(s) for the web pages.
* `test_block_types.py`: Potentially a script to test or display available block types in the UI.

This structure provides a modular and organized way to manage the `sdg_hub` toolkit, separating core logic (`src/sdg_hub`), examples (`examples/`), tests (`tests/`), and other components.