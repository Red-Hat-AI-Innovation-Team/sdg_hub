# SPDX-License-Identifier: Apache-2.0
"""Prompt formatting blocks for converting templates to chat messages.

This module provides blocks for formatting prompts from templates into
structured chat messages compatible with the Chat Completions API.
"""

# Standard
from typing import Any, Dict, List, Optional, Union
import re

# Third Party
from datasets import Dataset
from jinja2 import Template

# Local
from .block import Block
from ..logger_config import setup_logger
from ..registry import BlockRegistry, PromptRegistry

logger = setup_logger(__name__)


@BlockRegistry.register("PromptFormattingBlock")
class PromptFormattingBlock(Block):
    """Block for formatting prompts from templates into chat messages.

    This block takes input from dataset columns, applies a given prompt template,
    and outputs properly structured chat messages compatible with the Chat Completions API.
    Supports both legacy string output and modern chat messages format.

    Parameters
    ----------
    block_name : str
        Name of the block.
    input_cols : Union[str, List[str]]
        Input column name(s) that map to template variables.
    output_cols : Union[str, List[str]]
        Output column name(s) for the formatted prompts.
    config_path : str
        Path to the configuration file containing template and role mapping.
    output_format : str, optional
        Output format: "messages" (default) for chat messages, "string" for legacy format.
    model_prompt : str, optional
        Template string for model prompt, by default "{prompt}".
    **batch_kwargs : Dict[str, Any]
        Additional keyword arguments for batch processing.
    """

    def __init__(
        self,
        block_name: str,
        input_cols: Union[str, List[str]],
        output_cols: Union[str, List[str]],
        config_path: str,
        output_format: str = "messages",
        model_prompt: str = "{prompt}",
        **batch_kwargs: Dict[str, Any],
    ) -> None:
        super().__init__(block_name)
        
        # Standardize input/output columns to lists
        self.input_cols = [input_cols] if isinstance(input_cols, str) else input_cols
        self.output_cols = [output_cols] if isinstance(output_cols, str) else output_cols
        
        # Validate output format
        if output_format not in ["messages", "string"]:
            raise ValueError("output_format must be either 'messages' or 'string'")
        self.output_format = output_format
        
        # Load configuration
        self.block_config = self._load_config(config_path)
        
        # Setup template structure (default to LLMBlock structure for backward compatibility)
        self.prompt_struct = self.block_config.get(
            "prompt_struct", 
            """{system}\n{introduction}\n{principles}\n{examples}\n{generation}"""
        )
        
        # Setup role mapping for chat messages
        self.role_mapping = self.block_config.get("role_mapping", {
            "system": "system",
            "introduction": "user", 
            "principles": "user",
            "examples": "user",
            "generation": "user"
        })
        
        # Store config for template creation in mapping function
        self.filtered_config = {
            k: (v if v is not None else "") for k, v in self.block_config.items()
        }
        
        self.model_prompt = model_prompt
        self.num_procs = batch_kwargs.get("num_procs", 8)
        
        # Log initialization
        logger.info(
            f"Initialized PromptFormattingBlock '{block_name}' with output format '{output_format}'",
            extra={
                "block_name": block_name,
                "output_format": output_format,
                "input_cols": self.input_cols,
                "output_cols": self.output_cols,
            },
        )

    @staticmethod
    def _format_to_messages_static(
        sample: Dict[str, Any], 
        prompt_struct: str, 
        filtered_config: Dict[str, Any],
        role_mapping: Dict[str, str],
        model_prompt: str
    ) -> List[Dict[str, str]]:
        """Static method to format template to chat messages.

        Parameters
        ----------
        sample : Dict[str, Any]
            Input sample containing template variables.
        prompt_struct : str
            Template structure string.
        filtered_config : Dict[str, Any]
            Filtered configuration for template.
        role_mapping : Dict[str, str]
            Mapping of template variables to roles.
        model_prompt : str
            Model prompt template.

        Returns
        -------
        List[Dict[str, str]]
            List of chat messages with role and content.
        """
        # Group template variables by role and render them with sample data
        role_content = {}
        for var_name, role in role_mapping.items():
            if var_name in filtered_config:
                content_template = filtered_config[var_name]
                if content_template and str(content_template).strip():
                    # Render the content template with sample data
                    try:
                        content_template_obj = Template(str(content_template))
                        rendered_content = content_template_obj.render(sample).strip()
                        if rendered_content:  # Only include non-empty content
                            if role not in role_content:
                                role_content[role] = []
                            role_content[role].append(rendered_content)
                    except Exception as e:
                        # If rendering fails, use the raw content
                        logger.warning(f"Failed to render template for {var_name}: {e}")
                        if str(content_template).strip():
                            if role not in role_content:
                                role_content[role] = []
                            role_content[role].append(str(content_template).strip())
        
        # Create messages from grouped content
        messages = []
        for role, contents in role_content.items():
            if contents:
                # Join multiple contents for the same role
                combined_content = "\n".join(contents)
                messages.append({
                    "role": role,
                    "content": combined_content
                })
        
        # If no messages were created, fall back to legacy string format
        if not messages:
            # Create template from structure and config
            prompt_template = Template(prompt_struct.format(**filtered_config))
            
            # Render the template with sample data
            rendered_template = prompt_template.render(sample).strip()
            
            # Apply model prompt template if it's not the default "{prompt}"
            if model_prompt != "{prompt}":
                try:
                    final_prompt = PromptRegistry.render_template(
                        model_prompt, rendered_template, add_generation_prompt=True
                    ).strip()
                except KeyError:
                    # If template not found, use the rendered template as-is
                    final_prompt = rendered_template
            else:
                # For default "{prompt}", just use the rendered template
                final_prompt = rendered_template
            
            # Create a default user message
            messages.append({
                "role": "user",
                "content": final_prompt
            })
        
        return messages

    @staticmethod
    def _format_to_string_static(
        sample: Dict[str, Any], 
        prompt_struct: str, 
        filtered_config: Dict[str, Any],
        model_prompt: str
    ) -> str:
        """Static method to format template to single string (legacy mode).

        Parameters
        ----------
        sample : Dict[str, Any]
            Input sample containing template variables.
        prompt_struct : str
            Template structure string.
        filtered_config : Dict[str, Any]
            Filtered configuration for template.
        model_prompt : str
            Model prompt template.

        Returns
        -------
        str
            Formatted prompt string.
        """
        # Create template from structure and config
        prompt_template = Template(prompt_struct.format(**filtered_config))
        
        # Use the same logic as LLMBlock for backward compatibility
        prompt_templated_str = prompt_template.render(sample).strip()
        
        # Apply model prompt template if it's not the default "{prompt}"
        if model_prompt != "{prompt}":
            try:
                return PromptRegistry.render_template(
                    model_prompt, prompt_templated_str, add_generation_prompt=True
                ).strip()
            except KeyError:
                # If template not found, use the rendered template as-is
                return prompt_templated_str
        else:
            # For default "{prompt}", just use the rendered template
            return prompt_templated_str

    def _validate_sample(self, sample: Dict[str, Any]) -> bool:
        """Validate that the sample contains required template variables.

        Parameters
        ----------
        sample : Dict[str, Any]
            Input sample to validate.

        Returns
        -------
        bool
            True if sample is valid, False otherwise.
        """
        # Create template for validation
        prompt_template = Template(self.prompt_struct.format(**self.filtered_config))
        return self._validate(prompt_template, sample)

    def generate(self, samples: Dataset) -> Dataset:
        """Generate formatted prompts from the input dataset.

        Parameters
        ----------
        samples : Dataset
            Input dataset containing template variables.

        Returns
        -------
        Dataset
            Dataset with formatted prompts added to output columns.
        """
        logger.debug(f"Formatting prompts for {len(samples)} samples")
        
        if len(samples) == 0:
            logger.warning("No samples to format, returning empty dataset")
            return Dataset.from_list([])
        
        # Validate samples and remove invalid ones
        valid_samples = []
        for sample in samples:
            if self._validate_sample(sample):
                valid_samples.append(sample)
            else:
                logger.warning(f"Sample failed validation: {sample}")
        
        if len(valid_samples) == 0:
            logger.warning("No valid samples to format, returning empty dataset")
            return Dataset.from_list([])
        
        # Process samples using static methods to avoid multiprocessing issues
        def format_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
            """Format a single sample."""
            if self.output_format == "messages":
                formatted_output = self._format_to_messages_static(
                    sample, 
                    self.prompt_struct, 
                    self.filtered_config,
                    self.role_mapping,
                    self.model_prompt
                )
            else:  # string format
                formatted_output = self._format_to_string_static(
                    sample,
                    self.prompt_struct,
                    self.filtered_config,
                    self.model_prompt
                )
            
            # Add to output columns
            result = {**sample}
            for output_col in self.output_cols:
                result[output_col] = formatted_output
            
            return result
        
        # Apply formatting to all samples
        formatted_samples = samples.map(format_sample, num_proc=self.num_procs)
        
        logger.info(
            f"Successfully formatted {len(formatted_samples)} samples",
            extra={
                "block_name": self.block_name,
                "output_format": self.output_format,
                "sample_count": len(formatted_samples),
            },
        )
        
        return formatted_samples 