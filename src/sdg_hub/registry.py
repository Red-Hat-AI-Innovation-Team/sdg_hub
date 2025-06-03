# SPDX-License-Identifier: Apache-2.0
"""
Registry module for managing blocks and prompt templates.

This module provides two registry classes:
1. BlockRegistry: Manages registration and retrieval of block classes
2. PromptRegistry: Handles Jinja2 template registration and rendering for prompts

These registries provide a centralized way to manage and access different types
of blocks and prompt templates used in the SDG pipeline.
"""

# Standard
from typing import Union, List, Dict

# Third Party
from jinja2 import Template

# Local
from .logger_config import setup_logger

logger = setup_logger(__name__)


class BlockRegistry:
    """
    Registry for block classes to avoid manual additions to block type map.
    
    This class provides a centralized registry for block classes, allowing them
    to be registered and retrieved by name. It uses a class-level dictionary to
    store the mapping between block names and their corresponding classes.
    
    Attributes:
        _registry (Dict[str, type]): Class-level dictionary storing registered
            block names and their corresponding classes.
    """

    _registry: Dict[str, type] = {}

    @classmethod
    def register(cls, block_name: str):
        """
        Decorator to register a block class under a specified name.
        
        This decorator adds a block class to the registry with the specified name.
        It can be used to register block classes without modifying the registry
        directly.
        
        Args:
            block_name (str): Name under which to register the block.
                This name will be used to retrieve the block class later.
        
        Returns:
            callable: A decorator function that registers the block class.
        
        Example:
            @BlockRegistry.register("my_block")
            class MyBlock:
                pass
        """

        def decorator(block_class):
            cls._registry[block_name] = block_class
            logger.debug(
                f"Registered block '{block_name}' with class '{block_class.__name__}'"
            )
            return block_class

        return decorator

    @classmethod
    def get_registry(cls):
        """
        Retrieve the current registry map of block types.
        
        Returns:
            Dict[str, type]: Dictionary mapping block names to their corresponding
                classes.
        """
        logger.debug("Fetching the block registry map.")
        return cls._registry


class PromptRegistry:
    """
    Registry for managing Jinja2 prompt templates.
    
    This class provides functionality for registering, retrieving, and rendering
    Jinja2 templates used for generating prompts. It supports both single query
    strings and structured message lists.
    
    Attributes:
        _registry (Dict[str, Template]): Class-level dictionary storing registered
            template names and their corresponding Jinja2 Template instances.
    """

    _registry: Dict[str, Template] = {}

    @classmethod
    def register(cls, name: str):
        """
        Decorator to register a Jinja2 template function by name.
        
        This decorator takes a function that returns a template string and
        registers it in the registry. The function should return a string
        containing the Jinja2 template.
        
        Args:
            name (str): Name of the template to register.
                This name will be used to retrieve the template later.
        
        Returns:
            callable: A decorator function that registers the template.
        
        Example:
            @PromptRegistry.register("my_template")
            def my_template():
                return "Hello {{ name }}!"
        """

        def decorator(func):
            template_str = func()
            cls._registry[name] = Template(template_str)
            logger.debug(f"Registered prompt template '{name}'")
            return func

        return decorator

    @classmethod
    def get_template(cls, name: str) -> Template:
        """
        Retrieve a Jinja2 template by name.
        
        Args:
            name (str): Name of the template to retrieve.
        
        Returns:
            Template: The Jinja2 template instance.
        
        Raises:
            KeyError: If the template name is not found in the registry.
        """
        if name not in cls._registry:
            raise KeyError(f"Template '{name}' not found.")
        logger.debug(f"Retrieving prompt template '{name}'")
        return cls._registry[name]

    @classmethod
    def get_registry(cls):
        """
        Retrieve the current registry map of templates.
        
        Returns:
            Dict[str, Template]: Dictionary mapping template names to their
                corresponding Jinja2 Template instances.
        """
        logger.debug("Fetching the block registry map.")
        return cls._registry

    @classmethod
    def render_template(
        cls,
        name: str,
        messages: Union[str, List[Dict[str, str]]],
        add_generation_prompt: bool = True,
    ) -> str:
        """
        Render the template with the provided messages or query.
        
        This method handles both single query strings and structured message lists.
        For the "blank" template, it returns the query string as-is without any
        templating. For other templates, it renders the template with the provided
        messages.
        
        Args:
            name (str): Name of the template to render.
            messages (Union[str, List[Dict[str, str]]]): Either a single query
                string or a list of messages (each as a dict with 'role' and
                'content').
            add_generation_prompt (bool, optional): Whether to add a generation
                prompt at the end. Defaults to True.
        
        Returns:
            str: The rendered prompt as a string.
        
        Raises:
            ValueError: If the "blank" template is used with a list of messages
                instead of a single query string.
            KeyError: If the template name is not found in the registry.
        """
        # Special handling for "blank" template
        if name == "blank":
            if not isinstance(messages, str):
                raise ValueError(
                    "The 'blank' template can only be used with a single query string, not a list of messages."
                )
            return messages  # Return the query as-is without templating

        # Get the template
        template = cls.get_template(name)

        # If `messages` is a string, wrap it in a list with a default user role
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]

        # Render the template with the `messages` list
        return template.render(
            messages=messages, add_generation_prompt=add_generation_prompt
        )
