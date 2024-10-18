# Standard
from typing import Union, List, Dict

# Third Party
from jinja2 import Template

# Local
from .logger_config import setup_logger

logger = setup_logger(__name__)


class BlockRegistry:
    """Registry for block classes to avoid manual additions to block type map."""

    _registry: Dict[str, type] = {}

    @classmethod
    def register(cls, block_name: str):
        """
        Decorator to register a block class under a specified name.

        :param block_name: Name under which to register the block.
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

        :return: Dictionary of registered block names and classes.
        """
        logger.debug("Fetching the block registry map.")
        return cls._registry


class PromptRegistry:
    """Registry for managing Jinja2 prompt templates."""

    _registry: Dict[str, Template] = {}

    @classmethod
    def register(cls, name: str):
        """Decorator to register a Jinja2 template function by name.

        :param name: Name of the template to register.
        :return: A decorator that registers the Jinja2 template function.
        """

        def decorator(func):
            template_str = func()
            cls._registry[name] = Template(template_str)
            logger.debug(f"Registered prompt template '{name}'")
            return func

        return decorator

    @classmethod
    def get_template(cls, name: str) -> Template:
        """Retrieve a Jinja2 template by name.

        :param name: Name of the template to retrieve.
        :return: The Jinja2 template instance.
        """
        if name not in cls._registry:
            raise KeyError(f"Template '{name}' not found.")
        logger.debug(f"Retrieving prompt template '{name}'")
        return cls._registry[name]

    @classmethod
    def get_registry(cls):
        """
        Retrieve the current registry map of block types.

        :return: Dictionary of registered block names and classes.
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
        """Render the template with the provided messages or query.

        :param name: Name of the template to render.
        :param messages: Either a single query string or a list of messages (each as a dict with 'role' and 'content').
        :param add_generation_prompt: Whether to add a generation prompt at the end.
        :return: The rendered prompt as a string.
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
