# SPDX-License-Identifier: Apache-2.0
"""YAML utilities for flow configuration."""

# Standard
from pathlib import Path
from typing import Any, Dict

# Third Party
import yaml

# Local
from .logger_config import setup_logger

logger = setup_logger(__name__)


def save_flow_yaml(
    yaml_path: str,
    flow_config: Dict[str, Any],
    reason: str = "",
    sort_keys: bool = False,
    width: int = 240,
    indent: int = 2,
) -> None:
    """
    Save flow configuration to a YAML file.

    This utility function saves flow configurations to YAML files,
    ensuring consistent formatting and logging across the codebase.

    Parameters
    ----------
    yaml_path : str
        Path to the YAML file to write.
    flow_config : Dict[str, Any]
        Flow configuration to save.
    reason : str, optional
        Reason for saving, used in log message.
    width : int, optional
        Maximum line width for YAML output.
    indent : int, optional
        Indentation level for YAML output.
    """
    yaml_path = str(Path(yaml_path))  # Normalize path

    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(
            flow_config,
            f,
            default_flow_style=False,
            sort_keys=sort_keys,
            width=width,
            indent=indent,
        )

    log_msg = f"Saved flow configuration to YAML: {yaml_path}"
    if reason:
        log_msg = f"{log_msg} ({reason})"
    logger.debug(log_msg)
