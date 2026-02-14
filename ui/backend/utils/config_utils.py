# SPDX-License-Identifier: Apache-2.0
"""Configuration persistence and log analysis utilities."""

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict

from config import SAVED_CONFIG_FILE

logger = logging.getLogger(__name__)


def parse_llm_statistics_from_logs(raw_logs: str) -> Dict[str, Any]:
    """Parse raw generation logs to extract LLM call statistics.

    Extracts:
    - Total LLM requests
    - Per-block LLM request counts
    - Block information
    """
    # Strip ANSI codes
    clean_logs = re.sub(r'\x1b\[[0-9;]*m', '', raw_logs)
    clean_logs = re.sub(r'\[([0-9;]*)m', '', clean_logs)

    stats = {
        "total_llm_requests": 0,
        "llm_blocks": [],
        "blocks": {},
        "block_order": [],
        "total_blocks": 0,
    }

    lines = clean_logs.split('\n')
    current_block = None
    current_block_type = None

    for line in lines:
        block_match = re.search(
            r'Executing block (\d+)/(\d+):\s*([\w_]+)\s*\(?([\w]+)?\)?', line
        )
        if block_match:
            block_num, total_blocks, block_name, block_type = block_match.groups()
            current_block = block_name
            current_block_type = block_type or "Unknown"
            stats["total_blocks"] = int(total_blocks)

            if block_name not in stats["blocks"]:
                stats["blocks"][block_name] = {
                    "name": block_name,
                    "type": current_block_type,
                    "number": int(block_num),
                    "llm_requests": 0,
                    "is_llm_block": "LLM" in (block_type or ""),
                }
                stats["block_order"].append(block_name)

        llm_match = re.search(
            r'llm_chat_block.*Generation completed successfully for (\d+) samples',
            line,
            re.IGNORECASE,
        )
        if llm_match:
            samples = int(llm_match.group(1))
            stats["total_llm_requests"] += samples

            if current_block and current_block in stats["blocks"]:
                stats["blocks"][current_block]["llm_requests"] += samples
                stats["blocks"][current_block]["is_llm_block"] = True

        elif "Generation completed successfully for" in line and "llm_chat_block" not in line:
            generic_match = re.search(
                r'Generation completed successfully for (\d+) samples', line
            )
            if generic_match and current_block and "LLM" in current_block_type:
                samples = int(generic_match.group(1))
                stats["total_llm_requests"] += samples
                if current_block in stats["blocks"]:
                    stats["blocks"][current_block]["llm_requests"] += samples

    stats["llm_blocks"] = [
        {"name": b["name"], "requests": b["llm_requests"]}
        for b in stats["blocks"].values()
        if b.get("is_llm_block") and b["llm_requests"] > 0
    ]

    stats["blocks_list"] = [
        stats["blocks"][name]
        for name in stats["block_order"]
        if name in stats["blocks"]
    ]

    return stats


def load_saved_configurations_from_disk(
    saved_configurations: Dict,
    config_model_class=None,
):
    """Load saved configurations from JSON file.

    Args:
        saved_configurations: The mutable dict to populate.
        config_model_class: The SavedConfiguration Pydantic model class.
    """
    if not SAVED_CONFIG_FILE.exists():
        logger.info("No saved configurations file found; starting fresh.")
        return
    try:
        with open(SAVED_CONFIG_FILE, "r") as f:
            data = json.load(f)
        saved_configurations.clear()
        for item in data:
            try:
                if config_model_class:
                    config = config_model_class(**item)
                    saved_configurations[config.id] = config
                else:
                    saved_configurations[item.get("id", "")] = item
            except Exception as exc:
                logger.warning(f"Skipping invalid saved configuration entry: {exc}")
        logger.info(
            f"Loaded {len(saved_configurations)} saved configurations from disk."
        )
    except Exception as exc:
        logger.error(f"Failed to load saved configurations: {exc}")


def persist_saved_configurations(saved_configurations: Dict):
    """Persist saved configurations to disk."""
    try:
        SAVED_CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = SAVED_CONFIG_FILE.with_suffix(".tmp")
        data = [
            config.dict() if hasattr(config, "dict") else config
            for config in saved_configurations.values()
        ]
        with open(tmp_path, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_path, SAVED_CONFIG_FILE)
        try:
            SAVED_CONFIG_FILE.chmod(0o600)
        except Exception:
            pass
        logger.info(f"Persisted {len(saved_configurations)} configurations to disk.")
    except Exception as exc:
        logger.error(f"Failed to persist configurations: {exc}")
