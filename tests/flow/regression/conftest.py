# SPDX-License-Identifier: Apache-2.0
"""Infrastructure for flow regression tests.

Provides auto-discovery of flow YAMLs, mock LLM response generation
that satisfies downstream parsers, and seed dataset generation from
flow metadata.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch
import json

import pandas as pd
import pytest
import yaml

from sdg_hub.core.blocks.llm.llm_chat_block import LLMChatBlock
from sdg_hub.core.flow.base import Flow
import sdg_hub

FLOWS_DIR = Path(sdg_hub.__file__).resolve().parent / "flows"

SKIP_BLOCK_TYPES = frozenset(
    {
        "AgentBlock",
        "AgentResponseExtractorBlock",
        "PythonInterpreterBlock",
        "MCPAgentBlock",
    }
)

_PARSER_TYPES = frozenset({"TagParserBlock", "JSONParserBlock", "RegexParserBlock"})

_CHAIN_BREAKERS = frozenset({"LLMChatBlock", "PromptBuilderBlock"})


# ---------------------------------------------------------------------------
# Flow discovery
# ---------------------------------------------------------------------------


def discover_flow_yamls() -> list[Path]:
    """Auto-discover all flow YAML files, excluding unsupported block types."""
    all_yamls = sorted(FLOWS_DIR.rglob("flow.yaml"))
    return [y for y in all_yamls if not _has_unsupported_blocks(y)]


def flow_id(yaml_path: Path) -> str:
    """Generate a readable test ID from a flow YAML path."""
    return (
        str(yaml_path.relative_to(FLOWS_DIR))
        .replace("/flow.yaml", "")
        .replace("/", "::")
    )


def _has_unsupported_blocks(yaml_path: Path) -> bool:
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    block_types = {b["block_type"] for b in data.get("blocks", [])}
    return bool(block_types & SKIP_BLOCK_TYPES)


# ---------------------------------------------------------------------------
# MockResponseBuilder
# ---------------------------------------------------------------------------


class MockResponseBuilder:
    """Reads a flow YAML and builds mock LLM responses satisfying downstream parsers.

    Walks the block list to find each ``LLMChatBlock -> LLMResponseExtractorBlock
    -> [Parser] -> [ColumnValueFilterBlock]`` chain and generates text content
    that the parser will accept and that will pass any downstream filter.
    """

    def __init__(self, yaml_path: Path) -> None:
        with open(yaml_path) as f:
            self.flow_data = yaml.safe_load(f)
        self.blocks: list[dict[str, Any]] = self.flow_data.get("blocks", [])

    def build(self) -> dict[str, str]:
        """Return ``{llm_block_name: mock_content}`` for every LLMChatBlock."""
        response_map: dict[str, str] = {}
        for i, block in enumerate(self.blocks):
            if block["block_type"] != "LLMChatBlock":
                continue
            block_name = block["block_config"]["block_name"]
            parser_result = self._find_downstream_parser(i)
            if parser_result is not None:
                parser, parser_idx = parser_result
            else:
                parser, parser_idx = None, None
            filt = self._find_downstream_filter(i, parser_idx)
            response_map[block_name] = self._generate_content(parser, filt)
        return response_map

    # -- chain walking helpers ------------------------------------------------

    def _find_downstream_parser(
        self, llm_idx: int
    ) -> tuple[dict[str, Any], int] | None:
        for j in range(llm_idx + 1, min(llm_idx + 5, len(self.blocks))):
            btype = self.blocks[j]["block_type"]
            if btype in _PARSER_TYPES:
                return self.blocks[j], j
            if btype == "LLMChatBlock":
                break
        return None

    def _find_downstream_filter(
        self, llm_idx: int, parser_idx: int | None
    ) -> dict[str, Any] | None:
        start = (parser_idx + 1) if parser_idx is not None else (llm_idx + 1)
        for j in range(start, min(start + 4, len(self.blocks))):
            btype = self.blocks[j]["block_type"]
            if btype == "ColumnValueFilterBlock":
                return self.blocks[j]
            if btype in _CHAIN_BREAKERS:
                break
        return None

    # -- content generators ---------------------------------------------------

    def _generate_content(
        self,
        parser: dict[str, Any] | None,
        filt: dict[str, Any] | None,
    ) -> str:
        if parser is None:
            if filt:
                return self._filter_passing_value(filt)
            return "mock response text"

        btype = parser["block_type"]
        cfg = parser.get("block_config", {})

        if btype == "TagParserBlock":
            return self._tag_content(cfg, filt)
        if btype == "JSONParserBlock":
            return self._json_content(cfg)
        if btype == "RegexParserBlock":
            return self._regex_content(cfg)
        return "mock response text"

    def _tag_content(self, cfg: dict[str, Any], filt: dict[str, Any] | None) -> str:
        start_tags: list[str] = cfg.get("start_tags", [])
        end_tags: list[str] = cfg.get("end_tags", [])
        output_cols: list[str] = cfg.get("output_cols", [])

        if not start_tags:
            return "mock text content"

        parts: list[str] = []
        for i, (start, end) in enumerate(zip(start_tags, end_tags)):
            col = output_cols[i] if i < len(output_cols) else f"field_{i}"
            value = self._tag_value(col, filt, start)

            if start == "" and end == "":
                parts.append(value)
            elif start.startswith("###"):
                parts.append(f"{start}\n1. mock fact one\n2. mock fact two")
            else:
                parts.append(f"{start}{value}{end}")

        return "\n".join(parts)

    def _tag_value(
        self, col: str, filt: dict[str, Any] | None, start_tag: str = ""
    ) -> str:
        if filt:
            fcfg = filt.get("block_config", {})
            finput = fcfg.get("input_cols", [])
            fcol = (
                finput[0]
                if isinstance(finput, list) and finput
                else next(iter(finput), "")
                if isinstance(finput, dict)
                else ""
            )
            if fcol == col:
                return self._filter_passing_value(filt)
        return f"mock {col}"

    @staticmethod
    def _filter_passing_value(filt: dict[str, Any]) -> str:
        cfg = filt.get("block_config", {})
        val = cfg.get("filter_value")
        if val is None:
            block_name = cfg.get("block_name", "<unknown>")
            raise ValueError(
                f"ColumnValueFilterBlock '{block_name}' has no filter_value"
            )
        if isinstance(val, list):
            if not val:
                raise ValueError("ColumnValueFilterBlock has empty filter_value list")
            return str(val[0])
        return str(val)

    @staticmethod
    def _json_content(cfg: dict[str, Any]) -> str:
        output_cols: list[str] = cfg.get("output_cols", [])
        if not output_cols:
            return json.dumps({"result": "mock value", "score": 5})
        return json.dumps({col: f"mock {col}" for col in output_cols})

    @staticmethod
    def _regex_content(cfg: dict[str, Any]) -> str:
        pattern: str = cfg.get("parsing_pattern", "")
        if r"\d+" in pattern and r"\." in pattern:
            return "1. mock fact one\n2. mock fact two"
        if "Question" in pattern or "QUESTION" in pattern:
            return "[Question] mock question [Answer] mock answer"
        return "mock regex content"


# ---------------------------------------------------------------------------
# Seed dataset factory
# ---------------------------------------------------------------------------


def build_seed_dataset(flow: Flow, num_rows: int = 2) -> pd.DataFrame:
    """Auto-generate a minimal test dataset from the flow's dataset_requirements."""
    cols: list[str] = []
    if flow.metadata and flow.metadata.dataset_requirements:
        req = flow.metadata.dataset_requirements
        cols.extend(req.required_columns or [])
        cols.extend(getattr(req, "optional_columns", None) or [])

    if not cols:
        cols = ["text"]

    data: dict[str, Any] = {}
    for col in cols:
        if "pool" in col.lower():
            data[col] = [["item_a", "item_b", "item_c"]] * num_rows
        else:
            data[col] = [f"sample {col} text row {i}" for i in range(num_rows)]

    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# mock_litellm fixture
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_litellm():
    """Patch LLMChatBlock internals so no real LLM calls are made.

    Yields a callable ``set_responses(mapping)`` where *mapping* is
    ``{block_name: content_text}``.  Each LLMChatBlock looks up its own
    ``block_name`` and returns the corresponding content for every row.
    """
    response_map: dict[str, str] = {}

    def set_responses(mapping: dict[str, str]) -> None:
        response_map.update(mapping)

    def _patched_sync(
        self: LLMChatBlock,
        messages_list: list[list[dict[str, Any]]],
        completion_kwargs: dict[str, Any],
    ) -> list[list[dict[str, Any]]]:
        content = response_map.get(self.block_name, "mock fallback")
        return [[{"content": content}] for _ in messages_list]

    def _patched_async(
        self: LLMChatBlock,
        messages_list: list[list[dict[str, Any]]],
        completion_kwargs: dict[str, Any],
        flow_max_concurrency: int | None = None,
    ) -> list[list[dict[str, Any]]]:
        content = response_map.get(self.block_name, "mock fallback")
        return [[{"content": content}] for _ in messages_list]

    with (
        patch.object(LLMChatBlock, "_generate_sync", _patched_sync),
        patch.object(LLMChatBlock, "_run_async_generation", _patched_async),
    ):
        yield set_responses
