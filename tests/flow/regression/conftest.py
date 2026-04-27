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
    if not isinstance(data, dict):
        raise ValueError(f"Malformed flow YAML (expected mapping): {yaml_path}")
    try:
        block_types = {b["block_type"] for b in data.get("blocks", [])}
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Malformed block entry in {yaml_path}: {exc}") from exc
    return bool(block_types & SKIP_BLOCK_TYPES)


# ---------------------------------------------------------------------------
# MockResponseBuilder
# ---------------------------------------------------------------------------


class MockResponseBuilder:
    """Reads a flow YAML and builds mock LLM responses satisfying downstream parsers.

    Walks the block list to find each LLMChatBlock and its downstream parser
    and optional filter, skipping intermediate blocks like
    LLMResponseExtractorBlock that don't affect content shape.
    """

    def __init__(self, yaml_path: Path) -> None:
        with open(yaml_path) as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            raise ValueError(f"Malformed flow YAML (expected mapping): {yaml_path}")
        self.blocks: list[dict[str, Any]] = data.get("blocks", [])

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
        """Scan forward from *llm_idx* until a parser or chain-breaker is found."""
        for j in range(llm_idx + 1, len(self.blocks)):
            btype = self.blocks[j]["block_type"]
            if btype in _PARSER_TYPES:
                return self.blocks[j], j
            if btype in _CHAIN_BREAKERS:
                break
        return None

    def _find_downstream_filter(
        self, llm_idx: int, parser_idx: int | None
    ) -> dict[str, Any] | None:
        """Scan forward from *parser_idx* (or *llm_idx*) for a filter block."""
        start = (parser_idx + 1) if parser_idx is not None else (llm_idx + 1)
        for j in range(start, len(self.blocks)):
            btype = self.blocks[j]["block_type"]
            if btype == "ColumnValueFilterBlock":
                return self.blocks[j]
            if btype in _CHAIN_BREAKERS:
                break
        return None

    # -- content generators ---------------------------------------------------

    @staticmethod
    def _as_list(value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        return list(value)

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
            return self._json_content(cfg, filt)
        if btype == "RegexParserBlock":
            return self._regex_content(cfg, filt)
        raise ValueError(f"Unrecognized parser type: {btype!r}")

    def _tag_content(self, cfg: dict[str, Any], filt: dict[str, Any] | None) -> str:
        start_tags = self._as_list(cfg.get("start_tags"))
        end_tags = self._as_list(cfg.get("end_tags"))
        output_cols = self._as_list(cfg.get("output_cols"))

        if not start_tags:
            return "mock text content"

        if len(start_tags) != len(end_tags):
            raise ValueError(
                f"TagParserBlock has mismatched start_tags ({len(start_tags)}) "
                f"and end_tags ({len(end_tags)})"
            )

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
        if filt and self._filter_col(filt) == col:
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
    def _filter_col(filt: dict[str, Any]) -> str:
        """Return the column name a ColumnValueFilterBlock reads from."""
        fcfg = filt.get("block_config", {})
        finput = fcfg.get("input_cols", [])
        if isinstance(finput, list) and finput:
            return str(finput[0])
        if isinstance(finput, dict) and finput:
            return str(next(iter(finput)))
        return ""

    def _json_content(self, cfg: dict[str, Any], filt: dict[str, Any] | None) -> str:
        output_cols: list[str] = self._as_list(cfg.get("output_cols"))
        if not output_cols:
            data: dict[str, Any] = {"result": "mock value", "score": 5}
        else:
            data = {col: f"mock {col}" for col in output_cols}
        if filt:
            fcol = self._filter_col(filt)
            if fcol:
                data[fcol] = self._filter_passing_value(filt)
        return json.dumps(data)

    def _regex_content(self, cfg: dict[str, Any], filt: dict[str, Any] | None) -> str:
        pattern: str = cfg.get("parsing_pattern", "")
        if r"\d+" in pattern and r"\." in pattern:
            return "1. mock fact one\n2. mock fact two"
        if "Question" in pattern or "QUESTION" in pattern:
            return "[Question] mock question [Answer] mock answer"
        raise ValueError(
            f"Unrecognized regex pattern in RegexParserBlock, "
            f"add a handler in MockResponseBuilder._regex_content: {pattern!r}"
        )


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
        min_samples = getattr(req, "min_samples", 1) or 1
        num_rows = max(num_rows, min_samples)

    if not cols:
        cols = ["text"]

    data: dict[str, Any] = {}
    for col in cols:
        if "pool" in col.lower():
            data[col] = [["item_a", "item_b", "item_c"] for _ in range(num_rows)]
        else:
            data[col] = [f"sample {col} text row {i}" for i in range(num_rows)]

    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# mock_litellm fixture
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_litellm():
    """Patch ``LLMChatBlock._generate_sync`` and ``_run_async_generation``
    so no real LLM calls are made.

    Yields a callable ``set_responses(mapping)`` where *mapping* is
    ``{block_name: content_text}``.  Each LLMChatBlock looks up its own
    ``block_name`` and returns the corresponding content for every row.
    """
    response_map: dict[str, str] = {}

    def set_responses(mapping: dict[str, str]) -> None:
        response_map.update(mapping)

    def _content_for(block_name: str) -> str:
        if block_name not in response_map:
            raise AssertionError(
                f"No mock response configured for LLMChatBlock '{block_name}'"
            )
        return response_map[block_name]

    def _patched_sync(
        self: LLMChatBlock,
        messages_list: list[list[dict[str, Any]]],
        completion_kwargs: dict[str, Any],
    ) -> list[list[dict[str, Any]]]:
        content = _content_for(self.block_name)
        return [[{"content": content}] for _ in messages_list]

    def _patched_async(
        self: LLMChatBlock,
        messages_list: list[list[dict[str, Any]]],
        completion_kwargs: dict[str, Any],
        flow_max_concurrency: int | None,
    ) -> list[list[dict[str, Any]]]:
        content = _content_for(self.block_name)
        return [[{"content": content}] for _ in messages_list]

    with (
        patch.object(LLMChatBlock, "_generate_sync", _patched_sync),
        patch.object(LLMChatBlock, "_run_async_generation", _patched_async),
    ):
        yield set_responses
