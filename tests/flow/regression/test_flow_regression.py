# SPDX-License-Identifier: Apache-2.0
"""Parametrized regression tests for all shipped flow YAMLs.

Each flow is loaded via ``Flow.from_yaml()``, fed an auto-generated seed
dataset, and run end-to-end with mocked LLM responses.  The assertions
verify structural invariants (no crash, rows produced, columns added) --
not output correctness.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from sdg_hub.core.flow.base import Flow

from .conftest import (
    MockResponseBuilder,
    build_seed_dataset,
    discover_flow_yamls,
    flow_id,
)

_DISCOVERED_FLOWS = discover_flow_yamls()
assert _DISCOVERED_FLOWS, (
    "No flow YAMLs discovered; check FLOWS_DIR resolution and SKIP_BLOCK_TYPES."
)


@pytest.mark.parametrize(
    "flow_yaml",
    _DISCOVERED_FLOWS,
    ids=lambda p: flow_id(p),
)
def test_flow_runs_without_error(
    flow_yaml: Path,
    mock_litellm: Callable[[dict[str, str]], None],
) -> None:
    """Regression: flow loads, generates, and produces non-empty output."""
    response_map = MockResponseBuilder(flow_yaml).build()
    mock_litellm(response_map)

    flow = Flow.from_yaml(str(flow_yaml))
    dataset = build_seed_dataset(flow)
    flow.set_model_config(model="mock/model", api_key="mock-key")

    flow.dry_run(dataset)
    result = flow.generate(dataset)

    assert len(result) > 0, f"Flow produced empty output: {flow_yaml.name}"
    new_cols = set(result.columns) - set(dataset.columns)
    assert new_cols, f"Flow did not add any columns: {flow_yaml.name}"

    if flow.metadata and flow.metadata.output_columns:
        expected = set(flow.metadata.output_columns)
        assert expected.issubset(set(result.columns)), (
            f"Missing declared output_columns: {expected - set(result.columns)}"
        )
