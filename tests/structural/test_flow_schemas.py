"""Verify all built-in flow YAMLs have valid metadata.

See docs/agent-knowledge/flow-invariants.md for the full specification of
required metadata fields.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

FLOWS_DIR = Path(__file__).resolve().parents[2] / "src" / "sdg_hub" / "flows"

REQUIRED_METADATA_FIELDS = ("name", "author", "description", "tags")


def _discover_flow_yamls() -> list[Path]:
    """Return all .yaml/.yml files under src/sdg_hub/flows/ that are flow definitions."""
    candidates = sorted(FLOWS_DIR.rglob("*.yaml")) + sorted(FLOWS_DIR.rglob("*.yml"))
    flows: list[Path] = []
    for path in candidates:
        with open(path) as fh:
            try:
                data = yaml.safe_load(fh)
            except yaml.YAMLError:
                continue
        if isinstance(data, dict) and "metadata" in data and "blocks" in data:
            flows.append(path)
    return flows


def _flow_id(path: Path) -> str:
    """Return a human-readable parametrize ID based on relative path."""
    return str(path.relative_to(FLOWS_DIR))


FLOW_YAMLS = _discover_flow_yamls()


@pytest.mark.parametrize("flow_path", FLOW_YAMLS, ids=[_flow_id(p) for p in FLOW_YAMLS])
def test_flow_has_required_metadata(flow_path: Path) -> None:
    """Each flow YAML must include metadata with name, author, description, and tags."""
    with open(flow_path) as fh:
        data = yaml.safe_load(fh)

    metadata = data["metadata"]

    for field in REQUIRED_METADATA_FIELDS:
        assert field in metadata, (
            f"Flow {flow_path.relative_to(FLOWS_DIR)} is missing metadata.{field}. "
            f"See docs/agent-knowledge/flow-invariants.md for required fields."
        )

    tags = metadata["tags"]
    assert isinstance(tags, list) and len(tags) > 0, (
        f"Flow {flow_path.relative_to(FLOWS_DIR)} metadata.tags must be a non-empty list. "
        f"See docs/agent-knowledge/flow-invariants.md for required fields."
    )
