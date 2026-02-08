# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the SDG Hub KFP component."""

import json
import os
import tempfile

from sdg_hub.kfp.component import sdg
import pandas as pd
import pytest


class MockArtifact:
    """Mock KFP Output artifact with a writable path."""

    def __init__(self, path: str):
        self.path = path


@pytest.fixture
def tmp_dir():
    """Provide a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def output_artifact(tmp_dir):
    """Mock output dataset artifact."""
    return MockArtifact(os.path.join(tmp_dir, "output.jsonl"))


@pytest.fixture
def output_metrics(tmp_dir):
    """Mock output metrics artifact."""
    return MockArtifact(os.path.join(tmp_dir, "metrics.json"))


@pytest.fixture
def sample_input_file(tmp_dir):
    """Create a sample JSONL input file and return its path."""
    path = os.path.join(tmp_dir, "input.jsonl")
    df = pd.DataFrame(
        {
            "document": ["Doc one.", "Doc two.", "Doc three."],
            "domain": ["science", "tech", "math"],
        }
    )
    df.to_json(path, orient="records", lines=True)
    return path


def _call_sdg(output_artifact, output_metrics, **kwargs):
    """Helper to call the component's python_func with defaults."""
    defaults = {
        "input_pvc_path": "",
        "flow_id": "",
        "flow_yaml_path": "",
        "model": "",
        "max_concurrency": 10,
        "checkpoint_pvc_path": "",
        "save_freq": 100,
        "log_level": "INFO",
        "temperature": 0.7,
        "max_tokens": 2048,
    }
    defaults.update(kwargs)
    sdg.python_func(
        output_artifact=output_artifact,
        output_metrics=output_metrics,
        **defaults,
    )


class TestInputHandling:
    """Tests for input file loading."""

    def test_load_jsonl_input(self, output_artifact, output_metrics, sample_input_file):
        """Component reads a JSONL file and passes data through."""
        _call_sdg(
            output_artifact,
            output_metrics,
            input_pvc_path=sample_input_file,
        )

        result = pd.read_json(output_artifact.path, lines=True)
        assert len(result) == 3
        assert list(result.columns) == ["document", "domain"]
        assert result["document"].tolist() == ["Doc one.", "Doc two.", "Doc three."]

    def test_missing_input_file_raises(self, output_artifact, output_metrics):
        """Component raises FileNotFoundError for nonexistent input."""
        with pytest.raises(FileNotFoundError, match="Input file not found"):
            _call_sdg(
                output_artifact,
                output_metrics,
                input_pvc_path="/nonexistent/path/data.jsonl",
            )

    def test_no_input_creates_dummy(self, output_artifact, output_metrics):
        """Component creates dummy data when no input path is provided."""
        _call_sdg(
            output_artifact,
            output_metrics,
            input_pvc_path="",
            flow_id="test-flow",
        )

        result = pd.read_json(output_artifact.path, lines=True)
        assert len(result) == 1
        assert "message" in result.columns
        assert "flow_id" in result.columns
        assert result["flow_id"].iloc[0] == "test-flow"

    def test_no_input_no_flow_id(self, output_artifact, output_metrics):
        """Dummy data uses 'none' when flow_id is also empty."""
        _call_sdg(
            output_artifact,
            output_metrics,
            input_pvc_path="",
            flow_id="",
        )

        result = pd.read_json(output_artifact.path, lines=True)
        assert result["flow_id"].iloc[0] == "none"


class TestOutputHandling:
    """Tests for output artifact and metrics writing."""

    def test_output_matches_input(
        self, output_artifact, output_metrics, sample_input_file
    ):
        """Output artifact contains the same data as input."""
        input_df = pd.read_json(sample_input_file, lines=True)
        _call_sdg(
            output_artifact,
            output_metrics,
            input_pvc_path=sample_input_file,
        )

        output_df = pd.read_json(output_artifact.path, lines=True)
        pd.testing.assert_frame_equal(input_df, output_df)

    def test_metrics_written(self, output_artifact, output_metrics, sample_input_file):
        """Metrics file contains expected keys and values."""
        _call_sdg(
            output_artifact,
            output_metrics,
            input_pvc_path=sample_input_file,
        )

        with open(output_metrics.path) as f:
            metrics = json.load(f)

        assert "metrics" in metrics
        metric_names = {m["name"] for m in metrics["metrics"]}
        assert metric_names == {"input_rows", "output_rows", "execution_time_seconds"}

    def test_metrics_row_counts(
        self, output_artifact, output_metrics, sample_input_file
    ):
        """Metrics report correct row counts."""
        _call_sdg(
            output_artifact,
            output_metrics,
            input_pvc_path=sample_input_file,
        )

        with open(output_metrics.path) as f:
            metrics = json.load(f)

        by_name = {m["name"]: m["numberValue"] for m in metrics["metrics"]}
        assert by_name["input_rows"] == 3
        assert by_name["output_rows"] == 3

    def test_metrics_execution_time(self, output_artifact, output_metrics):
        """Metrics include a non-negative execution time."""
        _call_sdg(output_artifact, output_metrics)

        with open(output_metrics.path) as f:
            metrics = json.load(f)

        by_name = {m["name"]: m["numberValue"] for m in metrics["metrics"]}
        assert by_name["execution_time_seconds"] >= 0


class TestLogging:
    """Tests for logging configuration."""

    def test_debug_log_level(self, output_artifact, output_metrics):
        """Component accepts DEBUG log level without error."""
        _call_sdg(output_artifact, output_metrics, log_level="DEBUG")
        assert os.path.exists(output_artifact.path)

    def test_warning_log_level(self, output_artifact, output_metrics):
        """Component accepts WARNING log level without error."""
        _call_sdg(output_artifact, output_metrics, log_level="WARNING")
        assert os.path.exists(output_artifact.path)


class TestModuleExports:
    """Tests for package-level exports."""

    def test_sdg_importable_from_package(self):
        """sdg component is importable from sdg_hub.kfp."""
        from sdg_hub.kfp import sdg as imported_sdg

        assert imported_sdg is sdg

    def test_all_exports(self):
        """__all__ lists expected exports."""
        import sdg_hub.kfp as kfp_module

        assert hasattr(kfp_module, "__all__")
        assert "sdg" in kfp_module.__all__
