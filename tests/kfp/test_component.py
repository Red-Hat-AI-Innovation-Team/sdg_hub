# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the SDG Hub KFP component."""

from unittest.mock import MagicMock, patch
import json
import os
import tempfile

from sdg_hub.core.utils.error_handling import FlowValidationError
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


def _make_mock_flow(sample_input_file=None, return_df=None):
    """Create a mock Flow that passes through input data."""
    mock_flow = MagicMock()
    mock_flow.metadata.name = "mock-flow"
    mock_flow.metadata.version = "1.0.0"
    mock_flow.blocks = []
    mock_flow.is_model_config_required.return_value = False
    mock_flow.validate_dataset.return_value = []
    if return_df is not None:
        mock_flow.generate.return_value = return_df
    elif sample_input_file:
        mock_flow.generate.side_effect = lambda df, **kw: df.copy()
    else:
        mock_flow.generate.side_effect = lambda df, **kw: df.copy()
    return mock_flow


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


# =============================================================================
# INPUT HANDLING TESTS
# =============================================================================


class TestInputHandling:
    """Tests for input file loading."""

    @patch("sdg_hub.core.flow.base.Flow.from_yaml")
    @patch("sdg_hub.core.flow.registry.FlowRegistry.get_flow_path_safe")
    def test_load_jsonl_input(
        self,
        mock_get_path,
        mock_from_yaml,
        output_artifact,
        output_metrics,
        sample_input_file,
    ):
        """Component reads a JSONL file and passes data through."""
        mock_get_path.return_value = "/resolved/flow.yaml"
        mock_from_yaml.return_value = _make_mock_flow()

        _call_sdg(
            output_artifact,
            output_metrics,
            input_pvc_path=sample_input_file,
            flow_id="test-flow",
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
                flow_id="test-flow",
            )

    @patch("sdg_hub.core.flow.base.Flow.from_yaml")
    @patch("sdg_hub.core.flow.registry.FlowRegistry.get_flow_path_safe")
    def test_no_input_creates_dummy(
        self, mock_get_path, mock_from_yaml, output_artifact, output_metrics
    ):
        """Component creates dummy data when no input path is provided."""
        mock_get_path.return_value = "/resolved/flow.yaml"
        mock_from_yaml.return_value = _make_mock_flow()

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


# =============================================================================
# OUTPUT HANDLING TESTS
# =============================================================================


class TestOutputHandling:
    """Tests for output artifact and metrics writing."""

    @patch("sdg_hub.core.flow.base.Flow.from_yaml")
    @patch("sdg_hub.core.flow.registry.FlowRegistry.get_flow_path_safe")
    def test_output_matches_input(
        self,
        mock_get_path,
        mock_from_yaml,
        output_artifact,
        output_metrics,
        sample_input_file,
    ):
        """Output artifact contains the same data as input (passthrough)."""
        mock_get_path.return_value = "/resolved/flow.yaml"
        mock_from_yaml.return_value = _make_mock_flow()

        input_df = pd.read_json(sample_input_file, lines=True)
        _call_sdg(
            output_artifact,
            output_metrics,
            input_pvc_path=sample_input_file,
            flow_id="test-flow",
        )

        output_df = pd.read_json(output_artifact.path, lines=True)
        pd.testing.assert_frame_equal(input_df, output_df)

    @patch("sdg_hub.core.flow.base.Flow.from_yaml")
    @patch("sdg_hub.core.flow.registry.FlowRegistry.get_flow_path_safe")
    def test_metrics_written(
        self,
        mock_get_path,
        mock_from_yaml,
        output_artifact,
        output_metrics,
        sample_input_file,
    ):
        """Metrics file contains expected keys and values."""
        mock_get_path.return_value = "/resolved/flow.yaml"
        mock_from_yaml.return_value = _make_mock_flow()

        _call_sdg(
            output_artifact,
            output_metrics,
            input_pvc_path=sample_input_file,
            flow_id="test-flow",
        )

        with open(output_metrics.path) as f:
            metrics = json.load(f)

        assert "metrics" in metrics
        metric_names = {m["name"] for m in metrics["metrics"]}
        assert metric_names == {"input_rows", "output_rows", "execution_time_seconds"}

    @patch("sdg_hub.core.flow.base.Flow.from_yaml")
    @patch("sdg_hub.core.flow.registry.FlowRegistry.get_flow_path_safe")
    def test_metrics_row_counts(
        self,
        mock_get_path,
        mock_from_yaml,
        output_artifact,
        output_metrics,
        sample_input_file,
    ):
        """Metrics report correct row counts."""
        mock_get_path.return_value = "/resolved/flow.yaml"
        mock_from_yaml.return_value = _make_mock_flow()

        _call_sdg(
            output_artifact,
            output_metrics,
            input_pvc_path=sample_input_file,
            flow_id="test-flow",
        )

        with open(output_metrics.path) as f:
            metrics = json.load(f)

        by_name = {m["name"]: m["numberValue"] for m in metrics["metrics"]}
        assert by_name["input_rows"] == 3
        assert by_name["output_rows"] == 3

    @patch("sdg_hub.core.flow.base.Flow.from_yaml")
    @patch("sdg_hub.core.flow.registry.FlowRegistry.get_flow_path_safe")
    def test_metrics_execution_time(
        self, mock_get_path, mock_from_yaml, output_artifact, output_metrics
    ):
        """Metrics include a non-negative execution time."""
        mock_get_path.return_value = "/resolved/flow.yaml"
        mock_from_yaml.return_value = _make_mock_flow()

        _call_sdg(
            output_artifact,
            output_metrics,
            flow_id="test-flow",
        )

        with open(output_metrics.path) as f:
            metrics = json.load(f)

        by_name = {m["name"]: m["numberValue"] for m in metrics["metrics"]}
        assert by_name["execution_time_seconds"] >= 0


# =============================================================================
# LOGGING TESTS
# =============================================================================


class TestLogging:
    """Tests for logging configuration."""

    @patch("sdg_hub.core.flow.base.Flow.from_yaml")
    @patch("sdg_hub.core.flow.registry.FlowRegistry.get_flow_path_safe")
    def test_debug_log_level(
        self, mock_get_path, mock_from_yaml, output_artifact, output_metrics
    ):
        """Component accepts DEBUG log level without error."""
        mock_get_path.return_value = "/resolved/flow.yaml"
        mock_from_yaml.return_value = _make_mock_flow()

        _call_sdg(
            output_artifact, output_metrics, flow_id="test-flow", log_level="DEBUG"
        )
        assert os.path.exists(output_artifact.path)

    @patch("sdg_hub.core.flow.base.Flow.from_yaml")
    @patch("sdg_hub.core.flow.registry.FlowRegistry.get_flow_path_safe")
    def test_warning_log_level(
        self, mock_get_path, mock_from_yaml, output_artifact, output_metrics
    ):
        """Component accepts WARNING log level without error."""
        mock_get_path.return_value = "/resolved/flow.yaml"
        mock_from_yaml.return_value = _make_mock_flow()

        _call_sdg(
            output_artifact, output_metrics, flow_id="test-flow", log_level="WARNING"
        )
        assert os.path.exists(output_artifact.path)


# =============================================================================
# MODULE EXPORTS TESTS
# =============================================================================


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


# =============================================================================
# FLOW SELECTION TESTS
# =============================================================================


class TestFlowSelection:
    """Tests for flow selection logic."""

    def test_no_flow_specified_raises(
        self, output_artifact, output_metrics, sample_input_file
    ):
        """Raises ValueError when neither flow_id nor flow_yaml_path provided."""
        with pytest.raises(ValueError, match="Either 'flow_id' or 'flow_yaml_path'"):
            _call_sdg(
                output_artifact,
                output_metrics,
                input_pvc_path=sample_input_file,
                flow_id="",
                flow_yaml_path="",
            )

    def test_flow_yaml_path_not_found_raises(
        self, output_artifact, output_metrics, sample_input_file
    ):
        """Raises FileNotFoundError when flow_yaml_path does not exist."""
        with pytest.raises(FileNotFoundError, match="Custom flow YAML not found"):
            _call_sdg(
                output_artifact,
                output_metrics,
                input_pvc_path=sample_input_file,
                flow_yaml_path="/nonexistent/flow.yaml",
            )

    @patch("sdg_hub.core.flow.registry.FlowRegistry.get_flow_path_safe")
    def test_invalid_flow_id_raises(
        self, mock_get_path, output_artifact, output_metrics, sample_input_file
    ):
        """Raises ValueError when flow_id is not found in registry."""
        mock_get_path.side_effect = ValueError("Flow 'bad-id' not found.")
        with pytest.raises(ValueError, match="Flow lookup failed"):
            _call_sdg(
                output_artifact,
                output_metrics,
                input_pvc_path=sample_input_file,
                flow_id="bad-id",
            )

    @patch("sdg_hub.core.flow.base.Flow.from_yaml")
    @patch("sdg_hub.core.flow.registry.FlowRegistry.get_flow_path_safe")
    def test_flow_id_resolves_and_loads(
        self,
        mock_get_path,
        mock_from_yaml,
        output_artifact,
        output_metrics,
        sample_input_file,
    ):
        """flow_id resolves via registry and loads successfully."""
        mock_get_path.return_value = "/resolved/flow.yaml"
        mock_from_yaml.return_value = _make_mock_flow()

        _call_sdg(
            output_artifact,
            output_metrics,
            input_pvc_path=sample_input_file,
            flow_id="test-flow-id",
        )

        mock_get_path.assert_called_once_with("test-flow-id")
        mock_from_yaml.assert_called_once_with("/resolved/flow.yaml")

    @patch("sdg_hub.core.flow.base.Flow.from_yaml")
    def test_flow_yaml_path_loads_directly(
        self,
        mock_from_yaml,
        output_artifact,
        output_metrics,
        sample_input_file,
        tmp_dir,
    ):
        """flow_yaml_path loads flow directly without registry lookup."""
        yaml_path = os.path.join(tmp_dir, "custom_flow.yaml")
        with open(yaml_path, "w") as f:
            f.write("dummy")

        mock_from_yaml.return_value = _make_mock_flow()

        _call_sdg(
            output_artifact,
            output_metrics,
            input_pvc_path=sample_input_file,
            flow_yaml_path=yaml_path,
        )

        mock_from_yaml.assert_called_once_with(yaml_path)

    @patch("sdg_hub.core.flow.base.Flow.from_yaml")
    def test_flow_yaml_path_takes_precedence(
        self,
        mock_from_yaml,
        output_artifact,
        output_metrics,
        sample_input_file,
        tmp_dir,
    ):
        """When both flow_id and flow_yaml_path are provided, flow_yaml_path wins."""
        yaml_path = os.path.join(tmp_dir, "custom_flow.yaml")
        with open(yaml_path, "w") as f:
            f.write("dummy")

        mock_from_yaml.return_value = _make_mock_flow()

        _call_sdg(
            output_artifact,
            output_metrics,
            input_pvc_path=sample_input_file,
            flow_id="some-id",
            flow_yaml_path=yaml_path,
        )

        mock_from_yaml.assert_called_once_with(yaml_path)


# =============================================================================
# MODEL CONFIGURATION TESTS
# =============================================================================


class TestModelConfiguration:
    """Tests for model configuration logic."""

    @patch("sdg_hub.core.flow.base.Flow.from_yaml")
    @patch("sdg_hub.core.flow.registry.FlowRegistry.get_flow_path_safe")
    def test_llm_flow_without_model_raises(
        self,
        mock_get_path,
        mock_from_yaml,
        output_artifact,
        output_metrics,
        sample_input_file,
    ):
        """Raises ValueError when flow has LLM blocks but no model provided."""
        mock_get_path.return_value = "/resolved/flow.yaml"
        mock_flow = _make_mock_flow()
        mock_flow.is_model_config_required.return_value = True
        mock_from_yaml.return_value = mock_flow

        with pytest.raises(ValueError, match="requires a 'model' parameter"):
            _call_sdg(
                output_artifact,
                output_metrics,
                input_pvc_path=sample_input_file,
                flow_id="llm-flow-id",
                model="",
            )

    @patch("sdg_hub.core.flow.base.Flow.from_yaml")
    @patch("sdg_hub.core.flow.registry.FlowRegistry.get_flow_path_safe")
    def test_llm_flow_with_model_configures(
        self,
        mock_get_path,
        mock_from_yaml,
        output_artifact,
        output_metrics,
        sample_input_file,
    ):
        """Model config is applied when flow has LLM blocks and model is provided."""
        mock_get_path.return_value = "/resolved/flow.yaml"
        mock_flow = _make_mock_flow()
        mock_flow.is_model_config_required.return_value = True
        mock_from_yaml.return_value = mock_flow

        _call_sdg(
            output_artifact,
            output_metrics,
            input_pvc_path=sample_input_file,
            flow_id="llm-flow-id",
            model="hosted_vllm/test-model",
            temperature=0.5,
            max_tokens=1024,
        )

        mock_flow.set_model_config.assert_called_once()
        call_kwargs = mock_flow.set_model_config.call_args
        assert call_kwargs.kwargs["model"] == "hosted_vllm/test-model"
        assert call_kwargs.kwargs["temperature"] == 0.5
        assert call_kwargs.kwargs["max_tokens"] == 1024

    @patch("sdg_hub.core.flow.base.Flow.from_yaml")
    @patch("sdg_hub.core.flow.registry.FlowRegistry.get_flow_path_safe")
    def test_non_llm_flow_skips_model_config(
        self,
        mock_get_path,
        mock_from_yaml,
        output_artifact,
        output_metrics,
        sample_input_file,
    ):
        """No model config applied when flow has no LLM blocks."""
        mock_get_path.return_value = "/resolved/flow.yaml"
        mock_flow = _make_mock_flow()
        mock_flow.is_model_config_required.return_value = False
        mock_from_yaml.return_value = mock_flow

        _call_sdg(
            output_artifact,
            output_metrics,
            input_pvc_path=sample_input_file,
            flow_id="transform-id",
        )

        mock_flow.set_model_config.assert_not_called()

    @patch.dict(
        os.environ,
        {"LLM_API_KEY": "test-key", "LLM_API_BASE": "http://localhost:8080/v1"},
    )
    @patch("sdg_hub.core.flow.base.Flow.from_yaml")
    @patch("sdg_hub.core.flow.registry.FlowRegistry.get_flow_path_safe")
    def test_env_credentials_passed_to_model_config(
        self,
        mock_get_path,
        mock_from_yaml,
        output_artifact,
        output_metrics,
        sample_input_file,
    ):
        """API key and base URL from environment are passed to set_model_config."""
        mock_get_path.return_value = "/resolved/flow.yaml"
        mock_flow = _make_mock_flow()
        mock_flow.is_model_config_required.return_value = True
        mock_from_yaml.return_value = mock_flow

        _call_sdg(
            output_artifact,
            output_metrics,
            input_pvc_path=sample_input_file,
            flow_id="llm-flow-id",
            model="openai/gpt-4",
        )

        call_kwargs = mock_flow.set_model_config.call_args
        assert call_kwargs.kwargs["api_key"] == "test-key"
        assert call_kwargs.kwargs["api_base"] == "http://localhost:8080/v1"


# =============================================================================
# DATASET VALIDATION TESTS
# =============================================================================


class TestDatasetValidation:
    """Tests for dataset validation before flow execution."""

    @patch("sdg_hub.core.flow.base.Flow.from_yaml")
    @patch("sdg_hub.core.flow.registry.FlowRegistry.get_flow_path_safe")
    def test_dataset_validation_failure_raises(
        self,
        mock_get_path,
        mock_from_yaml,
        output_artifact,
        output_metrics,
        sample_input_file,
    ):
        """Raises FlowValidationError when dataset fails validation."""
        mock_get_path.return_value = "/resolved/flow.yaml"
        mock_flow = _make_mock_flow()
        mock_flow.validate_dataset.return_value = [
            "Missing required column: 'special_column'"
        ]
        mock_from_yaml.return_value = mock_flow

        with pytest.raises(FlowValidationError, match="Dataset validation failed"):
            _call_sdg(
                output_artifact,
                output_metrics,
                input_pvc_path=sample_input_file,
                flow_id="strict-flow-id",
            )


# =============================================================================
# FLOW EXECUTION TESTS
# =============================================================================


class TestFlowExecution:
    """Tests for flow execution integration."""

    @patch("sdg_hub.core.flow.base.Flow.from_yaml")
    @patch("sdg_hub.core.flow.registry.FlowRegistry.get_flow_path_safe")
    def test_generate_called_with_correct_params(
        self,
        mock_get_path,
        mock_from_yaml,
        output_artifact,
        output_metrics,
        sample_input_file,
    ):
        """flow.generate() is called with max_concurrency."""
        mock_get_path.return_value = "/resolved/flow.yaml"
        mock_from_yaml.return_value = _make_mock_flow()

        _call_sdg(
            output_artifact,
            output_metrics,
            input_pvc_path=sample_input_file,
            flow_id="test-flow-id",
            max_concurrency=20,
        )

        mock_flow = mock_from_yaml.return_value
        mock_flow.generate.assert_called_once()
        call_kwargs = mock_flow.generate.call_args.kwargs
        assert call_kwargs["max_concurrency"] == 20

    @patch("sdg_hub.core.flow.base.Flow.from_yaml")
    @patch("sdg_hub.core.flow.registry.FlowRegistry.get_flow_path_safe")
    def test_checkpointing_params_passed(
        self,
        mock_get_path,
        mock_from_yaml,
        output_artifact,
        output_metrics,
        sample_input_file,
    ):
        """Checkpoint params are forwarded to flow.generate()."""
        mock_get_path.return_value = "/resolved/flow.yaml"
        mock_from_yaml.return_value = _make_mock_flow()

        _call_sdg(
            output_artifact,
            output_metrics,
            input_pvc_path=sample_input_file,
            flow_id="test-flow-id",
            checkpoint_pvc_path="/mnt/checkpoints/",
            save_freq=50,
        )

        mock_flow = mock_from_yaml.return_value
        call_kwargs = mock_flow.generate.call_args.kwargs
        assert call_kwargs["checkpoint_dir"] == "/mnt/checkpoints/"
        assert call_kwargs["save_freq"] == 50

    @patch("sdg_hub.core.flow.base.Flow.from_yaml")
    @patch("sdg_hub.core.flow.registry.FlowRegistry.get_flow_path_safe")
    def test_no_checkpoint_when_not_configured(
        self,
        mock_get_path,
        mock_from_yaml,
        output_artifact,
        output_metrics,
        sample_input_file,
    ):
        """No checkpoint params passed when checkpoint_pvc_path is empty."""
        mock_get_path.return_value = "/resolved/flow.yaml"
        mock_from_yaml.return_value = _make_mock_flow()

        _call_sdg(
            output_artifact,
            output_metrics,
            input_pvc_path=sample_input_file,
            flow_id="test-flow-id",
            checkpoint_pvc_path="",
        )

        mock_flow = mock_from_yaml.return_value
        call_kwargs = mock_flow.generate.call_args.kwargs
        assert "checkpoint_dir" not in call_kwargs

    @patch("sdg_hub.core.flow.base.Flow.from_yaml")
    @patch("sdg_hub.core.flow.registry.FlowRegistry.get_flow_path_safe")
    def test_output_reflects_flow_result(
        self,
        mock_get_path,
        mock_from_yaml,
        output_artifact,
        output_metrics,
        sample_input_file,
    ):
        """Output artifact contains the DataFrame returned by flow.generate()."""
        mock_get_path.return_value = "/resolved/flow.yaml"
        enriched_df = pd.DataFrame(
            {
                "document": ["Doc one."],
                "domain": ["science"],
                "generated_qa": ["Q: What? A: Something."],
            }
        )
        mock_from_yaml.return_value = _make_mock_flow(return_df=enriched_df)

        _call_sdg(
            output_artifact,
            output_metrics,
            input_pvc_path=sample_input_file,
            flow_id="test-flow-id",
        )

        result = pd.read_json(output_artifact.path, lines=True)
        assert "generated_qa" in result.columns
        assert len(result) == 1

    @patch("sdg_hub.core.flow.base.Flow.from_yaml")
    @patch("sdg_hub.core.flow.registry.FlowRegistry.get_flow_path_safe")
    def test_flow_execution_error_propagates(
        self,
        mock_get_path,
        mock_from_yaml,
        output_artifact,
        output_metrics,
        sample_input_file,
    ):
        """Errors from flow.generate() propagate as-is."""
        mock_get_path.return_value = "/resolved/flow.yaml"
        mock_flow = _make_mock_flow()
        mock_flow.generate.side_effect = FlowValidationError("Block 'gen' failed")
        mock_from_yaml.return_value = mock_flow

        with pytest.raises(FlowValidationError, match="Block 'gen' failed"):
            _call_sdg(
                output_artifact,
                output_metrics,
                input_pvc_path=sample_input_file,
                flow_id="failing-flow-id",
            )
