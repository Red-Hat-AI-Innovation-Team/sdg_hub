# SPDX-License-Identifier: Apache-2.0
"""Shared test fixtures for backend API tests."""

# Standard
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Generator, List
from unittest.mock import MagicMock, patch

# Add backend to path (needed for importing api_server)
BACKEND_DIR = Path(__file__).parent.parent.parent / "backend"
sys.path.insert(0, str(BACKEND_DIR))

# Note: sdg_hub is installed via pip (pip install -e ../.. from backend/)

# Handle pyarrow type registration conflict
# This MUST happen before importing pandas/datasets to prevent "already defined" errors
# when the datasets library tries to register its custom Array2D types
import pyarrow as pa  # noqa: E402

# Store original register function and create a wrapper that ignores duplicates
_original_register = pa.register_extension_type

def _safe_register_extension_type(ext_type):
    """Wrapper that ignores duplicate type registration errors."""
    try:
        _original_register(ext_type)
    except pa.ArrowKeyError:
        # Type already registered, ignore
        pass

# Monkey-patch pyarrow to handle duplicate registration gracefully
pa.register_extension_type = _safe_register_extension_type

# Third Party
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402


# ============================================================================
# Mock Classes for sdg_hub
# ============================================================================

class MockFlowMetadata:
    """Mock FlowMetadata class."""
    
    def __init__(
        self,
        name: str = "Test Flow",
        id: str = "test-flow-id",
        description: str = "A test flow",
        version: str = "1.0.0",
        author: str = "Test Author",
        tags: List[str] = None,
        recommended_models: Any = None,
    ):
        self.name = name
        self.id = id
        self.description = description
        self.version = version
        self.author = author
        self.tags = tags or ["test"]
        self.recommended_models = recommended_models


class MockRecommendedModels:
    """Mock RecommendedModels class."""
    
    def __init__(
        self,
        default: str = "test-model",
        compatible: List[str] = None,
        experimental: List[str] = None,
    ):
        self.default = default
        self.compatible = compatible or []
        self.experimental = experimental or []


class MockDatasetRequirements:
    """Mock DatasetRequirements class."""
    
    def __init__(
        self,
        required_columns: List[str] = None,
        optional_columns: List[str] = None,
        min_samples: int = 1,
    ):
        self.required_columns = required_columns or ["input"]
        self.optional_columns = optional_columns or []
        self.min_samples = min_samples
    
    def model_dump(self) -> Dict[str, Any]:
        return {
            "required_columns": self.required_columns,
            "optional_columns": self.optional_columns,
            "min_samples": self.min_samples,
        }


class MockBlock:
    """Mock Block class for testing."""
    
    def __init__(
        self,
        block_name: str = "test_block",
        input_cols: List[str] = None,
        output_cols: List[str] = None,
        **kwargs
    ):
        self.block_name = block_name
        self.input_cols = input_cols or ["input"]
        self.output_cols = output_cols or ["output"]
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def __call__(self, dataset, **kwargs):
        """Mock block execution."""
        result = dataset.copy()
        for col in self.output_cols:
            result[col] = [f"{self.block_name}_{col}_{i}" for i in range(len(dataset))]
        return result


class MockFlow:
    """Mock Flow class for testing."""
    
    def __init__(
        self,
        blocks: List[MockBlock] = None,
        metadata: MockFlowMetadata = None,
    ):
        self.blocks = blocks or []
        self.metadata = metadata or MockFlowMetadata()
        self._model_config_set = False
        self._model_config = {}
    
    def get_model_recommendations(self) -> Dict[str, Any]:
        """Get model recommendations."""
        if self.metadata.recommended_models:
            return {
                "default": self.metadata.recommended_models.default,
                "compatible": self.metadata.recommended_models.compatible,
                "experimental": self.metadata.recommended_models.experimental,
            }
        return {"default": None, "compatible": [], "experimental": []}
    
    def get_default_model(self) -> str:
        """Get default model."""
        if self.metadata.recommended_models:
            return self.metadata.recommended_models.default
        return None
    
    def is_model_config_required(self) -> bool:
        """Check if model config is required."""
        return len(self.blocks) > 0
    
    def is_model_config_set(self) -> bool:
        """Check if model config is set."""
        return self._model_config_set
    
    def set_model_config(self, **kwargs):
        """Set model configuration."""
        self._model_config = kwargs
        self._model_config_set = True
    
    def get_dataset_requirements(self) -> MockDatasetRequirements:
        """Get dataset requirements."""
        return MockDatasetRequirements()
    
    def get_dataset_schema(self) -> pd.DataFrame:
        """Get dataset schema."""
        return pd.DataFrame(columns=["input", "output"])
    
    def get_info(self) -> Dict[str, Any]:
        """Get flow information."""
        return {
            "metadata": {
                "name": self.metadata.name,
                "description": self.metadata.description,
                "version": self.metadata.version,
                "author": self.metadata.author,
                "tags": self.metadata.tags,
            },
            "total_blocks": len(self.blocks),
            "block_names": [b.block_name for b in self.blocks],
            "blocks": [
                {
                    "block_name": b.block_name,
                    "input_cols": b.input_cols,
                    "output_cols": b.output_cols,
                }
                for b in self.blocks
            ],
        }
    
    def dry_run(self, dataset, sample_size=2, **kwargs) -> Dict[str, Any]:
        """Perform dry run."""
        actual_size = min(sample_size, len(dataset))
        return {
            "flow_name": self.metadata.name,
            "sample_size": actual_size,
            "original_dataset_size": len(dataset),
            "execution_successful": True,
            "blocks_executed": [
                {
                    "block_name": b.block_name,
                    "input_rows": actual_size,
                    "output_rows": actual_size,
                    "parameters_used": {},
                }
                for b in self.blocks
            ],
        }
    
    def generate(self, dataset, **kwargs) -> pd.DataFrame:
        """Generate output."""
        result = dataset.copy()
        for block in self.blocks:
            result = block(result, **kwargs)
        return result
    
    @classmethod
    def from_yaml(cls, path: str) -> "MockFlow":
        """Load flow from YAML file."""
        import yaml
        
        flow_name = f"Flow from {Path(path).name}"
        blocks = [MockBlock()]
        recommended_models = None
        
        # Try to read actual YAML file if it exists
        try:
            with open(path, "r") as f:
                flow_data = yaml.safe_load(f)
                if flow_data and "metadata" in flow_data:
                    metadata = flow_data["metadata"]
                    flow_name = metadata.get("name", flow_name)
                    if "recommended_models" in metadata:
                        rec_models = metadata["recommended_models"]
                        recommended_models = MockRecommendedModels(
                            default=rec_models.get("default"),
                            compatible=rec_models.get("compatible", []),
                            experimental=rec_models.get("experimental", []),
                        )
        except Exception:
            pass  # Use defaults if file can't be read
        
        return cls(
            blocks=blocks,
            metadata=MockFlowMetadata(
                name=flow_name,
                recommended_models=recommended_models,
            ),
        )


class MockFlowRegistry:
    """Mock FlowRegistry class."""
    
    _flows = {}
    _search_paths = []
    
    @classmethod
    def discover_flows(cls):
        """Discover flows."""
        cls._flows = {
            "Test Flow": "/path/to/test_flow/flow.yaml",
            "Another Flow": "/path/to/another_flow/flow.yaml",
        }
    
    @classmethod
    def list_flows(cls) -> List[Dict[str, str]]:
        """List all flows."""
        return [{"name": name} for name in cls._flows.keys()]
    
    @classmethod
    def search_flows(cls, tag: str = None) -> List[Dict[str, str]]:
        """Search flows by tag."""
        return cls.list_flows()
    
    @classmethod
    def get_flow_path(cls, flow_name: str) -> str:
        """Get flow path."""
        return cls._flows.get(flow_name)
    
    @classmethod
    def clear(cls):
        """Clear registry."""
        cls._flows.clear()
        cls._search_paths.clear()


class MockBlockRegistry:
    """Mock BlockRegistry class."""
    
    _blocks = {}
    _metadata = {}
    
    @classmethod
    def discover_blocks(cls):
        """Discover blocks."""
        cls._blocks = {
            "LLMChatBlock": MockBlock,
            "PromptBuilderBlock": MockBlock,
            "TextConcatBlock": MockBlock,
        }
    
    @classmethod
    def list_blocks(cls) -> List[str]:
        """List all blocks."""
        return list(cls._blocks.keys())
    
    @classmethod
    def _get(cls, block_type: str):
        """Get block class."""
        return cls._blocks.get(block_type, MockBlock)


# ============================================================================
# Pytest Fixtures
# ============================================================================

@pytest.fixture
def temp_dir() -> Generator[str, None, None]:
    """Create a temporary directory for tests."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Cleanup
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def sample_dataset() -> pd.DataFrame:
    """Create a sample dataset for testing."""
    return pd.DataFrame({
        "input": ["test input 1", "test input 2", "test input 3"],
        "label": ["label1", "label2", "label3"],
    })


@pytest.fixture
def empty_dataset() -> pd.DataFrame:
    """Create an empty dataset for testing."""
    return pd.DataFrame({"input": [], "label": []})


@pytest.fixture
def sample_jsonl_file(temp_dir) -> str:
    """Create a sample JSONL file."""
    file_path = Path(temp_dir) / "test_data.jsonl"
    data = [
        {"input": "test 1", "label": "a"},
        {"input": "test 2", "label": "b"},
        {"input": "test 3", "label": "c"},
    ]
    with open(file_path, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
    return str(file_path)


@pytest.fixture
def sample_csv_file(temp_dir) -> str:
    """Create a sample CSV file."""
    file_path = Path(temp_dir) / "test_data.csv"
    df = pd.DataFrame({
        "input": ["test 1", "test 2", "test 3"],
        "label": ["a", "b", "c"],
    })
    df.to_csv(file_path, index=False)
    return str(file_path)


@pytest.fixture
def mock_flow() -> MockFlow:
    """Create a mock flow for testing."""
    return MockFlow(
        blocks=[MockBlock(block_name="test_block")],
        metadata=MockFlowMetadata(
            name="Test Flow",
            recommended_models=MockRecommendedModels(
                default="test-model",
                compatible=["alt-model"],
            ),
        ),
    )


@pytest.fixture
def mock_sdg_hub(mock_flow):
    """Mock the entire sdg_hub module.

    After the backend refactoring, Flow / FlowRegistry / BlockRegistry are
    imported directly by each router module and by utility modules, so we
    must patch every location that uses them.
    """
    # Create a mock sdg_hub module with proper attributes so that
    # local imports like `from sdg_hub import FlowRegistry` inside
    # utility functions (e.g. utils/security.py) also get the mocks.
    mock_sdg_hub_module = MagicMock()
    mock_flow_cls = MagicMock(return_value=mock_flow)
    mock_flow_cls.from_yaml = MockFlow.from_yaml
    mock_sdg_hub_module.Flow = mock_flow_cls
    mock_sdg_hub_module.FlowRegistry = MockFlowRegistry
    mock_sdg_hub_module.BlockRegistry = MockBlockRegistry

    with patch.dict("sys.modules", {
        "sdg_hub": mock_sdg_hub_module,
    }):
        # Modules that import Flow/FlowRegistry/BlockRegistry at module level
        # need explicit patches because the names were already bound at import time.
        flow_targets = [
            "routers.flows.Flow",
            "routers.execution.Flow",
            "routers.config.Flow",
            "routers.configurations.Flow",
        ]
        flow_registry_targets = [
            "routers.flows.FlowRegistry",
            "routers.execution.FlowRegistry",
            "routers.config.FlowRegistry",
            "routers.configurations.FlowRegistry",
        ]
        block_registry_targets = [
            "routers.execution.BlockRegistry",
            "routers.config.BlockRegistry",
        ]

        patches = []

        for target in flow_targets:
            p = patch(target, mock_flow_cls)
            patches.append(p)
        for target in flow_registry_targets:
            p = patch(target, MockFlowRegistry)
            patches.append(p)
        for target in block_registry_targets:
            p = patch(target, MockBlockRegistry)
            patches.append(p)

        for p in patches:
            p.start()

        MockFlowRegistry.discover_flows()
        MockBlockRegistry.discover_blocks()

        try:
            yield {
                "Flow": mock_flow_cls,
                "FlowRegistry": MockFlowRegistry,
                "BlockRegistry": MockBlockRegistry,
            }
        finally:
            for p in patches:
                p.stop()


@pytest.fixture
def test_client(mock_sdg_hub, temp_dir) -> TestClient:
    """Create a FastAPI test client with mocked dependencies."""
    # Set environment variables for testing
    os.environ["SDG_HUB_MAX_UPLOAD_MB"] = "10"
    
    # Resolve temp_path to handle macOS /var -> /private/var symlinks.
    # is_path_within_allowed_dirs resolves candidate paths, so allowed dirs
    # must also be resolved for relative_to() checks to succeed.
    temp_path = Path(temp_dir).resolve()
    uploads_dir = (temp_path / "uploads").resolve()
    custom_flows_dir = (temp_path / "custom_flows").resolve()
    checkpoints_dir = (temp_path / "checkpoints").resolve()
    outputs_dir = (temp_path / "outputs").resolve()
    
    # Create directories
    uploads_dir.mkdir(exist_ok=True)
    custom_flows_dir.mkdir(exist_ok=True)
    checkpoints_dir.mkdir(exist_ok=True)
    outputs_dir.mkdir(exist_ok=True)
    
    # ALLOWED_FLOW_READ_DIRS and ALLOWED_DATASET_DIRS are evaluated at module
    # load time from the original directory constants.  The security-hardened
    # endpoints validate paths against these lists, so we must include the temp
    # directory (and its subdirectories) so that test-created flow / dataset
    # files pass validation.
    allowed_flow_dirs = [custom_flows_dir, temp_path]
    allowed_dataset_dirs = [uploads_dir, outputs_dir, temp_path]
    saved_config_file = temp_path / "saved_configurations.json"
    runs_history_file = temp_path / "runs_history.json"

    # After refactoring, config values live in `config` module and are
    # imported by router / utility modules.  We must patch every location.
    config_patches = {
        "UPLOADS_DIR": (uploads_dir, [
            "config",
            "routers.datasets",
            "routers.execution",
            "routers.runs",
        ]),
        "CUSTOM_FLOWS_DIR": (custom_flows_dir, [
            "config",
            "routers.flows",
            "routers.execution",
            "routers.workspace",
            "routers.config",
            "utils.security",
        ]),
        "CHECKPOINTS_DIR": (checkpoints_dir, [
            "config",
            "utils.checkpoint_utils",
        ]),
        "OUTPUTS_DIR": (outputs_dir, [
            "config",
            "routers.execution",
        ]),
        "DATA_DIR": (temp_path, [
            "config",
            "utils.security",
            "routers.runs",
        ]),
        "ALLOWED_FLOW_READ_DIRS": (allowed_flow_dirs, [
            "config",
            "routers.flows",
            "routers.execution",
            "routers.workspace",
            "utils.security",
        ]),
        "ALLOWED_DATASET_DIRS": (allowed_dataset_dirs, [
            "config",
            "utils.security",
            "routers.runs",
        ]),
    }

    file_patches = [
        ("SAVED_CONFIG_FILE", saved_config_file, [
            "config",
            "utils.config_utils",
        ]),
        ("RUNS_HISTORY_FILE", runs_history_file, [
            "config",
            "utils.dataset_utils",
        ]),
    ]

    patches = []
    for name, (value, modules) in config_patches.items():
        for mod in modules:
            patches.append(patch(f"{mod}.{name}", value))
    for name, value, modules in file_patches:
        for mod in modules:
            patches.append(patch(f"{mod}.{name}", value))

    for p in patches:
        p.start()

    try:
        # Import and create client
        from api_server import app
        client = TestClient(app)
        yield client
    finally:
        for p in patches:
            p.stop()


@pytest.fixture
def saved_config_file(temp_dir) -> str:
    """Create a saved configurations file."""
    file_path = Path(temp_dir) / "saved_configurations.json"
    configs = [
        {
            "id": "config-1",
            "flow_name": "Test Flow",
            "flow_id": "test-flow-id",
            "model_configuration": {"model": "test-model"},
            "dataset_configuration": {"data_files": "test.jsonl"},
            "status": "configured",
            "created_at": "2024-01-01T00:00:00",
        }
    ]
    with open(file_path, "w") as f:
        json.dump(configs, f)
    return str(file_path)


@pytest.fixture
def runs_history_file(temp_dir) -> str:
    """Create a runs history file."""
    file_path = Path(temp_dir) / "runs_history.json"
    runs = [
        {
            "run_id": "run-1",
            "config_id": "config-1",
            "flow_name": "Test Flow",
            "status": "completed",
            "start_time": "2024-01-01T00:00:00",
            "output_samples": 100,
        }
    ]
    with open(file_path, "w") as f:
        json.dump(runs, f)
    return str(file_path)


# ============================================================================
# Helper Functions
# ============================================================================

def create_mock_response(status: str = "success", **kwargs) -> Dict[str, Any]:
    """Create a mock API response."""
    return {"status": status, **kwargs}


def create_test_config(
    flow_name: str = "Test Flow",
    model: str = "test-model",
    data_files: str = "test.jsonl",
) -> Dict[str, Any]:
    """Create a test configuration."""
    return {
        "flow_name": flow_name,
        "flow_id": "test-id",
        "flow_path": "/path/to/flow.yaml",
        "model_configuration": {
            "model": model,
            "api_base": "http://localhost:8000/v1",
            "api_key": "env:TEST_API_KEY",
        },
        "dataset_configuration": {
            "data_files": data_files,
            "file_format": "auto",
        },
        "dry_run_configuration": {
            "sample_size": 2,
            "enable_time_estimation": True,
            "max_concurrency": 10,
        },
        "tags": ["test"],
        "status": "configured",
    }

