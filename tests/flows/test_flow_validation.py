import pytest
from unittest.mock import patch, MagicMock
from sdg_hub.flow import Flow

@pytest.fixture
def flow_with_invalid_config():
    return Flow(llm_client=None)

def test_validate_config_path_missing_file(flow_with_invalid_config, tmp_path):
    """Test validate_config_files catches missing config_path."""
    flow_with_invalid_config.chained_blocks = [
        {
            "block_type": MagicMock,
            "block_config": {
                "block_name": "invalid_block",
                "config_path": str(tmp_path / "nonexistent.yaml")
            },
        }
    ]

    result = flow_with_invalid_config.validate_config_files()

    assert not result.valid
    assert any("does not exist" in msg for msg in result.errors)

def test_validate_config_path_permission_denied(flow_with_invalid_config, tmp_path):
    config_path = tmp_path / "unreadable.yaml"
    config_path.write_text("valid: yaml")
    config_path.chmod(0o000)  

    flow_with_invalid_config.chained_blocks = [
        {
            "block_type": MagicMock,
            "block_config": {
                "block_name": "bad_perm_block",
                "config_path": str(config_path),
            },
        }
    ]

    try:
        result = flow_with_invalid_config.validate_config_files()
        assert not result.valid
        assert any("not readable" in msg for msg in result.errors)
    finally:
        config_path.chmod(0o644) 