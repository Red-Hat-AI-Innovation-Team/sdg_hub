"""Tests for Flow path handling functionality."""

# Standard
from unittest.mock import MagicMock, patch
import os
import unittest

# Third Party
import yaml

# First Party
from sdg_hub.flow import Flow


class TestFlow(unittest.TestCase):
    def setUp(self):
        self.flow = Flow(MagicMock())

    def test_config_relative_to_flow(self):
        flow = self.flow.get_flow_from_file("tests/flows/testdata/test_flow_1.yaml")
        block = flow.chained_blocks[0]["block_type"](
            **flow.chained_blocks[0]["block_config"]
        )

        self.assertEqual(block.block_config["introduction"], "intro")

    def test_config_relative_to_package(self):
        with open(
            "tests/flows/testdata/test_flow_1.yaml", "r", encoding="utf-8"
        ) as yaml_file:
            y = yaml.safe_load(yaml_file)
        y[0]["block_config"]["config_path"] = (
            "configs/skills/simple_generate_qa_freeform.yaml"
        )
        with patch("yaml.safe_load", new_callable=MagicMock) as mock_safe_load:
            mock_safe_load.return_value = y
            # Mock the validation to avoid conflicts with yaml.safe_load mock
            with patch.object(self.flow, "validate_config_files") as mock_validate:
                # First Party
                from sdg_hub.flow import ValidationResult

                mock_validate.return_value = ValidationResult(valid=True, errors=[])
                flow = self.flow.get_flow_from_file(
                    "tests/flows/testdata/test_flow_1.yaml"
                )
        block = flow.chained_blocks[0]["block_type"](
            **flow.chained_blocks[0]["block_config"]
        )

        self.assertEqual(
            block.block_config["introduction"],
            "Develop a series of question and answer pairs to perform a task.",
        )

    def test_config_absolute(self):
        with open(
            "tests/flows/testdata/test_flow_1.yaml", "r", encoding="utf-8"
        ) as yaml_file:
            y = yaml.safe_load(yaml_file)
        y[0]["block_config"]["config_path"] = os.path.abspath(
            "src/sdg_hub/configs/skills/simple_generate_qa_freeform.yaml"
        )

        with patch("yaml.safe_load", new_callable=MagicMock) as mock_safe_load:
            mock_safe_load.return_value = y
            # Mock the validation to avoid conflicts with yaml.safe_load mock
            with patch.object(self.flow, "validate_config_files") as mock_validate:
                # First Party
                from sdg_hub.flow import ValidationResult

                mock_validate.return_value = ValidationResult(valid=True, errors=[])
                flow = self.flow.get_flow_from_file(
                    "tests/flows/testdata/test_flow_1.yaml"
                )
        block = flow.chained_blocks[0]["block_type"](
            **flow.chained_blocks[0]["block_config"]
        )

        self.assertEqual(
            block.block_config["introduction"],
            "Develop a series of question and answer pairs to perform a task.",
        )

    def test_config_list_mix(self):
        with open(
            "tests/flows/testdata/test_flow_2.yaml", "r", encoding="utf-8"
        ) as yaml_file:
            y = yaml.safe_load(yaml_file)
        y[0]["block_config"]["config_paths"]["k3"] = os.path.abspath(
            "src/sdg_hub/configs/skills/simple_generate_qa_freeform.yaml"
        )

        with patch("yaml.safe_load", new_callable=MagicMock) as mock_safe_load:
            mock_safe_load.return_value = y
            # Mock the validation to avoid conflicts with yaml.safe_load mock
            with patch.object(self.flow, "validate_config_files") as mock_validate:
                # First Party
                from sdg_hub.flow import ValidationResult

                mock_validate.return_value = ValidationResult(valid=True, errors=[])
                flow = self.flow.get_flow_from_file(
                    "tests/flows/testdata/test_flow_2.yaml"
                )
        block = flow.chained_blocks[0]["block_type"](
            **flow.chained_blocks[0]["block_config"]
        )

        self.assertEqual(block.block_config["introduction"], "intro")
        self.assertEqual(len(block.prompt_template), 3)
