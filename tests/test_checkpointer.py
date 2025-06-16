# Standard
import os
import tempfile
import unittest
import json
from unittest.mock import patch

# Third Party
from datasets import Dataset, Features, Value

# First Party
from sdg_hub.checkpointer import Checkpointer


class TestCheckpointer(unittest.TestCase):
    def setUp(self):
        # Initialize test environment with temporary directory and sample dataset
        self.temp_dir = tempfile.TemporaryDirectory()
        self.checkpointer = Checkpointer(checkpoint_dir=self.temp_dir.name, save_freq=2)
        self.test_dataset = Dataset.from_dict(
            {
                "instruction": [
                    "Generate a question about Python programming",
                    "Create a question about data structures",
                    "Write a question about algorithms",
                ],
                "input": [
                    "Python basics",
                    "Data structures in Python",
                    "Sorting algorithms",
                ],
                "output": [
                    "What is the difference between a list and a tuple in Python?",
                    "How does a binary search tree work?",
                    "Explain the time complexity of quicksort",
                ],
                "metadata": [
                    json.dumps({"difficulty": "beginner", "topic": "python"}),
                    json.dumps(
                        {"difficulty": "intermediate", "topic": "data_structures"}
                    ),
                    json.dumps({"difficulty": "advanced", "topic": "algorithms"}),
                ],
            },
            features=Features(
                {
                    "instruction": Value("string"),
                    "input": Value("string"),
                    "output": Value("string"),
                    "metadata": Value("string"),
                }
            ),
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_basic_checkpointing(self):
        # Verify initial state and checkpoint creation
        remaining_data, pre_generated = self.checkpointer.load_existing_data(
            self.test_dataset
        )
        self.assertEqual(remaining_data.num_rows, self.test_dataset.num_rows)
        self.assertIsNone(pre_generated)

        self.checkpointer.save_intermediate_checkpoint(self.test_dataset)
        self.assertTrue(
            any(
                f.startswith("data_checkpoint_") for f in os.listdir(self.temp_dir.name)
            )
        )

    def test_load_existing_data(self):
        # Test loading with overlapping and new data
        self.checkpointer.save_intermediate_checkpoint(self.test_dataset)

        new_dataset = Dataset.from_dict(
            {
                "instruction": [
                    "Generate a question about Python programming",
                    "Create a question about machine learning",
                    "Write a question about databases",
                ],
                "input": ["Python basics", "ML fundamentals", "SQL basics"],
                "output": [
                    "What is the difference between a list and a tuple in Python?",
                    "What is supervised learning?",
                    "What is a primary key?",
                ],
                "metadata": [
                    json.dumps({"difficulty": "beginner", "topic": "python"}),
                    json.dumps({"difficulty": "intermediate", "topic": "ml"}),
                    json.dumps({"difficulty": "beginner", "topic": "databases"}),
                ],
            },
            features=Features(
                {
                    "instruction": Value("string"),
                    "input": Value("string"),
                    "output": Value("string"),
                    "metadata": Value("string"),
                }
            ),
        )

        remaining_data, pre_generated = self.checkpointer.load_existing_data(
            new_dataset
        )

        # Verify correct handling of overlapping and new data
        self.assertEqual(remaining_data.num_rows, 2)
        self.assertIsNotNone(pre_generated)
        self.assertEqual(pre_generated.num_rows, 3)
        self.assertEqual(
            set(remaining_data["instruction"]),
            {
                "Create a question about machine learning",
                "Write a question about databases",
            },
        )

    def test_save_frequency(self):
        # Test save frequency logic with different indices
        self.assertTrue(self.checkpointer.should_save_checkpoint(1))
        self.assertFalse(self.checkpointer.should_save_checkpoint(2))
        self.assertTrue(self.checkpointer.should_save_checkpoint(3))

        # Verify behavior when checkpointing is disabled
        disabled_checkpointer = Checkpointer(checkpoint_dir=None, save_freq=2)
        self.assertFalse(disabled_checkpointer.should_save_checkpoint(1))

    def test_missing_data_identification(self):
        # Test identification of missing data between datasets
        generated_data = Dataset.from_dict(
            {
                "instruction": [
                    "Generate a question about Python programming",
                    "Create a question about data structures",
                ],
                "input": ["Python basics", "Data structures in Python"],
                "output": [
                    "What is the difference between a list and a tuple in Python?",
                    "How does a binary search tree work?",
                ],
                "metadata": [
                    json.dumps({"difficulty": "beginner", "topic": "python"}),
                    json.dumps(
                        {"difficulty": "intermediate", "topic": "data_structures"}
                    ),
                ],
            },
            features=Features(
                {
                    "instruction": Value("string"),
                    "input": Value("string"),
                    "output": Value("string"),
                    "metadata": Value("string"),
                }
            ),
        )

        missing_data = self.checkpointer._get_missing_data(
            self.test_dataset, generated_data
        )

        # Verify correct identification of missing data
        self.assertEqual(missing_data.num_rows, 1)
        self.assertEqual(
            missing_data["instruction"][0], "Write a question about algorithms"
        )
        self.assertEqual(missing_data["input"][0], "Sorting algorithms")
        self.assertEqual(
            missing_data["output"][0], "Explain the time complexity of quicksort"
        )
        self.assertEqual(
            json.loads(missing_data["metadata"][0]),
            {"difficulty": "advanced", "topic": "algorithms"},
        )

    @patch("sdg_hub.checkpointer.logger")
    def test_error_handling(self, mock_logger):
        # Test behavior with invalid checkpoint directory
        invalid_checkpointer = Checkpointer(checkpoint_dir="/invalid/path", save_freq=1)
        remaining_data, pre_generated = invalid_checkpointer.load_existing_data(
            self.test_dataset
        )

        self.assertEqual(remaining_data.num_rows, self.test_dataset.num_rows)
        self.assertIsNone(pre_generated)
        mock_logger.info.assert_called()


if __name__ == "__main__":
    unittest.main()
