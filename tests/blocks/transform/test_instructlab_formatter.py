# SPDX-License-Identifier: Apache-2.0
"""Tests for InstructLabFormatterBlock."""

# Standard
# Third Party
import pandas as pd
import pytest
import yaml

# First Party
from sdg_hub.core.blocks.transform import InstructLabFormatterBlock


@pytest.fixture
def qa_dataset():
    """Create a sample Q&A dataset with two taxonomy groups."""
    return pd.DataFrame(
        {
            "question": [
                "What is photosynthesis?",
                "How do plants convert light to energy?",
                "What role does chlorophyll play?",
                "Why is photosynthesis important?",
                "What are the products of photosynthesis?",
                "What is a binary star system?",
                "How do binary stars orbit?",
                "What types of binary stars exist?",
                "How are binary stars detected?",
                "Why are binary stars important?",
            ],
            "answer": [
                "Photosynthesis is the process by which plants convert sunlight into chemical energy.",
                "Plants use chloroplasts containing chlorophyll to capture light energy and convert CO2 and water into glucose.",
                "Chlorophyll is the green pigment that absorbs light energy, primarily in the blue and red wavelengths.",
                "Photosynthesis produces oxygen and glucose, forming the base of most food chains on Earth.",
                "The main products are glucose (C6H12O6) and oxygen (O2), with water as a byproduct.",
                "A binary star system consists of two stars orbiting around their common center of mass.",
                "Binary stars orbit their barycenter, with each star tracing an elliptical path.",
                "Binary stars include visual, spectroscopic, eclipsing, and astrometric binaries.",
                "Binary stars are detected through visual observation, spectral line shifts, or brightness variations.",
                "Binary stars allow astronomers to directly measure stellar masses and test stellar evolution models.",
            ],
            "document_text": [
                "Photosynthesis is a biological process..."
            ]
            * 5
            + [
                "Binary star systems are common in the universe..."
            ]
            * 5,
            "domain": ["biology"] * 5 + ["astronomy"] * 5,
            "taxonomy_path": ["knowledge/science/biology"] * 5
            + ["knowledge/science/astronomy"] * 5,
        }
    )


@pytest.fixture
def small_dataset():
    """Create a dataset with too few examples for one group."""
    return pd.DataFrame(
        {
            "question": ["Q1", "Q2", "Q3"],
            "answer": ["A1", "A2", "A3"],
            "document_text": ["doc"] * 3,
            "domain": ["test"] * 3,
            "taxonomy_path": ["knowledge/test"] * 3,
        }
    )


def test_basic_formatting(qa_dataset):
    """Test that formatter produces valid qna.yaml for each taxonomy group."""
    block = InstructLabFormatterBlock(
        block_name="test_formatter",
        input_cols=["question", "answer", "document_text", "domain", "taxonomy_path"],
        output_cols=["qna_yaml", "attribution_txt", "taxonomy_path", "num_examples"],
    )

    result = block.generate(qa_dataset)

    assert len(result) == 2
    assert "qna_yaml" in result.columns
    assert "attribution_txt" in result.columns
    assert "taxonomy_path" in result.columns
    assert "num_examples" in result.columns


def test_yaml_structure(qa_dataset):
    """Test that generated YAML follows InstructLab schema."""
    block = InstructLabFormatterBlock(
        block_name="test_formatter",
        input_cols=["question", "answer", "document_text", "domain", "taxonomy_path"],
        output_cols=["qna_yaml", "attribution_txt", "taxonomy_path", "num_examples"],
    )

    result = block.generate(qa_dataset)

    for _, row in result.iterrows():
        parsed = yaml.safe_load(row["qna_yaml"])
        assert parsed["version"] == 3
        assert "domain" in parsed
        assert "task_description" in parsed
        assert "created_by" in parsed
        assert "seed_examples" in parsed
        assert len(parsed["seed_examples"]) == 5

        for example in parsed["seed_examples"]:
            assert "question" in example
            assert "answer" in example
            assert "context" in example


def test_custom_created_by(qa_dataset):
    """Test that created_by field is configurable."""
    block = InstructLabFormatterBlock(
        block_name="test_formatter",
        input_cols=["question", "answer", "document_text", "domain", "taxonomy_path"],
        output_cols=["qna_yaml", "attribution_txt", "taxonomy_path", "num_examples"],
        created_by="test-user",
    )

    result = block.generate(qa_dataset)
    parsed = yaml.safe_load(result.iloc[0]["qna_yaml"])
    assert parsed["created_by"] == "test-user"


def test_min_examples_filtering(small_dataset):
    """Test that groups below min_examples are dropped."""
    block = InstructLabFormatterBlock(
        block_name="test_formatter",
        input_cols=["question", "answer", "document_text", "domain", "taxonomy_path"],
        output_cols=["qna_yaml", "attribution_txt", "taxonomy_path", "num_examples"],
        min_examples=5,
    )

    result = block.generate(small_dataset)
    assert len(result) == 0


def test_min_examples_override(small_dataset):
    """Test that min_examples can be lowered."""
    block = InstructLabFormatterBlock(
        block_name="test_formatter",
        input_cols=["question", "answer", "document_text", "domain", "taxonomy_path"],
        output_cols=["qna_yaml", "attribution_txt", "taxonomy_path", "num_examples"],
        min_examples=2,
    )

    result = block.generate(small_dataset)
    assert len(result) == 1
    assert result.iloc[0]["num_examples"] == 3


def test_attribution_txt(qa_dataset):
    """Test that attribution.txt is generated correctly."""
    block = InstructLabFormatterBlock(
        block_name="test_formatter",
        input_cols=["question", "answer", "document_text", "domain", "taxonomy_path"],
        output_cols=["qna_yaml", "attribution_txt", "taxonomy_path", "num_examples"],
        created_by="contributor",
    )

    result = block.generate(qa_dataset)
    attr = result.iloc[0]["attribution_txt"]
    assert "contributor" in attr
    assert "Apache-2.0" in attr


def test_column_mapping(qa_dataset):
    """Test that input_cols dict mapping works."""
    renamed = qa_dataset.rename(columns={"question": "q", "answer": "a"})
    block = InstructLabFormatterBlock(
        block_name="test_formatter",
        input_cols={
            "question": "q",
            "answer": "a",
            "document_text": "document_text",
            "domain": "domain",
            "taxonomy_path": "taxonomy_path",
        },
        output_cols=["qna_yaml", "attribution_txt", "taxonomy_path", "num_examples"],
    )

    result = block.generate(renamed)
    assert len(result) == 2
    parsed = yaml.safe_load(result.iloc[0]["qna_yaml"])
    assert len(parsed["seed_examples"]) == 5


def test_num_examples_count(qa_dataset):
    """Test that num_examples accurately reflects group size."""
    block = InstructLabFormatterBlock(
        block_name="test_formatter",
        input_cols=["question", "answer", "document_text", "domain", "taxonomy_path"],
        output_cols=["qna_yaml", "attribution_txt", "taxonomy_path", "num_examples"],
    )

    result = block.generate(qa_dataset)
    for _, row in result.iterrows():
        assert row["num_examples"] == 5
