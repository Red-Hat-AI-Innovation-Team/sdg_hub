# SPDX-License-Identifier: Apache-2.0
"""Evaluation blocks for SDG Hub."""

from .evaluate_faithfulness_block import EvaluateFaithfulnessBlock
from .evaluate_relevancy_block import EvaluateRelevancyBlock

__all__ = ["EvaluateFaithfulnessBlock", "EvaluateRelevancyBlock"]
