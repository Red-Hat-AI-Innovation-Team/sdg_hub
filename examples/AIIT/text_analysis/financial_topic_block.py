# SPDX-License-Identifier: Apache-2.0
"""Financial Topic Classification Block - Custom Block Example.

This module demonstrates how to create a custom transform block that extends
the structured insights flow with domain-specific analysis. It classifies
financial news articles into predefined topic categories.

This is an example of extending SDG Hub flows with custom business logic.
"""

# Standard
from typing import Any, Dict, Tuple
import re

# Third Party
from datasets import Dataset
from pydantic import Field

# First Party
from sdg_hub.core.blocks.base import BaseBlock
from sdg_hub.core.blocks.registry import BlockRegistry

# Setup logger (fallback if utils not available)
try:
    # First Party
    from sdg_hub.utils.logger_config import setup_logger

    logger = setup_logger(__name__)
except ImportError:
    # Standard
    import logging

    logger = logging.getLogger(__name__)


@BlockRegistry.register(
    "FinancialTopicBlock",
    "transform",
    "Classifies financial news articles into topic categories using keyword matching",
)
class FinancialTopicBlock(BaseBlock):
    """Custom block for financial topic classification.

    This block demonstrates how to create domain-specific analysis blocks
    that can be integrated into existing SDG Hub flows. It uses rule-based
    keyword matching to classify financial news into topic categories.

    Categories:
    - Markets: Trading, stocks, indexes, market movements
    - Corporate: Company news, earnings, acquisitions, leadership
    - Economy: Economic indicators, policy, inflation, employment
    - Technology: Fintech, digital payments, blockchain, AI in finance
    - Regulation: Regulatory changes, compliance, legal issues
    - Crisis: Financial crises, market crashes, emergency responses
    - Other: Articles that don't fit other categories

    Attributes
    ----------
    input_cols : List[str]
        Input columns containing text to classify. Expects 'text' column.
    output_cols : List[str]
        Output columns for classification results. Must specify exactly one.
    confidence_threshold : float
        Minimum confidence score to assign a topic (default: 0.3).
    use_headlines : bool
        Whether to also consider headlines in classification (default: True).
    """

    confidence_threshold: float = Field(
        default=0.3,
        description="Minimum confidence score to assign a topic",
        ge=0.0,
        le=1.0,
    )
    use_headlines: bool = Field(
        default=True, description="Whether to consider headlines in classification"
    )

    def validate_output_cols(self):
        """Validate that exactly one output column is specified."""
        if not self.output_cols or len(self.output_cols) != 1:
            raise ValueError("FinancialTopicBlock requires exactly one output column")
        return self.output_cols

    # Topic keyword definitions
    TOPIC_KEYWORDS = {
        "Markets": {
            "primary": [
                "stock",
                "trading",
                "market",
                "index",
                "dow",
                "s&p",
                "nasdaq",
                "exchange",
                "shares",
                "equity",
            ],
            "secondary": [
                "bull",
                "bear",
                "rally",
                "decline",
                "volatility",
                "volume",
                "price",
                "gain",
                "loss",
                "futures",
            ],
            "weight": 1.0,
        },
        "Corporate": {
            "primary": [
                "company",
                "corporation",
                "ceo",
                "earnings",
                "revenue",
                "profit",
                "acquisition",
                "merger",
            ],
            "secondary": [
                "executive",
                "board",
                "shareholder",
                "quarterly",
                "annual",
                "guidance",
                "outlook",
                "deal",
            ],
            "weight": 1.0,
        },
        "Economy": {
            "primary": [
                "economy",
                "economic",
                "gdp",
                "inflation",
                "employment",
                "unemployment",
                "fed",
                "federal reserve",
            ],
            "secondary": [
                "growth",
                "recession",
                "recovery",
                "policy",
                "monetary",
                "fiscal",
                "interest rate",
                "jobs",
            ],
            "weight": 1.0,
        },
        "Technology": {
            "primary": [
                "technology",
                "fintech",
                "digital",
                "blockchain",
                "cryptocurrency",
                "bitcoin",
                "ai",
                "artificial intelligence",
            ],
            "secondary": [
                "innovation",
                "platform",
                "software",
                "payment",
                "mobile",
                "online",
                "cyber",
                "automation",
            ],
            "weight": 1.0,
        },
        "Regulation": {
            "primary": [
                "regulation",
                "regulatory",
                "compliance",
                "sec",
                "cftc",
                "legal",
                "lawsuit",
                "court",
            ],
            "secondary": [
                "rule",
                "law",
                "enforcement",
                "fine",
                "penalty",
                "investigation",
                "audit",
                "oversight",
            ],
            "weight": 1.0,
        },
        "Crisis": {
            "primary": [
                "crisis",
                "crash",
                "collapse",
                "bailout",
                "emergency",
                "default",
                "bankruptcy",
            ],
            "secondary": [
                "panic",
                "turmoil",
                "instability",
                "rescue",
                "intervention",
                "liquidity",
                "systemic",
            ],
            "weight": 1.2,  # Crisis topics get higher weight due to importance
        },
    }

    def _extract_text_for_analysis(self, sample: Dict[str, Any]) -> str:
        """Extract text content for topic classification."""
        text_parts = []

        # Always include main text
        if "text" in sample:
            text_parts.append(sample["text"])

        # Optionally include headline
        if self.use_headlines and "Headline" in sample:
            # Give headline content extra weight by including it twice
            headline = sample["Headline"]
            text_parts.extend([headline, headline])

        return " ".join(text_parts).lower()

    def _calculate_topic_scores(self, text: str) -> Dict[str, float]:
        """Calculate confidence scores for each topic category."""
        word_count = len(text.split())
        if word_count == 0:
            return {}

        topic_scores = {}

        for topic, keywords_data in self.TOPIC_KEYWORDS.items():
            score = 0.0
            weight = keywords_data.get("weight", 1.0)

            # Score primary keywords (higher weight)
            for keyword in keywords_data["primary"]:
                matches = len(re.findall(r"\b" + re.escape(keyword) + r"\b", text))
                score += matches * 2.0  # Primary keywords worth more

            # Score secondary keywords
            for keyword in keywords_data["secondary"]:
                matches = len(re.findall(r"\b" + re.escape(keyword) + r"\b", text))
                score += matches * 1.0

            # Normalize by text length and apply topic weight
            normalized_score = (score / word_count) * weight * 100
            topic_scores[topic] = normalized_score

        return topic_scores

    def _classify_topic(self, text: str) -> Tuple[str, float]:
        """Classify text into a topic category."""
        topic_scores = self._calculate_topic_scores(text)

        if not topic_scores:
            return "Other", 0.0

        # Find the topic with highest score
        best_topic = max(topic_scores, key=topic_scores.get)
        best_score = topic_scores[best_topic]

        # Apply confidence threshold
        if best_score < self.confidence_threshold:
            return "Other", best_score

        return best_topic, best_score

    def generate(self, samples: Dataset, **kwargs: Any) -> Dataset:
        """Generate topic classifications for all samples.

        Parameters
        ----------
        samples : Dataset
            Input dataset containing text to classify.

        Returns
        -------
        Dataset
            Dataset with topic classification added to output column.
        """
        output_col = self.output_cols[0]

        def _classify_sample(sample):
            """Classify a single sample."""
            try:
                # Extract text for analysis
                text = self._extract_text_for_analysis(sample)

                # Classify the topic
                topic, confidence = self._classify_topic(text)

                # Store result
                sample[output_col] = topic

                # Optionally store confidence score
                confidence_col = f"{output_col}_confidence"
                sample[confidence_col] = round(confidence, 3)

                logger.debug(
                    f"Classified text as '{topic}' with confidence {confidence:.3f}"
                )

            except Exception as e:
                logger.error(f"Error classifying sample: {e}")
                sample[output_col] = "Other"
                sample[f"{output_col}_confidence"] = 0.0

            return sample

        # Apply classification to all samples
        result = samples.map(_classify_sample)

        # Log summary statistics
        if len(result) > 0:
            topics = [sample[output_col] for sample in result]
            topic_counts = {topic: topics.count(topic) for topic in set(topics)}
            logger.info(f"Topic classification complete. Distribution: {topic_counts}")

        return result
