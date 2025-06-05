# SPDX-License-Identifier: Apache-2.0

"""Module for formatting and standardizing JSON output from various text analysis results."""

# Standard
import json
from typing import Dict, List, Optional, Any

# Third Party
from datasets import Dataset
import yaml

# First Party
from sdg_hub.blocks import Block, BlockRegistry


@BlockRegistry.register("JSONFormat")
class JSONFormat(Block):
    """A block for formatting and standardizing JSON output from text analysis results.

    This block processes various text analysis outputs (summary, keywords, named entities,
    sentiment) and formats them into a standardized JSON structure.
    """

    def __init__(self, block_name: str, output_column: str) -> None:
        super().__init__(block_name)
        self.output_column = output_column

    @staticmethod
    def _parse_named_entities(raw_text: str) -> Dict[str, Optional[List[str]]]:
        """Parse named entities from YAML-formatted text.

        Parameters
        ----------
        raw_text : str
            YAML-formatted text containing named entities.

        Returns
        -------
        Dict[str, Optional[List[str]]]
            Dictionary containing parsed named entities for organizations, people,
            locations, and dates. Returns None for each category if parsing fails.
        """
        try:
            parsed = yaml.safe_load(raw_text)
            return {
                "organizations": parsed.get("organizations", [])
                if isinstance(parsed, dict)
                else [],
                "people": parsed.get("people", []) if isinstance(parsed, dict) else [],
                "locations": parsed.get("locations", [])
                if isinstance(parsed, dict)
                else [],
                "dates": parsed.get("dates", []) if isinstance(parsed, dict) else [],
            }
        except Exception:
            return {
                "organizations": None,
                "people": None,
                "locations": None,
                "dates": None,
            }

    @staticmethod
    def _map_format_json(
        samples: Dataset, output_column: str, num_proc: int = 1
    ) -> Dataset:
        """Map JSON formatting function over the dataset samples.

        Parameters
        ----------
        samples : Dataset
            The input dataset containing text analysis results.
        output_column : str
            The name of the column where formatted JSON will be stored.
        num_proc : int, optional
            Number of processes to use for parallel processing, by default 1.

        Returns
        -------
        Dataset
            The dataset with added formatted JSON in the output column.
        """

        def format_json(sample: Dict[str, Any]) -> Dict[str, Any]:
            json_output = {
                "summary": sample.get("summary", None),
                "keywords": None,
                "named_entities": {
                    "organizations": None,
                    "people": None,
                    "locations": None,
                    "dates": None,
                },
                "sentiment": sample.get("sentiment", None),
            }

            try:
                if isinstance(sample.get("keywords"), str):
                    json_output["keywords"] = [
                        kw.strip() for kw in sample["keywords"].split(",") if kw.strip()
                    ]
            except Exception:
                json_output["keywords"] = None

            try:
                if isinstance(sample.get("named_entities"), str):
                    json_output["named_entities"] = JSONFormat._parse_named_entities(
                        sample["named_entities"]
                    )
            except Exception:
                json_output["named_entities"] = None

            sample[output_column] = json.dumps(json_output)
            return sample

        return samples.map(format_json, num_proc=num_proc)

    def generate(self, samples: Dataset) -> Dataset:
        """Generate formatted JSON from text analysis results in the dataset.

        Parameters
        ----------
        samples : Dataset
            The input dataset containing text analysis results.

        Returns
        -------
        Dataset
            The dataset with added formatted JSON in the output column.
        """
        samples = self._map_format_json(samples, self.output_column)
        return samples
