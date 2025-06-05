# SPDX-License-Identifier: Apache-2.0

"""Module for parsing PDF documents into markdown format using Docling."""

# Standard
from typing import Any, Dict

# Third Party
from datasets import Dataset
from docling.document_converter import DocumentConverter

# First Party
from sdg_hub.blocks import Block, BlockRegistry


@BlockRegistry.register("DoclingParsePDF")
class DoclingParsePDF(Block):
    """A block for parsing PDF documents using Docling.

    This block takes a dataset containing PDF file paths and converts them to markdown format
    using the Docling document converter.
    """

    def __init__(
        self, block_name: str, pdf_path_column: str, output_column: str
    ) -> None:
        super().__init__(block_name)
        self.pdf_path_column = pdf_path_column
        self.output_column = output_column
        self.converter = DocumentConverter()

    @staticmethod
    def _map_parse_pdf(
        samples: Dataset,
        pdf_path_column: str,
        output_column: str,
        converter: DocumentConverter,
        num_proc: int = 1,
    ) -> Dataset:
        """Map PDF parsing function over the dataset samples.

        Parameters
        ----------
        samples : Dataset
            The input dataset containing PDF file paths.
        pdf_path_column : str
            The name of the column containing PDF file paths.
        output_column : str
            The name of the column where markdown output will be stored.
        converter : DocumentConverter
            The Docling document converter instance.
        num_proc : int, optional
            Number of processes to use for parallel processing, by default 1.

        Returns
        -------
        Dataset
            The dataset with added markdown content in the output column.

        Notes
        -----
        This method processes each sample in the dataset by converting the PDF
        specified in the pdf_path_column to markdown format and storing the result
        in the output_column.
        """

        def parse_pdf(sample: Dict[str, Any]) -> Dict[str, Any]:
            pdf_path = sample[pdf_path_column]
            result = converter.convert(pdf_path)
            sample[output_column] = result.document.export_to_markdown()
            return sample

        return samples.map(parse_pdf, num_proc=num_proc)

    def generate(self, samples: Dataset) -> Dataset:
        """Generate markdown content from PDF files in the dataset.

        Parameters
        ----------
        samples : Dataset
            The input dataset containing PDF file paths.

        Returns
        -------
        Dataset
            The dataset with added markdown content in the output column.
        """
        samples = self._map_parse_pdf(
            samples, self.pdf_path_column, self.output_column, self.converter
        )
        return samples
