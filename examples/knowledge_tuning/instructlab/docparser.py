# Standard
from pathlib import Path
from typing import Iterable
import json
import time

# Third Party
from docling.datamodel.base_models import ConversionStatus
from docling.datamodel.document import ConvertedDocument, DocumentConversionInput
from docling.document_converter import DocumentConverter
from logger_config import setup_logger
import click

logger = setup_logger(__name__)


def export_documents(
    converted_docs: Iterable[ConvertedDocument],
    output_dir: Path,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    success_count = 0
    failure_count = 0

    for doc in converted_docs:
        if doc.status == ConversionStatus.SUCCESS:
            success_count += 1
            doc_filename = doc.input.file.stem

            # Export Deep Search document JSON format:
            with (output_dir / f"{doc_filename}.json").open("w") as fp:
                fp.write(json.dumps(doc.render_as_dict()))

            # Export Markdown format:
            with (output_dir / f"{doc_filename}.md").open("w") as fp:
                fp.write(doc.render_as_markdown())
        else:
            logger.info(f"Document {doc.input.file} failed to convert.")
            failure_count += 1

    logger.info(
        f"Processed {success_count + failure_count} docs, of which {failure_count} failed"
    )


@click.command()
@click.option(
    "--input-dir",
    "-i",
    type=click.Path(path_type=Path),
    help="Directory containing the documents to convert",
    required=True,
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(path_type=Path),
    help="Directory to save the converted documents",
    required=True,
)
def main(input_dir: Path, output_dir: Path):
    file_paths = list(input_dir.glob("*.pdf"))
    artifacts_path = DocumentConverter.download_models_hf()
    doc_converter = DocumentConverter(artifacts_path=artifacts_path)
    inputs = DocumentConversionInput.from_paths(file_paths)

    start_time = time.time()
    converted_docs = doc_converter.convert(inputs)
    export_documents(converted_docs, output_dir)
    end_time = time.time()

    logger.info(f"Parsing documents took {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
