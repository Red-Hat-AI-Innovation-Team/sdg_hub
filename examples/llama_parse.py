import asyncio
from pathlib import Path
import nest_asyncio

nest_asyncio.apply()

from llama_parse import LlamaParse

parser = LlamaParse(
    api_key="your-key",  # can also be set in your env as LLAMA_CLOUD_API_KEY
    result_type="markdown",  # "markdown" and "text" are available
    num_workers=9,  # if multiple files passed, split in `num_workers` API calls
    verbose=True,
    language="en",  # Optionally you can define a language, default=en
    use_vendor_multimodal_model=True,
    vendor_multimodal_model_name="openai-gpt4o",
    vendor_multimodal_api_key='your-key',
)

if __name__ == "__main__":
    boa_docs_dir = Path.cwd() / "boa_docs"
    files = [str(f)for f in boa_docs_dir.iterdir() if f.suffix.lower() == '.pdf']
    # files = [{'file':f} for f in files]
    async def parse_file(file_path):
        doc_list = await parser.aload_data(file_path)
        doc_text = "\n---\n".join(page.text for page in doc_list)
        output_path = Path("boa_docs_llama_parse") / f"{Path(file_path).stem}.md"
        output_path.parent.mkdir(exist_ok=True)
        output_path.write_text(doc_text)
        return doc_list
        
    async def run_parser():
        tasks = [parse_file(f) for f in files]
        return await asyncio.gather(*tasks)
    
    documents = asyncio.run(run_parser())

    from IPython import embed; embed()
