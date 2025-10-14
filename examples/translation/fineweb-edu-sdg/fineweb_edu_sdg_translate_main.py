# Third Party
from datasets import load_dataset
from dotenv import load_dotenv
from datasets import Dataset

# First Party
from sdg_hub import Flow
import os

# Load environment variables from .env file
load_dotenv()


# load documents for generating qa-pairs
def load_data():
    hf_path = os.getenv("HF_DATA_PATH", "HuggingFaceFW/fineweb-edu")
    subset_name = os.getenv("FINEWEB_SUBSET_NAME", "CC-MAIN-2024-10")
    num_documents = int(os.getenv("NUM_DOCUMENTS", 10))

    print(f"Loading documents from: {hf_path}")

    fw_stream = load_dataset(hf_path, name=subset_name, split="train", streaming=True)

    from itertools import islice

    # limit to num_documents documents
    fw = list(islice(fw_stream, num_documents))

    dataset = []
    for each_document in fw:
        dataset.append({"document": each_document["text"]})

    hf_dataset = Dataset.from_list(dataset)
    return hf_dataset


# Setup model configuration in flow object
def set_model_config(flow_object):
    model_provider = os.getenv("MODEL_PROVIDER", "hosted_vllm")
    print(f"Using model provider: {model_provider}")
    # Set model provider
    if model_provider == "hosted_vllm":
        vllm_model = os.getenv(
            "VLLM_MODEL", "hosted_vllm/meta-llama/Llama-3.3-70B-Instruct"
        )
        vllm_api_base = os.getenv("VLLM_API_BASE", "http://localhost:8000/v1")
        vllm_api_key = os.getenv("VLLM_API_KEY", "EMPTY")

        flow_object.set_model_config(
            model=vllm_model,
            api_base=vllm_api_base,
            api_key=vllm_api_key,
        )
    elif model_provider == "openai":
        openai_api_key = os.getenv("OPENAI_API_KEY")
        openai_model = os.getenv("OPENAI_MODEL", "openai/gpt-4")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY is required when MODEL_PROVIDER=openai.")
        flow_object.set_model_config(
            model=openai_model,
            api_key=openai_api_key,
        )
    elif model_provider == "ollama":
        ollama_model = os.getenv("OLLAMA_MODEL", "ollama/gemma2")
        ollama_api_base = os.getenv("OLLAMA_API_BASE", "http://localhost:11434")
        flow_object.set_model_config(
            model=ollama_model,
            api_base=ollama_api_base,
        )
    elif model_provider == "maas":
        maas_model = os.getenv("MAAS_MODEL")
        maas_api_base = os.getenv("MAAS_API_BASE")
        maas_api_key = os.getenv("MAAS_API_KEY")
        if not (maas_model and maas_api_base and maas_api_key):
            raise ValueError(
                "MAAS_MODEL, MAAS_API_BASE, and MAAS_API_KEY are required when MODEL_PROVIDER=maas."
            )
        flow_object.set_model_config(
            model=maas_model,
            api_base=maas_api_base,
            api_key=maas_api_key,
        )
    return flow_object


if __name__ == "__main__":
    documents = load_data()

    # Load the flow
    flow_path = os.getenv("FLOW_PATH")
    if not flow_path:
        raise ValueError(
            "FLOW_PATH is required. Set it in .env before running the notebook."
        )
    flow = Flow.from_yaml(flow_path)

    # Get all recommended Models
    recommendations = flow.get_model_recommendations()
    print(f"Compatible models: {recommendations['compatible']}")
    print(f"Experimental models: {recommendations['experimental']}")

    # Print the default model
    default_model = flow.get_default_model()
    print(f"Default model: {default_model}")

    # Get runtime parameters
    save_data_path = os.getenv("OUTPUT_DATA_FOLDER", "")

    # Set model configuration
    flow = set_model_config(flow)

    # Generate qna pairs data
    translated_data = flow.generate(documents)

    out_dir = os.path.join(save_data_path, "qna_pairs_translated")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "qna.jsonl")

    translated_data.to_json(
        out_path,
        orient="records",
        lines=True,
    )

    print(f"✓ SDG summary: {len(translated_data)} records")

    print(f"✓ Columns: {list(translated_data.column_names)}")
