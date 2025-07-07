import os
import sys
from tqdm import tqdm
from datasets import Dataset, load_dataset

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "..")))
from knowledge_utils import chunk_document

# output directory
output_dir = f"sdg_demo_output/"

chunk_size = 50
max_model_context_length = 2048

try:
    kannada_wiki = load_dataset("wikimedia/wikipedia", "20231101.kn")["train"]

    kannada_documents = []

    max_documents = 1000
    doc_count = 0
    for each_doc in tqdm(kannada_wiki, desc="For each document"):
        document = [
            {
                "document": chunk,
                "title": each_doc["title"],
            }
            for chunk in chunk_document(
                each_doc["text"],
                server_ctx_size=max_model_context_length,
                chunk_word_count=chunk_size,
            )
        ]
        kannada_documents.extend(document)
        doc_count += 1

        if doc_count >= max_documents:
            break

    kannada_doc_with_icl = []

    icl_context = """
    Shimoga, officially Shivamogga, is a city and the district headquarters of Shimoga district in the Karnataka state of India. The city lies on the banks of the Tunga River. Being the gateway for the hilly region of the Western Ghats, the city is popularly nicknamed the "Gateway of Malnad". The population of Shimoga city is 322,650 as per 2011 census. The city has been selected for the Smart Cities Mission ' standing in the fourth position in the state and 25th in the country as of November 2020.
    """

    for each_document in kannada_documents:
        icl_dict = {}
        icl_dict["icl_document"] = icl_context

        icl_dict["icl_query"] = "Shivamogga is a city in which country?"
        icl_dict["icl_response"] = "Shivamogga is a city in India."

        icl_dict["title"] = each_document["title"]
        icl_dict["text"] = each_document["document"]

        kannada_doc_with_icl.append(icl_dict)

    seed_data = Dataset.from_list(kannada_doc_with_icl)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    output_file = f"{output_dir}/seed_data.jsonl"

    seed_data.to_json(output_file, orient="records", lines=True, force_ascii=False)
except Exception as e:
    print(f"Failed to load Kannada Wikipedia dataset: {e}")
    print("Please ensure you have internet connectivity and the dataset is available.")
    exit(1)
