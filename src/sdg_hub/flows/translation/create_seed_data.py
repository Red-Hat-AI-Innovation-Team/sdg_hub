import os
import sys

from datasets import Dataset, load_dataset
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "..")))
from knowledge_utils import chunk_document

# output directory
output_dir = "sdg_demo_output/"

chunk_size = 50
max_model_context_length = 2048

# ============================================================================
# CUSTOM DOMAIN CONFIGURATION
# ============================================================================
# Add your own custom domains and keywords here.
# Format: {'domain_name': ['keyword1', 'keyword2', ...]}
# Keywords can be in any language (Kannada, English, etc.)

CUSTOM_DOMAINS = {
    # Example custom domains (uncomment and modify as needed):
    # 'economics': [
    #     'ಆರ್ಥಿಕತೆ', 'ಬ್ಯಾಂಕ್', 'ವ್ಯಾಪಾರ', 'ಉದ್ಯಮ', 'ಹಣಕಾಸು',
    #     'economics', 'bank', 'business', 'industry', 'finance', 'market', 'trade'
    # ],
    # 'education': [
    #     'ಶಿಕ್ಷಣ', 'ಶಾಲೆ', 'ಕಾಲೇಜು', 'ವಿಶ್ವವಿದ್ಯಾಲಯ', 'ಅಧ್ಯಯನ',
    #     'education', 'school', 'college', 'university', 'study', 'learning'
    # ],
    # 'health': [
    #     'ಆರೋಗ್ಯ', 'ಆಸ್ಪತ್ರೆ', 'ವೈದ್ಯ', 'ಔಷಧ', 'ಚಿಕಿತ್ಸೆ',
    #     'health', 'hospital', 'doctor', 'medicine', 'treatment', 'medical'
    # ]
}

# To add your own domains:
# 1. Uncomment the examples above or add new entries
# 2. Use both source language and English keywords for better coverage
# 3. Run the script to see your custom domains in the output


def classify_domain(title, text, custom_domains=None):
    """
    Classify the domain of a Wikipedia article based on title and content.

    Args:
        title (str): Article title
        text (str): Article content
        custom_domains (dict, optional): Custom domain keywords in format:
            {'domain_name': ['keyword1', 'keyword2', ...]}

    Returns:
        str: Domain classification (geography, history, science, culture, politics, etc.)
    """
    title_lower = title.lower()
    text_lower = text.lower()

    # Default domain keywords
    default_domains = {
        "geography": [
            "ನಗರ",
            "ಜಿಲ್ಲೆ",
            "ರಾಜ್ಯ",
            "ದೇಶ",
            "ಪರ್ವತ",
            "ನದಿ",
            "ಸಮುದ್ರ",
            "ಕರಾವಳಿ",
            "city",
            "district",
            "state",
            "country",
            "mountain",
            "river",
            "sea",
            "coast",
            "ಗ್ರಾಮ",
            "ತಾಲೂಕು",
            "ಪ್ರಾಂತ್ಯ",
            "ಭೂಗೋಳ",
        ],
        "history": [
            "ಇತಿಹಾಸ",
            "ರಾಜ",
            "ರಾಣಿ",
            "ಸಾಮ್ರಾಜ್ಯ",
            "ಯುದ್ಧ",
            "ಕಾಲ",
            "ಶತಮಾನ",
            "history",
            "king",
            "queen",
            "empire",
            "war",
            "century",
            "ancient",
            "medieval",
            "ಪ್ರಾಚೀನ",
            "ಮಧ್ಯಕಾಲೀನ",
            "ಆಧುನಿಕ",
        ],
        "science": [
            "ವಿಜ್ಞಾನ",
            "ತಂತ್ರಜ್ಞಾನ",
            "ಗಣಿತ",
            "ಭೌತಶಾಸ್ತ್ರ",
            "ರಸಾಯನಶಾಸ್ತ್ರ",
            "ಜೀವಶಾಸ್ತ್ರ",
            "science",
            "technology",
            "mathematics",
            "physics",
            "chemistry",
            "biology",
            "ಸಂಶೋಧನೆ",
            "ಆವಿಷ್ಕಾರ",
            "ಇಂಜಿನಿಯರಿಂಗ್",
        ],
        "culture": [
            "ಸಂಸ್ಕೃತಿ",
            "ಕಲೆ",
            "ಸಾಹಿತ್ಯ",
            "ಸಂಗೀತ",
            "ನೃತ್ಯ",
            "ಚಿತ್ರಕಲೆ",
            "ಶಿಲ್ಪ",
            "culture",
            "art",
            "literature",
            "music",
            "dance",
            "painting",
            "sculpture",
            "ಭಾಷೆ",
            "ಸಂಪ್ರದಾಯ",
            "ಹಬ್ಬ",
            "ಧರ್ಮ",
            "ದೇವಾಲಯ",
        ],
        "politics": [
            "ರಾಜಕೀಯ",
            "ಸರ್ಕಾರ",
            "ಚುನಾವಣೆ",
            "ಸಂಸತ್ತು",
            "ಮುಖ್ಯಮಂತ್ರಿ",
            "ಪ್ರಧಾನಿ",
            "politics",
            "government",
            "election",
            "parliament",
            "minister",
            "policy",
            "ಪಕ್ಷ",
            "ನೀತಿ",
            "ಆಡಳಿತ",
        ],
        "sports": [
            "ಕ್ರೀಡೆ",
            "ಆಟ",
            "ಪಂದ್ಯ",
            "ಟೂರ್ನಮೆಂಟ್",
            "ಚಾಂಪಿಯನ್",
            "ಕ್ರಿಕೆಟ್",
            "ಫುಟ್ಬಾಲ್",
            "sports",
            "game",
            "match",
            "tournament",
            "champion",
            "cricket",
            "football",
            "ಒಲಿಂಪಿಕ್ಸ್",
            "ಪದಕ",
            "ಆಟಗಾರ",
        ],
    }

    # Merge custom domains with default domains
    all_domains = default_domains.copy()
    if custom_domains:
        all_domains.update(custom_domains)
        print(f"Added custom domains: {list(custom_domains.keys())}")

    # Count keyword matches for all domains
    domain_scores = {}
    for domain_name, keywords in all_domains.items():
        score = sum(1 for kw in keywords if kw in title_lower or kw in text_lower)
        domain_scores[domain_name] = score

    # Return domain with highest score, default to 'general'
    best_domain = max(domain_scores, key=domain_scores.get)
    return best_domain if domain_scores[best_domain] > 0 else "general"


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

        # Add domain classification (with custom domains if defined)
        icl_dict["domain"] = classify_domain(
            each_document["title"],
            each_document["document"],
            custom_domains=CUSTOM_DOMAINS if CUSTOM_DOMAINS else None,
        )

        kannada_doc_with_icl.append(icl_dict)

    seed_data = Dataset.from_list(kannada_doc_with_icl)

    # Print domain distribution
    from collections import Counter

    domain_counts = Counter([doc["domain"] for doc in kannada_doc_with_icl])
    print("\nDomain distribution:")
    for domain, count in domain_counts.most_common():
        print(f"  {domain}: {count} documents")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    output_file = f"{output_dir}/seed_data.jsonl"

    seed_data.to_json(output_file, orient="records", lines=True, force_ascii=False)
    print(
        f"\nSaved {len(kannada_doc_with_icl)} documents with domain classification to {output_file}"
    )
except Exception as e:
    print(f"Failed to load Kannada Wikipedia dataset: {e}")
    print("Please ensure you have internet connectivity and the dataset is available.")
    exit(1)
