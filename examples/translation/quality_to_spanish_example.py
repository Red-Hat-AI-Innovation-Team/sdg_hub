"""
Example usage of the QuALITY to Spanish translation flow.

This script demonstrates how to:
1. Load the QuALITY dataset
2. Configure the translation flow with specific models
3. Run translation on a subset of documents
4. Inspect and save the results
"""

from datasets import load_dataset
from sdg_hub.core.flow import Flow

def main():
    # ==========================================
    # 1. Load QuALITY Dataset
    # ==========================================
    print("Loading QuALITY dataset...")
    dataset = load_dataset("zitongyang/entigraph-quality-corpus", split="train")

    # For testing, use a small subset (e.g., first 5 documents)
    # For production, remove or increase this limit
    dataset = dataset.select(range(5))

    print(f"Loaded {len(dataset)} documents for translation")
    print(f"Columns: {dataset.column_names}")

    # ==========================================
    # 2. Load Translation Flow
    # ==========================================
    print("\nLoading translation flow...")
    flow = Flow.from_registry("translation/quality_to_spanish")

    # ==========================================
    # 3. Configure Models (External Configuration)
    # ==========================================
    print("\nConfiguring translation models...")

    # Set translation model (GPT-4o recommended for cost/quality balance)
    translation_blocks = [
        "translate_document_llm",
        "translate_icl_document_llm",
        "translate_icl_query_1_llm",
        "translate_icl_query_2_llm",
        "translate_icl_query_3_llm",
        "translate_domain_llm"
    ]

    for block_name in translation_blocks:
        flow.set_model_config(
            block_name,
            model_name="gpt-4o-2024-11-20",
            # Optional: Add provider-specific parameters
            # api_key="your-api-key",  # If not using environment variables
        )

    # Set verification model (Claude 3.5 Sonnet for strong evaluation)
    flow.set_model_config(
        "verify_translation_llm",
        model_name="claude-3-5-sonnet-20241022",
        # Optional: Add provider-specific parameters
        # api_key="your-api-key",  # If not using environment variables
    )

    print("Models configured:")
    print(f"  Translation: gpt-4o-2024-11-20")
    print(f"  Verification: claude-3-5-sonnet-20241022")

    # ==========================================
    # 4. Run Translation Flow
    # ==========================================
    print("\nRunning translation flow...")
    print("This may take several minutes depending on the number of documents...")

    translated_dataset = flow.run(dataset)

    # ==========================================
    # 5. Inspect Results
    # ==========================================
    print("\n" + "="*60)
    print("TRANSLATION RESULTS")
    print("="*60)

    print(f"\nTotal documents processed: {len(translated_dataset)}")
    print(f"Output columns: {translated_dataset.column_names}")

    # Show summary statistics
    if "translation_quality_score" in translated_dataset.column_names:
        scores = [int(s) for s in translated_dataset["translation_quality_score"]]
        print(f"\nTranslation Quality Scores:")
        print(f"  Average: {sum(scores) / len(scores):.2f}")
        print(f"  Min: {min(scores)}")
        print(f"  Max: {max(scores)}")
        print(f"  Score distribution:")
        for score in range(1, 6):
            count = scores.count(score)
            print(f"    Score {score}: {count} ({count/len(scores)*100:.1f}%)")

    # Display first translation example
    print("\n" + "="*60)
    print("EXAMPLE TRANSLATION (First Document)")
    print("="*60)

    example = translated_dataset[0]

    print("\n[ENGLISH ORIGINAL (first 500 chars)]:")
    print(example["document"][:500] + "...")

    print("\n[SPANISH TRANSLATION (first 500 chars)]:")
    print(example["document_spanish"][:500] + "...")

    if "translation_quality_score" in example:
        print(f"\n[QUALITY SCORE]: {example['translation_quality_score']}/5")
        print(f"\n[QUALITY EXPLANATION]:")
        print(example["translation_quality_explanation"][:500] + "...")

    # ==========================================
    # 6. Save Results
    # ==========================================
    output_path = "quality_spanish_translated.jsonl"
    print(f"\n\nSaving translated dataset to {output_path}...")

    translated_dataset.to_json(output_path)

    print(f"✓ Translation complete! Results saved to {output_path}")

    # ==========================================
    # 7. Optional: Filter for High-Quality Only
    # ==========================================
    print("\n" + "="*60)
    print("FILTERING FOR HIGH-QUALITY TRANSLATIONS")
    print("="*60)

    if "translation_quality_score" in translated_dataset.column_names:
        # Note: The flow already includes a filter block for scores 4-5
        # But if you want to manually filter later, you can do:
        high_quality = translated_dataset.filter(
            lambda x: int(x["translation_quality_score"]) >= 4
        )

        print(f"\nHigh-quality translations (score ≥4): {len(high_quality)}/{len(translated_dataset)}")

        if len(high_quality) > 0:
            high_quality_path = "quality_spanish_high_quality.jsonl"
            high_quality.to_json(high_quality_path)
            print(f"✓ High-quality subset saved to {high_quality_path}")


if __name__ == "__main__":
    # Make sure to set environment variables for API keys:
    # export OPENAI_API_KEY="your-openai-key"
    # export ANTHROPIC_API_KEY="your-anthropic-key"

    main()
