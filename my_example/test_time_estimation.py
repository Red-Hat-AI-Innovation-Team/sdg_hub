#!/usr/bin/env python3
"""
Test script for SDG Hub Time Estimation Feature
Runs 1 and 5 sample dry runs to predict execution time for 200 samples
"""

# Standard
import sys

# Add src to path for imports
sys.path.insert(0, "/Users/ashtarkb/Desktop/sdg_hub_latest/sdg_hub/src")

# Standard

# Third Party
from datasets import load_dataset

# First Party
from sdg_hub import Flow
from sdg_hub.core.utils.time_estimator import estimate_execution_time

# Configure paths
FLOW_YAML = "/Users/ashtarkb/Desktop/sdg_hub_latest/sdg_hub/src/sdg_hub/flows/qa_generation/document_grounded_qa/multi_summary_qa/instructlab/flow.yaml"
DATA_PATH = (
    "/Users/ashtarkb/Desktop/sdg_hub_latest/sdg_hub/my_example/ibm_seed_data.jsonl"
)


def main():
    print("=" * 80)
    print("🚀 Time Estimation: 1 & 5 Sample Dry Runs → Predict 200 Samples")
    print("=" * 80)
    print()

    # Load the flow
    print("📋 Loading flow...")
    flow = Flow.from_yaml(FLOW_YAML)
    print(f"✅ Loaded flow: {flow.metadata.name}")
    print()

    # Configure model
    print("🔧 Configuring model...")
    flow.set_model_config(
        model="hosted_vllm/mixtral-8x7b-instruct",
        api_base="http://localhost:8080/v1",
    )
    print("✅ Model configured: hosted_vllm/mixtral-8x7b-instruct")
    print()

    # Load dataset
    print("📊 Loading dataset...")
    ds = load_dataset("json", data_files=DATA_PATH, split="train")

    # Add domain column if not present
    if "domain" not in ds.column_names:
        ds = ds.add_column("domain", ["technology"] * len(ds))

    # Remove duplicates based on document column
    unique_column = "document"
    df = ds.to_pandas()
    unique_indices = df[unique_column].drop_duplicates().index.tolist()
    unique_ds = ds.select(unique_indices)

    print(f"✅ Dataset loaded: {len(unique_ds)} unique documents")
    print()

    # Step 1: Dry run with 1 sample
    print("=" * 80)
    print("🧪 DRY RUN 1: Running with 1 Sample")
    print("=" * 80)

    print("Running dry run with 1 sample...")
    dry_result_1 = flow.dry_run(dataset=unique_ds, sample_size=1)

    print(f"✅ Completed in {dry_result_1['execution_time_seconds']:.2f} seconds")
    print("\n📊 Block Details:")
    for block in dry_result_1.get("blocks_executed", []):
        if "LLM" in block.get("block_type", "") or "Evaluate" in block.get(
            "block_type", ""
        ):
            print(
                f"   - {block['block_name']}: {block.get('execution_time_seconds', 0):.2f}s, {block.get('input_rows', 0)} requests"
            )
    print()

    # Step 2: Dry run with 5 samples
    print("=" * 80)
    print("🧪 DRY RUN 2: Running with 5 Samples")
    print("=" * 80)

    print("Running dry run with 5 samples...")
    dry_result_5 = flow.dry_run(dataset=unique_ds, sample_size=5)

    print(f"✅ Completed in {dry_result_5['execution_time_seconds']:.2f} seconds")
    print("\n📊 Block Details:")
    for block in dry_result_5.get("blocks_executed", []):
        if "LLM" in block.get("block_type", "") or "Evaluate" in block.get(
            "block_type", ""
        ):
            print(
                f"   - {block['block_name']}: {block.get('execution_time_seconds', 0):.2f}s, {block.get('input_rows', 0)} requests"
            )
    print()

    # Step 3: Predict for 200 samples
    print("=" * 80)
    print("⏱️  PREDICTION: Estimating Time for 200 Samples")
    print("=" * 80)
    print()

    # Use our time estimator directly
    prediction = estimate_execution_time(
        dry_run_1=dry_result_1,
        dry_run_2=dry_result_5,
        total_dataset_size=200,
        max_concurrency=100,
    )

    # Display results
    print("📈 PREDICTION RESULTS:")
    print("=" * 60)

    est_mins = prediction["estimated_time_seconds"] / 60
    est_hours = prediction["estimated_time_seconds"] / 3600

    print("\n📊 Target Dataset Size: 200 samples")
    print("\n⏱️  Time Estimate:")
    if prediction["estimated_time_seconds"] < 3600:
        print(
            f"   {est_mins:.1f} minutes ({prediction['estimated_time_seconds']:.0f} seconds)"
        )
    else:
        print(f"   {est_hours:.1f} hours ({est_mins:.0f} minutes)")

    if prediction.get("total_estimated_requests"):
        print("\n📝 Request Estimates:")
        print(f"   Total LLM requests: {prediction['total_estimated_requests']:,}")
        print(
            f"   Requests per sample: {prediction['total_estimated_requests'] / 200:.1f}"
        )

    # Show per-block breakdown
    if "block_estimates" in prediction:
        print("\n🔍 Per-Block Analysis:")
        for block in prediction["block_estimates"]:
            block_mins = block["estimated_time"] / 60
            print(f"\n   {block['block']}:")
            print(f"     • Time: {block_mins:.1f} minutes")
            print(f"     • Requests: {block['estimated_requests']:.0f}")
            print(f"     • Throughput: {block['throughput']:.2f} req/s")
            print(f"     • Amplification: {block['amplification']:.1f}x")

    # Summary
    print("\n" + "=" * 80)
    print("📊 SUMMARY")
    print("=" * 80)

    print("\n✅ Based on dry runs with 1 and 5 samples:")
    print(f"   • Processing 200 samples will take ~{est_mins:.0f} minutes")
    print(
        f"   • This will make ~{prediction.get('total_estimated_requests', 0):,} LLM requests"
    )
    print("   • Using max_concurrency=100")

    # Cost estimation
    cost_per_1k = 0.50  # Example cost
    if prediction.get("total_estimated_requests"):
        est_cost = (prediction["total_estimated_requests"] / 1000) * cost_per_1k
        print(f"   • Estimated cost: ${est_cost:.2f} (at ${cost_per_1k}/1k requests)")

    return prediction


if __name__ == "__main__":
    try:
        estimate = main()
        print("\n✨ Estimation completed successfully!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        # Standard
        import traceback

        traceback.print_exc()
        sys.exit(1)
