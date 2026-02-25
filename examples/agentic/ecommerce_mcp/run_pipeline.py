"""End-to-end test runner for the Toucan Tool-Use Data Generation flow.

This script loads the ShopInsights dataset, configures the Toucan flow with
a teacher LLM (GPT-5.2) and a Langflow agent (Qwen3 with MCP server), and
runs the full pipeline.

Prerequisites:
    1. MCP server running:     uv run python examples/agentic/ecommerce_mcp/server.py
    2. Dataset created:        uv run python examples/agentic/ecommerce_mcp/create_dataset.py
    3. Langflow running with Qwen3 agent connected to the MCP server

Usage:
    uv run python examples/agentic/ecommerce_mcp/run_pipeline.py \
        --dataset ./ecommerce_dataset \
        --openai-key $OPENAI_API_KEY \
        --langflow-url http://localhost:7860/api/v1/run/<flow-id> \
        [--langflow-key <key>] \
        [--teacher-model openai/gpt-5.2] \
        [--checkpoint-dir ./checkpoints]
"""

from __future__ import annotations

from pathlib import Path
import argparse
import os
import sys

from sdg_hub import Flow
import datasets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Toucan tool-use data generation with ShopInsights MCP server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        default="./ecommerce_dataset",
        help="Path to the HuggingFace dataset created by create_dataset.py",
    )
    parser.add_argument(
        "--openai-key",
        default=os.environ.get("OPENAI_API_KEY"),
        help="OpenAI API key for the teacher model (or set OPENAI_API_KEY env var)",
    )
    parser.add_argument(
        "--teacher-model",
        default="openai/gpt-5.2",
        help="LiteLLM model string for the teacher LLM (default: openai/gpt-5.2)",
    )
    parser.add_argument(
        "--langflow-url",
        default=os.environ.get(
            "LANGFLOW_URL", "http://localhost:7860/api/v1/run/default"
        ),
        help="Langflow agent API URL (or set LANGFLOW_URL env var)",
    )
    parser.add_argument(
        "--langflow-key",
        default=os.environ.get("LANGFLOW_API_KEY"),
        help="Langflow API key (or set LANGFLOW_API_KEY env var)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default="./checkpoints",
        help="Directory for checkpointing (default: ./checkpoints)",
    )
    parser.add_argument(
        "--output",
        default="toucan_ecommerce_results.parquet",
        help="Output parquet filename (default: toucan_ecommerce_results.parquet)",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=None,
        help="Max concurrent async requests (default: use flow default)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ---- Validate inputs ----
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(
            f"Error: Dataset not found at {dataset_path}\n"
            "Run create_dataset.py first:\n"
            "  uv run python examples/agentic/ecommerce_mcp/create_dataset.py"
        )
        sys.exit(1)

    if not args.openai_key:
        print(
            "Error: OpenAI API key required.\n"
            "Provide via --openai-key or set OPENAI_API_KEY env var."
        )
        sys.exit(1)

    # ---- Load dataset ----
    print(f"Loading dataset from {dataset_path} ...")
    ds = datasets.load_from_disk(str(dataset_path))
    print(f"  Rows: {len(ds)}, Columns: {ds.column_names}")
    print(f"  Tools: {len(ds[0]['tool_list'])}")

    # ---- Load flow ----
    flow_path = (
        Path(__file__).resolve().parents[3]
        / "src"
        / "sdg_hub"
        / "flows"
        / "agentic"
        / "tool_datagen"
        / "flow.yaml"
    )
    print(f"\nLoading flow from {flow_path} ...")
    flow = Flow.from_yaml(str(flow_path))
    flow.print_info()

    # ---- Configure teacher model (for question gen + quality scoring) ----
    print(f"\nConfiguring teacher model: {args.teacher_model}")
    flow.set_model_config(
        model=args.teacher_model,
        api_key=args.openai_key,
    )

    # ---- Configure Langflow agent ----
    print(f"Configuring Langflow agent: {args.langflow_url}")
    agent_kwargs = {
        "agent_framework": "langflow",
        "agent_url": args.langflow_url,
    }
    if args.langflow_key:
        agent_kwargs["agent_api_key"] = args.langflow_key
    flow.set_agent_config(**agent_kwargs)

    # ---- Run pipeline ----
    print(f"\nStarting pipeline (checkpoints: {args.checkpoint_dir}) ...")
    generate_kwargs = {
        "checkpoint_dir": args.checkpoint_dir,
    }
    if args.max_concurrency is not None:
        generate_kwargs["max_concurrency"] = args.max_concurrency

    result = flow.generate(ds, **generate_kwargs)

    # ---- Save results ----
    print(f"\nPipeline complete. Generated {len(result)} training examples.")

    if hasattr(result, "to_pandas"):
        df = result.to_pandas()
    else:
        df = result

    output_path = Path(args.output)
    df.to_parquet(output_path, index=False)
    print(f"Results saved to {output_path.resolve()}")
    print(f"Columns: {list(df.columns)}")


if __name__ == "__main__":
    main()
