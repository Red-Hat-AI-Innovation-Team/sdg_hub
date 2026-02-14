# SPDX-License-Identifier: Apache-2.0
"""Dry-run worker executed in a subprocess via multiprocessing."""


def dry_run_worker(
    log_queue,
    flow_path,
    model_config,
    dataset_params,
    sample_size,
    enable_time_estimation,
    max_concurrency,
):
    """Worker process for running dry run.

    Args:
        log_queue: Queue for sending logs back to main process
        flow_path: Path to the flow YAML file
        model_config: Model configuration dict
        dataset_params: Dataset parameters dict
        sample_size: Number of samples for dry run
        enable_time_estimation: Whether to estimate time
        max_concurrency: Maximum concurrent requests
    """
    try:
        import sys
        import time

        class TeeOutput:
            def __init__(self, queue):
                self.queue = queue

            def write(self, text):
                if text:
                    self.queue.put(
                        {"type": "log", "message": text, "timestamp": time.time()}
                    )

            def flush(self):
                pass

        sys.stdout = TeeOutput(log_queue)
        sys.stderr = TeeOutput(log_queue)

        # Load flow
        from sdg_hub import Flow

        flow = Flow.from_yaml(flow_path)

        # Apply model configuration
        if model_config:
            kwargs = {}
            if model_config.get("model"):
                kwargs["model"] = model_config["model"]
            if model_config.get("api_base"):
                kwargs["api_base"] = model_config["api_base"]
            if model_config.get("api_key"):
                kwargs["api_key"] = model_config["api_key"]
            if model_config.get("additional_params"):
                kwargs.update(model_config["additional_params"])

            if kwargs:
                flow.set_model_config(**kwargs)

        # Load dataset as pandas DataFrame
        from pathlib import Path

        import pandas as pd

        if dataset_params:
            data_files = dataset_params["data_files"]
            file_path = Path(data_files)
            file_format = dataset_params.get("file_format", "auto")

            # Auto-detect format from extension if needed
            if file_format == "auto":
                suffix = file_path.suffix.lower()
                format_map = {
                    ".jsonl": "jsonl",
                    ".json": "json",
                    ".csv": "csv",
                    ".parquet": "parquet",
                    ".pq": "parquet",
                }
                file_format = format_map.get(suffix, "jsonl")

            # Load as pandas DataFrame
            if file_format == "jsonl":
                df = pd.read_json(data_files, lines=True)
            elif file_format == "json":
                df = pd.read_json(data_files)
            elif file_format == "csv":
                csv_delimiter = dataset_params.get("csv_delimiter", ",")
                csv_encoding = dataset_params.get("csv_encoding", "utf-8")
                df = pd.read_csv(
                    data_files, delimiter=csv_delimiter, encoding=csv_encoding
                )
            elif file_format == "parquet":
                df = pd.read_parquet(data_files)
            else:
                df = pd.read_json(data_files, lines=True)

            # Apply shuffle if requested
            if dataset_params.get("shuffle"):
                df = df.sample(
                    frac=1, random_state=dataset_params.get("seed", 42)
                ).reset_index(drop=True)
        else:
            log_queue.put(
                {"type": "error", "message": "No dataset parameters provided"}
            )
            return

        # Run dry run
        dry_result = flow.dry_run(
            df,
            sample_size=sample_size,
            enable_time_estimation=enable_time_estimation,
            max_concurrency=max_concurrency,
        )

        # Send completion event
        log_queue.put({"type": "complete", "result": dry_result})

    except Exception as e:
        import traceback

        log_queue.put(
            {
                "type": "error",
                "message": f"Dry run failed: {str(e)}\n{traceback.format_exc()}",
            }
        )
