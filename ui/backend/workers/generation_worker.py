# SPDX-License-Identifier: Apache-2.0
"""Generation worker executed in a subprocess via multiprocessing."""

from config import CHECKPOINTS_DIR


def generation_worker(
    log_queue,
    flow_path,
    model_config,
    dataset_params,
    max_concurrency,
    log_dir,
    checkpoint_dir=None,
    save_freq=None,
    resume_from_checkpoint=False,
):
    """Worker process for running flow generation.

    Args:
        log_queue: Queue for sending logs back to main process
        flow_path: Path to the flow YAML file
        model_config: Model configuration dict
        dataset_params: Dataset parameters dict
        max_concurrency: Maximum concurrent requests
        log_dir: Directory for logs
        checkpoint_dir: Directory for saving checkpoints (optional)
        save_freq: Save checkpoint every N samples (optional)
        resume_from_checkpoint: If True, resume from existing checkpoints
    """
    try:
        # Disable rich console formatting to prevent broken output
        # This must be done BEFORE importing any modules that use rich
        import os
        os.environ['NO_COLOR'] = '1'  # Disable color output
        os.environ['TERM'] = 'dumb'   # Set terminal to dumb to disable fancy formatting
        os.environ['COLUMNS'] = '200' # Set wide terminal to prevent wrapping
        os.environ['FORCE_TERMINAL'] = '0'  # Disable forced terminal
        
        # Redirect stdout/stderr to queue
        import sys
        import time
        import re
        import logging

        # Counter to track LLM requests (using dict to allow mutation in nested scope)
        llm_request_counter = {'count': 0}

        class TeeOutput:
            def __init__(self, queue, counter=None):
                self.queue = queue
                self.counter = counter

            def write(self, text):
                if text:
                    self.queue.put(
                        {"type": "log", "message": text, "timestamp": time.time()}
                    )
                    # Count LLM requests from completion log messages
                    # Format: "llm_chat_block - INFO - Generation completed successfully for X samples"
                    if self.counter is not None and 'llm_chat_block' in text.lower() and 'generation completed successfully for' in text.lower():
                        match = re.search(r'generation completed successfully for (\d+) samples', text.lower())
                        if match:
                            self.counter['count'] += int(match.group(1))

            def flush(self):
                pass

        # Custom logging handler to capture log messages and count LLM requests
        class QueueLoggingHandler(logging.Handler):
            def __init__(self, queue, counter=None):
                super().__init__()
                self.queue = queue
                self.counter = counter

            def emit(self, record):
                try:
                    msg = self.format(record)
                    self.queue.put(
                        {"type": "log", "message": msg + '\n', "timestamp": time.time()}
                    )
                    # Count LLM requests from llm_chat_block logger
                    if self.counter is not None and 'llm_chat_block' in msg.lower() and 'generation completed successfully for' in msg.lower():
                        match = re.search(r'generation completed successfully for (\d+) samples', msg.lower())
                        if match:
                            self.counter['count'] += int(match.group(1))
                except Exception:
                    self.handleError(record)

        sys.stdout = TeeOutput(log_queue, llm_request_counter)
        sys.stderr = TeeOutput(log_queue, llm_request_counter)
        
        # Add custom handler to root logger to capture all log messages
        queue_handler = QueueLoggingHandler(log_queue, llm_request_counter)
        queue_handler.setLevel(logging.INFO)
        queue_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logging.getLogger().addHandler(queue_handler)
        # Also add to sdg_hub logger specifically
        logging.getLogger('sdg_hub').addHandler(queue_handler)

        # Load flow
        from sdg_hub import Flow

        flow = Flow.from_yaml(flow_path)

        # Apply model configuration
        if model_config:
            # Re-apply configuration logic
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

        # Load dataset as pandas DataFrame for optimal performance
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

            # Limit samples if specified
            if dataset_params.get("num_samples"):
                df = df.head(min(dataset_params["num_samples"], len(df)))
            
            # Add any missing columns with provided values
            added_columns = dataset_params.get("added_columns")
            if added_columns:
                for col_name, col_value in added_columns.items():
                    if col_name not in df.columns:
                        df[col_name] = col_value
                        print(f"Added missing column '{col_name}' to dataset")
        else:
            log_queue.put(
                {"type": "error", "message": "No dataset parameters provided"}
            )
            return

        # Clear checkpoints if not resuming (starting fresh)
        if checkpoint_dir and not resume_from_checkpoint:
            import shutil

            checkpoint_path = Path(checkpoint_dir)
            # Validate checkpoint_path is within CHECKPOINTS_DIR before removing
            try:
                checkpoint_path.resolve().relative_to(CHECKPOINTS_DIR.resolve())
                if checkpoint_path.exists():
                    shutil.rmtree(checkpoint_path)
            except ValueError:
                log_queue.put(
                    {"type": "error", "message": "Invalid checkpoint directory"}
                )
                log_queue.put(
                    {
                        "type": "log",
                        "message": "🗑️ Cleared existing checkpoints for fresh start\n",
                        "timestamp": time.time(),
                    }
                )

        # Prepare checkpoint parameters
        generate_kwargs = {"max_concurrency": max_concurrency, "log_dir": log_dir}

        if checkpoint_dir:
            generate_kwargs["checkpoint_dir"] = checkpoint_dir
            if save_freq:
                generate_kwargs["save_freq"] = save_freq

            if resume_from_checkpoint:
                log_queue.put(
                    {
                        "type": "log",
                        "message": "📂 Resuming from checkpoint...\n",
                        "timestamp": time.time(),
                    }
                )
            else:
                log_queue.put(
                    {
                        "type": "log",
                        "message": f"💾 Checkpointing enabled (save every {save_freq or 'completion'} samples)\n",
                        "timestamp": time.time(),
                    }
                )

        # Run generation with pandas DataFrame
        generated_df = flow.generate(df, **generate_kwargs)

        # Convert result to list for pickling back (pandas DataFrame)
        # Handle both pandas DataFrame and HuggingFace Dataset returns
        if hasattr(generated_df, "to_dict"):
            # pandas DataFrame
            dataset_list = generated_df.to_dict(orient="records")
            column_names = generated_df.columns.tolist()
        else:
            # HuggingFace Dataset (fallback for backward compatibility)
            dataset_list = generated_df.to_list()
            column_names = list(generated_df.column_names)

        # Clean up logging handler
        logging.getLogger().removeHandler(queue_handler)
        logging.getLogger('sdg_hub').removeHandler(queue_handler)
        
        log_queue.put(
            {
                "type": "result",
                "dataset_list": dataset_list,
                "column_names": column_names,
                "llm_requests": llm_request_counter['count'],
            }
        )

    except Exception as e:
        # Try to clean up logging handler on error too
        try:
            logging.getLogger().removeHandler(queue_handler)
            logging.getLogger('sdg_hub').removeHandler(queue_handler)
        except:
            pass
        import traceback

        traceback.print_exc()
        log_queue.put({"type": "error", "message": str(e)})
