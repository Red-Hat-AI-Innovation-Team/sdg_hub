# SPDX-License-Identifier: Apache-2.0
"""Time estimation utility for predicting full dataset execution time from dry_run results."""

# Standard
from typing import Dict, Optional
import math

# Default max concurrent requests used during dry runs
DRY_RUN_MAX_CONCURRENT = 100

# Conservative estimation factor (20% buffer for API variability, network latency, etc.)
ESTIMATION_BUFFER_FACTOR = 1.2


def is_llm_using_block(block_info: Dict) -> bool:
    """Detect if a block uses LLMs.

    Identifies blocks that make LLM API calls based on their type or parameters.
    This is used to calculate request amplification for LLM blocks.

    Parameters
    ----------
    block_info : Dict
        Block information from dry_run results containing block_type and parameters_used.

    Returns
    -------
    bool
        True if the block uses LLMs, False otherwise.

    Examples
    --------
    >>> block = {"block_type": "LLMChatBlock", "parameters_used": {"model": "gpt-4"}}
    >>> is_llm_using_block(block)
    True
    """
    block_type = block_info.get("block_type", "")

    # Direct LLM blocks or evaluation/verification blocks
    if any(kw in block_type for kw in ["LLMChatBlock", "Evaluate", "Verify"]):
        return True

    # Check for model parameters
    params = block_info.get("parameters_used", {})
    if any(key in params for key in ["model", "api_base", "api_key"]):
        return True

    return False


def calculate_block_throughput(
    block_1: Dict, block_2: Dict, samples_1: int, samples_2: int
) -> Dict:
    """Calculate throughput and amplification from two dry runs.

    Analyzes performance metrics from two dry runs with different sample sizes
    to estimate throughput (requests/second), amplification factor, and startup overhead.

    Parameters
    ----------
    block_1 : Dict
        Block execution info from first dry run.
    block_2 : Dict
        Block execution info from second dry run.
    samples_1 : int
        Number of samples in first dry run.
    samples_2 : int
        Number of samples in second dry run.

    Returns
    -------
    Dict
        Dictionary containing:
        - throughput: float, requests per second
        - amplification: float, average requests per input sample
        - startup_overhead: float, fixed startup time in seconds

    Raises
    ------
    ValueError
        If throughput cannot be calculated due to invalid measurements.

    Examples
    --------
    >>> block1 = {"execution_time_seconds": 1.0, "input_rows": 1, "block_name": "test"}
    >>> block2 = {"execution_time_seconds": 2.0, "input_rows": 5, "block_name": "test"}
    >>> result = calculate_block_throughput(block1, block2, 1, 5)
    >>> assert result["throughput"] > 0
    """
    time_1 = block_1.get("execution_time_seconds", 0)
    time_2 = block_2.get("execution_time_seconds", 0)
    requests_1 = block_1.get("input_rows", 0)
    requests_2 = block_2.get("input_rows", 0)

    # Calculate amplification (requests per sample)
    amp_1 = requests_1 / samples_1 if samples_1 > 0 else 1
    amp_2 = requests_2 / samples_2 if samples_2 > 0 else 1
    avg_amplification = (amp_1 + amp_2) / 2

    # Use linear scaling to extract throughput and overhead from two data points
    # Model: time = startup_overhead + (requests / throughput)

    if requests_2 > requests_1 and time_2 > time_1:
        # Calculate marginal time per request (slope of the line)
        marginal_time = (time_2 - time_1) / (requests_2 - requests_1)

        # Throughput is the inverse of marginal time
        measured_throughput = 1.0 / marginal_time if marginal_time > 0 else 0

        # Y-intercept is the startup overhead
        startup_overhead = max(0, time_1 - (requests_1 * marginal_time))
    else:
        # Fallback to simple calculation if we don't have good data for scaling
        throughput_1 = requests_1 / time_1 if time_1 > 0 else 0
        throughput_2 = requests_2 / time_2 if time_2 > 0 else 0
        measured_throughput = max(throughput_1, throughput_2)

        # Estimate overhead as a small fraction of time
        startup_overhead = min(2.0, time_1 * 0.1)  # Assume 10% overhead, max 2 seconds

    # If we have no valid measurements, raise an error
    if measured_throughput == 0:
        raise ValueError(
            f"Cannot calculate throughput for block '{block_1.get('block_name', 'unknown')}': "
            f"No valid measurements from dry runs (time_1={time_1}, time_2={time_2}, "
            f"requests_1={requests_1}, requests_2={requests_2})"
        )

    return {
        "throughput": measured_throughput,
        "amplification": avg_amplification,
        "startup_overhead": startup_overhead,
    }


def calculate_time_with_pipeline(
    num_requests: float,
    throughput: float,
    startup_overhead: float,
    max_concurrent: int = DRY_RUN_MAX_CONCURRENT,
) -> float:
    """Calculate time considering pipeline behavior and max concurrent limit.

    Models the execution time for a given number of requests based on throughput,
    startup overhead, and concurrency constraints. Applies non-linear scaling
    for diminishing returns at high concurrency levels.

    Parameters
    ----------
    num_requests : float
        Total number of requests to process.
    throughput : float
        Base throughput in requests per second.
    startup_overhead : float
        Fixed startup time overhead in seconds.
    max_concurrent : int, optional
        Maximum number of concurrent requests, by default 100.

    Returns
    -------
    float
        Estimated total execution time in seconds.

    Examples
    --------
    >>> time = calculate_time_with_pipeline(1000, 10.0, 0.5, 50)
    >>> assert time > 0
    """
    if num_requests <= 0:
        return 0

    # Validate and clamp max_concurrent to avoid division by zero
    if max_concurrent is None or max_concurrent <= 0:
        max_concurrent = 1

    # The throughput is what we measured - it represents the server's processing capability
    if max_concurrent == 1:
        # Sequential execution - no pipelining benefit
        effective_throughput = throughput
    else:
        # Concurrent execution - small pipelining benefit
        # At most 10% improvement from perfect pipelining (conservative estimate)
        # Logarithmic growth to model diminishing returns
        pipelining_factor = 1.0 + (0.1 * math.log(max_concurrent) / math.log(100))
        pipelining_factor = min(pipelining_factor, 1.1)  # Cap at 10% improvement
        effective_throughput = throughput * pipelining_factor

    # Calculate total time
    base_time = startup_overhead + (num_requests / effective_throughput)

    return base_time


def estimate_execution_time(
    dry_run_1: Dict,
    dry_run_2: Optional[Dict] = None,
    total_dataset_size: Optional[int] = None,
    max_concurrency: Optional[int] = None,
) -> Dict:
    """Estimate execution time based on dry run results.

    Estimates the total execution time for a full dataset based on one or two
    dry runs with smaller sample sizes. For async blocks (with two dry runs),
    calculates throughput and concurrency benefits. For sync blocks (single dry run),
    performs simple linear scaling.

    The estimates include a conservative buffer (20%) to account for API variability,
    network latency, and other real-world factors.

    Parameters
    ----------
    dry_run_1 : Dict
        Results from first dry run, must contain 'sample_size' and 'execution_time_seconds'.
    dry_run_2 : Optional[Dict], optional
        Results from second dry run for async estimation, by default None.
    total_dataset_size : Optional[int], optional
        Size of full dataset to estimate for. If None, uses original_dataset_size from dry_run_1.
    max_concurrency : Optional[int], optional
        Maximum concurrent requests allowed, by default 100.

    Returns
    -------
    Dict
        Estimation results containing:
        - estimated_time_seconds: float, estimated time with current configuration (includes buffer)
        - total_estimated_requests: int, total LLM requests (0 for sync blocks)
        - block_estimates: list, per-block estimates (for async blocks)
        - note: str, additional information about the estimation

    Examples
    --------
    >>> dry_run = {"sample_size": 2, "execution_time_seconds": 10.0}
    >>> result = estimate_execution_time(dry_run, total_dataset_size=100)
    >>> assert result["estimated_time_seconds"] > 0
    >>>
    >>> # With two dry runs for async estimation
    >>> dry_run_1 = {"sample_size": 1, "execution_time_seconds": 5.0, "blocks_executed": [...]}
    >>> dry_run_2 = {"sample_size": 5, "execution_time_seconds": 20.0, "blocks_executed": [...]}
    >>> result = estimate_execution_time(dry_run_1, dry_run_2, total_dataset_size=1000)
    >>> assert result["estimated_time_seconds"] > 0
    """
    # Set defaults
    if max_concurrency is None:
        max_concurrency = DRY_RUN_MAX_CONCURRENT

    if total_dataset_size is None:
        total_dataset_size = dry_run_1.get(
            "original_dataset_size", dry_run_1["sample_size"]
        )

    # Get sample sizes
    samples_1 = dry_run_1["sample_size"]
    samples_2 = (
        dry_run_2["sample_size"] if dry_run_2 else 5
    )  # Default to 5 if not provided

    # If only one dry run, do simple scaling
    if dry_run_2 is None:
        # Process each block individually for synchronous execution
        blocks_executed = dry_run_1.get("blocks_executed", [])
        if not blocks_executed:
            # Fallback to simple scaling if no block details available
            total_time = dry_run_1["execution_time_seconds"]
            simple_estimate = (total_time / samples_1) * total_dataset_size
            # Apply conservative buffer
            simple_estimate = simple_estimate * ESTIMATION_BUFFER_FACTOR
            return {
                "estimated_time_seconds": simple_estimate,
                "total_estimated_requests": 0,
                "note": "Synchronous execution - linear scaling from dry run",
            }

        # Calculate time for each block and sum them
        total_estimated_time = 0
        for block in blocks_executed:
            block_time = block.get("execution_time_seconds", 0)
            input_rows = block.get("input_rows", samples_1)

            # Calculate time per row for this block
            if input_rows > 0:
                time_per_row = block_time / input_rows
                block_total_time = time_per_row * total_dataset_size
                total_estimated_time += block_total_time

        # Apply conservative buffer
        total_estimated_time = total_estimated_time * ESTIMATION_BUFFER_FACTOR
        return {
            "estimated_time_seconds": total_estimated_time,
            "total_estimated_requests": 0,
            "note": "Synchronous execution - no concurrency",
        }

    # Analyze each block with async execution
    block_estimates = []
    total_time = 0
    total_requests = 0

    # Process each block
    for i, block_1 in enumerate(dry_run_1.get("blocks_executed", [])):
        if i >= len(dry_run_2.get("blocks_executed", [])):
            break

        block_2 = dry_run_2["blocks_executed"][i]

        # Only process LLM blocks
        if not is_llm_using_block(block_1):
            continue

        # Calculate throughput and amplification
        analysis = calculate_block_throughput(block_1, block_2, samples_1, samples_2)

        # Estimate requests for full dataset
        estimated_requests = total_dataset_size * analysis["amplification"]

        # Calculate time with pipeline model
        block_time = calculate_time_with_pipeline(
            estimated_requests,
            analysis["throughput"],
            analysis["startup_overhead"],
            max_concurrency,
        )

        total_time += block_time
        total_requests += estimated_requests

        block_estimates.append(
            {
                "block": block_1["block_name"],
                "estimated_requests": estimated_requests,
                "throughput": analysis["throughput"],
                "estimated_time": block_time,
                "amplification": analysis["amplification"],
                "startup_overhead": analysis["startup_overhead"],
            }
        )

    # Apply conservative buffer to account for API variability, network issues, etc.
    total_time = total_time * ESTIMATION_BUFFER_FACTOR

    return {
        "estimated_time_seconds": total_time,
        "total_estimated_requests": int(total_requests),
        "block_estimates": block_estimates,
    }
