# User Guide: Running Generation

This guide covers executing data generation flows and monitoring progress.

## Starting Generation

### From the Flows Page

1. **Select configuration(s)** — Check the box next to flow(s) to run
2. **Click Actions → Run** — Or use the play button on individual rows
3. **Monitor progress** — Click flow name to expand terminal view

### From the Wizard

Click **Save and Run** on the Review step to save and immediately start generation.

## Execution Modes

### Single Flow

Run one configuration at a time:

1. Find configuration in table
2. Click the **▶ Run** button
3. Configuration expands to show terminal

### Batch Execution

Run multiple configurations simultaneously:

1. Select multiple configurations (checkboxes)
2. Click **Actions → Run**
3. All selected flows start in parallel

## Live Monitoring

### Terminal View

When a flow is running, the expanded view shows real-time output:

```
┌─────────────────────────────────────────────────────────────────┐
│ Advanced QA Generation Flow                    [Stop] [Clear]   │
├─────────────────────────────────────────────────────────────────┤
│ $ Starting flow 'Advanced QA Generation' v1.0.0                 │
│ $ Processing 100 samples across 4 blocks                        │
│                                                                 │
│ Executing block 1/4: generate_summary (ChatCompletionBlock)     │
│ [generate_summary] LLM Requests:  45%|████▌    | 45/100         │
│                                                                 │
│ 🔢 [generate_summary] Tokens → in: 12,345 | out: 4,567          │
│                                                                 │
│ Block 'generate_summary' completed in 23.5s                     │
│                                                                 │
│ Executing block 2/4: generate_questions...                      │
└─────────────────────────────────────────────────────────────────┘
```

### Progress Indicators

| Indicator | Meaning |
|-----------|---------|
| `[block_name] LLM Requests: 45%` | Progress through current block |
| `🔢 Tokens → in: X \| out: Y` | Token usage statistics |
| `Block completed in Xs` | Block timing |
| `Executing block N/M` | Overall flow progress |

### Live Monitoring Panel

For detailed metrics, the monitoring panel shows:

**Overall Progress:**

- Progress bar (0-100%)
- Blocks completed vs total
- Current block name
- Running/completed status

**Block Status:**

- Each block's status (pending/running/completed)
- Per-block token usage
- Per-block timing

**Token Statistics:**

- Total tokens used
- Input vs output breakdown
- Cost estimation (if configured)

## Stopping Generation

### Single Flow

1. Click **Stop** button in the expanded view
2. Or click the ⬛ button in the table row

### Batch Stop

1. Select running configurations
2. Click **Actions → Stop**

### What Happens on Stop

1. Current LLM request completes
2. Checkpoint is saved (if enabled)
3. Status changes to "cancelled"
4. Partial results are preserved

## Checkpointing & Resume

### Automatic Checkpointing

During generation, progress is saved periodically:

```
💾 Checkpoint saved: 50/100 samples completed
```

Checkpoints are stored in: `backend/checkpoints/{config_id}/`

### Resuming Failed Runs

If generation fails or is stopped:

1. The configuration shows "Failed" or "Cancelled" status
2. Click **Resume** (if checkpoints exist)
3. Generation continues from last checkpoint

### Checkpoint Information

The detail view shows checkpoint status:

```
Checkpoint Status:
✅ 3 checkpoints available
📊 75 samples completed
⏱️ Last checkpoint: 5 minutes ago
```

### Clearing Checkpoints

To start fresh:

1. Expand the configuration
2. Click **Clear Checkpoints**
3. Click **Run** for fresh start

## Output Files

### File Location

Generated data is saved to: `backend/outputs/`

**Naming convention:**

```
{flow_name}_{timestamp}.jsonl

Example:
advanced_qa_generation_20241127_143052.jsonl
```

### Download Output

From the expanded configuration view:

1. After completion, see "Output File" link
2. Click **Download** button
3. File downloads as JSONL

Or from Run History:

1. Navigate to **Flow Runs History**
2. Find the run
3. Click **Download** in actions

### Output Format

Generated data is JSONL with all columns:

```jsonl
{"document": "...", "domain": "...", "question": "...", "answer": "...", "score": 0.95}
{"document": "...", "domain": "...", "question": "...", "answer": "...", "score": 0.87}
```

## Execution States

| State | Icon | Meaning |
|-------|------|---------|
| `configured` | ✅ | Ready to run |
| `running` | 🔄 | Currently executing |
| `completed` | ✅ | Successfully finished |
| `failed` | ❌ | Error occurred |
| `cancelled` | ⚠️ | Stopped by user |

## Troubleshooting

### "Generation Failed"

**Check the terminal output for errors:**

Common issues:

| Error | Cause | Solution |
|-------|-------|----------|
| Connection refused | Model server down | Start vLLM/API server |
| 401 Unauthorized | Invalid API key | Check key in config |
| Rate limit exceeded | Too many requests | Reduce concurrency |
| Context too long | Input too large | Check input sizes |

### "Stuck at 0%"

- Verify model server is running
- Check network connectivity
- Review API base URL

### "Partial Results"

If generation stops partway:

1. Results up to failure point are saved
2. Check checkpoints for recoverable data
3. Resume or restart as needed

### "Out of Memory"

- Reduce `max_concurrency`
- Process fewer samples per run
- Use larger instance/more RAM

## Performance Tips

### Optimize Throughput

**Concurrency setting:**

```
max_concurrency: 10  # Default
```

Increase for faster runs (if server supports it):

```
max_concurrency: 50  # Aggressive
```

Decrease if hitting rate limits:

```
max_concurrency: 5   # Conservative
```

### Monitor Resources

Watch for:

- High CPU/memory on model server
- Rate limit warnings in logs
- Increasing latency per request

### Batch Sizes

For large datasets, consider:

- Multiple runs with `num_samples` limit
- Parallel execution of different configs
- Staggered start times

## Cost Management

### Estimation

Before full runs, estimate costs:

1. Run dry run (2-5 samples)
2. Note token usage from logs
3. Multiply by total samples
4. Apply provider pricing

### Optimization

Reduce costs by:

- Using smaller models for testing
- Limiting `max_tokens` appropriately
- Removing unnecessary blocks
- Caching repeated operations

## Next Steps

- [Run History](history.md) — View past runs
- [Flow Builder](flow-builder.md) — Optimize your flows
- [Model Configuration](model-configuration.md) — Tune model settings

