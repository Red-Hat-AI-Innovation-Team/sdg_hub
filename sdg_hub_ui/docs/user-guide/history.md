# User Guide: Run History

Track and manage all your data generation runs.

## Flow Runs History Page

Navigate to **Flow Runs History** in the sidebar to view all past executions.

## Run Records

Each run is recorded with:

| Field | Description |
|-------|-------------|
| **Run ID** | Unique identifier |
| **Config ID** | Associated configuration |
| **Flow Name** | Flow that was executed |
| **Flow Type** | "existing" or "custom" |
| **Model** | LLM used |
| **Status** | running/completed/failed/cancelled |
| **Start Time** | When run began |
| **End Time** | When run finished |
| **Duration** | Total execution time |
| **Input Samples** | Rows processed |
| **Output Samples** | Rows generated |
| **Output Columns** | Columns in output |
| **Dataset File** | Input data source |
| **Output File** | Generated output path |
| **Error Message** | Error details (if failed) |

## Viewing History

### List View

All runs are displayed in a table:

```
┌─────────┬─────────────────────┬──────────┬──────────┬─────────┬─────────┐
│ Status  │ Flow Name           │ Model    │ Duration │ Samples │ Actions │
├─────────┼─────────────────────┼──────────┼──────────┼─────────┼─────────┤
│ ✅      │ QA Generation       │ llama-70B│ 5m 23s   │ 100     │ 📥 🗑   │
│ ❌      │ Summary Flow        │ gpt-4o   │ 2m 15s   │ 50      │ 📥 🗑   │
│ ⚠️      │ Custom Pipeline     │ llama-70B│ 1m 03s   │ 25      │ 📥 🗑   │
└─────────┴─────────────────────┴──────────┴──────────┴─────────┴─────────┘
```

### Filtering

Filter runs by:

- **Status** — completed, failed, cancelled
- **Flow name** — Search by flow
- **Date range** — Filter by time period
- **Model** — Filter by LLM used

### Sorting

Click column headers to sort:

- Start time (newest/oldest)
- Duration (fastest/slowest)
- Output samples (most/least)

## Run Details

Click a run to expand details:

```
┌─────────────────────────────────────────────────────────────────┐
│ Run: run_abc123_1732789012                                      │
├─────────────────────────────────────────────────────────────────┤
│ Configuration                                                   │
│ ├── Flow: Advanced QA Generation (existing)                     │
│ ├── Model: hosted_vllm/meta-llama/Llama-3.3-70B-Instruct       │
│ ├── API Base: http://localhost:8000/v1                         │
│ └── Dataset: uploads/seed_data.jsonl                           │
│                                                                 │
│ Execution                                                       │
│ ├── Status: ✅ Completed                                        │
│ ├── Started: Nov 27, 2024 2:30:00 PM                           │
│ ├── Finished: Nov 27, 2024 2:35:23 PM                          │
│ ├── Duration: 5 minutes 23 seconds                             │
│ ├── Input: 100 samples                                         │
│ └── Output: 100 samples, 42 columns                            │
│                                                                 │
│ Output File                                                     │
│ └── outputs/advanced_qa_generation_20241127_143523.jsonl       │
│                                                                 │
│                                          [Download] [Delete]    │
└─────────────────────────────────────────────────────────────────┘
```

## Downloading Outputs

### From History Page

1. Find the run in the list
2. Click the **📥 Download** button
3. JSONL file downloads to your browser

### File Contents

Downloaded files contain all generated columns:

```jsonl
{
  "document": "Original input text...",
  "domain": "science",
  "summary": "Generated summary...",
  "question_1": "Generated question...",
  "answer_1": "Generated answer...",
  "faithfulness_score": 0.95
}
```

### Bulk Download

Select multiple runs and use **Actions → Download All** to get a ZIP archive.

## Managing Runs

### Delete Single Run

1. Click **🗑 Delete** button on the run
2. Confirm deletion

**Note:** This deletes the run record AND the output file.

### Bulk Delete

1. Select multiple runs
2. Click **Actions → Delete Selected**
3. Confirm deletion

### Run Retention

By default, all runs are kept indefinitely.

To configure auto-cleanup, set:

```bash
export SDG_HUB_RUN_RETENTION_DAYS=30  # Delete runs older than 30 days
```

## Run Metrics

### Aggregated Statistics

The history page shows summary stats:

```
Total Runs: 47
├── Completed: 42 (89%)
├── Failed: 3 (6%)
└── Cancelled: 2 (4%)

Total Samples Generated: 12,450
Average Duration: 8m 32s
```

### Performance Trends

Track generation performance over time:

- Average duration per run
- Success rate trends
- Samples processed per day

## Troubleshooting Failed Runs

### Viewing Error Details

Failed runs include error information:

```
Error Message:
Connection refused: Unable to connect to model server at localhost:8000
```

### Common Failure Patterns

| Error | Likely Cause | Solution |
|-------|--------------|----------|
| Connection refused | Server not running | Start model server |
| 401 Unauthorized | Bad API key | Update configuration |
| Rate limit | Too many requests | Reduce concurrency |
| Timeout | Slow responses | Increase timeout or reduce load |
| Out of memory | Large batch | Reduce samples or concurrency |

### Recovering from Failures

1. **Check error message** — Understand what failed
2. **Fix configuration** — Update model/dataset settings
3. **Resume if possible** — Use checkpoints to continue
4. **Restart fresh** — Clear checkpoints and re-run

## Export & Reporting

### Export Run History

Download run history as CSV:

1. Click **Export History**
2. Choose date range
3. Download CSV file

### CSV Fields

```csv
run_id,config_id,flow_name,model,status,start_time,duration_seconds,input_samples,output_samples
run_abc123,...,QA Generation,llama-70B,completed,2024-11-27T14:30:00,323,100,100
```

### Integration with Analytics

Use exported data for:

- Cost tracking
- Performance analysis
- Usage reporting
- Capacity planning

## Best Practices

### Naming Conventions

Use descriptive flow names for easy identification:

```
✅ "Customer FAQ Generation - v2"
✅ "Product Docs QA - 2024 Q4"
❌ "test"
❌ "flow1"
```

### Regular Cleanup

Periodically review and delete:

- Failed test runs
- Old development runs
- Duplicate runs

### Output Organization

Download and organize outputs:

```
outputs/
├── 2024-11/
│   ├── qa_generation_20241115.jsonl
│   └── summary_flow_20241120.jsonl
└── 2024-12/
    └── ...
```

## Next Steps

- [Running Generation](generation.md) — Execute new flows
- [Flow Configuration](flow-configuration.md) — Modify configurations
- [Overview](overview.md) — Return to UI overview

