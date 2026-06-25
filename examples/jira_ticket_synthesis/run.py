"""Synthesize JIRA tickets from seed examples using few-shot ICL.

Usage:
    # With your own seed data (CSV with a 'document' column):
    python run.py --input seeds.csv

    # With placeholder seed data (for testing the pipeline):
    python run.py

    # Customize the model:
    python run.py --model openai/gpt-4o --input seeds.csv
"""

from pathlib import Path
import argparse
import os

from dotenv import load_dotenv
import pandas as pd

from sdg_hub import Flow

load_dotenv()

SCRIPT_DIR = Path(__file__).parent
FLOW_YAML = SCRIPT_DIR / "flow.yaml"


def make_placeholder_seeds(n: int = 20) -> pd.DataFrame:
    """Create placeholder JIRA ticket seeds for testing the pipeline."""
    templates = [
        (
            "BUG: Login page returns 500 on invalid email format\n"
            "Priority: High | Component: Auth\n"
            "Steps to reproduce:\n1. Navigate to /login\n2. Enter 'not-an-email' "
            "in the email field\n3. Click Submit\n"
            "Expected: Validation error shown to user\n"
            "Actual: HTTP 500 Internal Server Error\n"
            "Stack trace points to EmailValidator.parse() "
            "throwing unhandled FormatException."
        ),
        (
            "FEATURE: Add CSV export to analytics dashboard\n"
            "Priority: Medium | Component: Analytics\n"
            "As a product manager, I want to export the current dashboard view "
            "as a CSV file so that I can share metrics with stakeholders who "
            "don't have dashboard access.\n"
            "Acceptance criteria:\n- Export button visible on all dashboard tabs\n"
            "- CSV includes all visible columns with current filters applied\n"
            "- File name includes dashboard name and date"
        ),
        (
            "BUG: Memory leak in background job processor\n"
            "Priority: Critical | Component: Worker\n"
            "The Sidekiq worker process grows to 2GB+ RSS after ~6 hours of "
            "continuous operation. Heap dump shows accumulation of "
            "ActiveRecord::Result objects that are never GC'd.\n"
            "Workaround: Restart workers every 4 hours via cron.\n"
            "Root cause suspected: connection pool not releasing results "
            "when batch size > 1000."
        ),
        (
            "TASK: Upgrade PostgreSQL from 14 to 16\n"
            "Priority: Medium | Component: Infrastructure\n"
            "Current production database is running PostgreSQL 14.9. "
            "Version 16 brings improved query parallelism and SIMD-accelerated "
            "aggregation that should help with our reporting queries.\n"
            "Plan:\n1. Test on staging with production snapshot\n"
            "2. Run regression suite against new version\n"
            "3. Schedule maintenance window for migration\n"
            "4. Update connection strings in Vault"
        ),
        (
            "BUG: Race condition in inventory reservation\n"
            "Priority: Critical | Component: Orders\n"
            "When two users add the last item to cart simultaneously, "
            "both succeed and we over-commit inventory. Happens ~3 times/week "
            "on high-demand items.\n"
            "Expected: Second request should fail with 'out of stock'.\n"
            "Fix approach: Use SELECT FOR UPDATE or optimistic locking "
            "on inventory_items.quantity."
        ),
        (
            "FEATURE: Implement SSO via SAML 2.0 for enterprise customers\n"
            "Priority: High | Component: Auth\n"
            "Enterprise customers on the Business plan need SAML-based SSO. "
            "Integration with Okta and Azure AD required at launch.\n"
            "Requirements:\n- SP-initiated flow\n- JIT user provisioning\n"
            "- Group-to-role mapping configurable per tenant\n"
            "- IdP metadata import via URL or XML upload"
        ),
        (
            "BUG: Search results inconsistent after index rebuild\n"
            "Priority: High | Component: Search\n"
            "After the nightly Elasticsearch index rebuild, search results "
            "differ from pre-rebuild for ~15 minutes. Users report missing "
            "results for recently updated documents.\n"
            "Investigation: The alias swap happens before the new index "
            "finishes warming. Adding a health check gate should fix this."
        ),
        (
            "TASK: Set up Datadog APM for payment service\n"
            "Priority: Medium | Component: Observability\n"
            "The payment service currently has basic logging but no distributed "
            "tracing. We need APM instrumentation to debug latency spikes "
            "during peak hours.\n"
            "Scope:\n- Install dd-trace agent sidecar\n"
            "- Instrument HTTP handlers and DB calls\n"
            "- Add custom spans for payment gateway round-trips\n"
            "- Set up latency percentile dashboards"
        ),
        (
            "FEATURE: Real-time collaboration on documents\n"
            "Priority: High | Component: Editor\n"
            "Users need to co-edit documents simultaneously with live cursors "
            "and presence indicators.\n"
            "Technical approach: Implement CRDT-based text synchronization "
            "using Yjs. WebSocket connections for real-time updates.\n"
            "MVP scope:\n- 2-user concurrent editing\n"
            "- Cursor position sharing\n- Conflict-free text merge\n"
            "- Connection status indicator"
        ),
        (
            "BUG: Timezone handling broken for recurring events\n"
            "Priority: High | Component: Calendar\n"
            "Recurring events created in PST display at wrong times for "
            "users in EST after DST transition.\n"
            "Root cause: Events store wall-clock time without IANA timezone. "
            "The recurrence expansion uses server timezone (UTC) instead of "
            "the creator's timezone.\n"
            "Fix: Store IANA tz identifier with each recurrence rule and "
            "expand in the user's local timezone."
        ),
        (
            "TASK: Implement rate limiting for public API\n"
            "Priority: High | Component: API Gateway\n"
            "The public API currently has no rate limiting, making it "
            "vulnerable to abuse. Need token-bucket rate limiting per API key.\n"
            "Requirements:\n- 100 requests/minute for free tier\n"
            "- 1000 requests/minute for paid tier\n"
            "- Rate limit headers in response (X-RateLimit-*)\n"
            "- Redis-backed distributed counter\n"
            "- Graceful 429 response with Retry-After header"
        ),
        (
            "BUG: File upload silently corrupts files > 100MB\n"
            "Priority: Critical | Component: Storage\n"
            "Files uploaded via the web UI that exceed 100MB are stored "
            "but corrupted on download. The SHA-256 checksum mismatch rate "
            "is 100% for files in the 100-500MB range.\n"
            "Cause: The multipart upload chunking code uses a 32-bit offset "
            "that overflows. Chunks after the first 100MB write to wrong "
            "positions in the reassembled file."
        ),
        (
            "FEATURE: Webhook delivery with retry and dead-letter queue\n"
            "Priority: Medium | Component: Integrations\n"
            "Customers need reliable webhook delivery for event notifications. "
            "Current implementation is fire-and-forget with no retries.\n"
            "Requirements:\n- Exponential backoff retry (max 5 attempts)\n"
            "- Dead-letter queue for permanently failed deliveries\n"
            "- Webhook delivery log viewable in customer dashboard\n"
            "- HMAC signature verification on payloads\n"
            "- Configurable event filtering per endpoint"
        ),
        (
            "BUG: Dashboard loading time increased 3x after adding filters\n"
            "Priority: High | Component: Frontend\n"
            "After PR #4521 added date range filters, the main dashboard "
            "takes 8-12 seconds to load (was 2-3 seconds). Profiling shows "
            "the filter component triggers 6 redundant API calls on mount.\n"
            "Fix: Debounce filter state changes and batch initial queries "
            "into a single request."
        ),
        (
            "TASK: Migrate CI/CD from Jenkins to GitHub Actions\n"
            "Priority: Medium | Component: DevOps\n"
            "Jenkins server is EOL and maintenance burden is high. "
            "Migrating to GitHub Actions for all repositories.\n"
            "Migration plan:\n1. Convert Jenkinsfiles to workflow YAML\n"
            "2. Set up self-hosted runners for GPU tests\n"
            "3. Migrate secrets to GitHub Secrets\n"
            "4. Parallel-run both systems for 2 weeks\n"
            "5. Decommission Jenkins"
        ),
        (
            "BUG: Email notifications sent in wrong language\n"
            "Priority: Medium | Component: Notifications\n"
            "Users with locale set to 'ja' (Japanese) receive email "
            "notifications in English. The notification service falls back "
            "to 'en' when the translation key exists but the value is empty.\n"
            "Affected templates: password_reset, order_confirmation, "
            "shipping_update.\n"
            "Fix: Treat empty translation values as missing and fall back "
            "to the default locale's non-empty value."
        ),
        (
            "FEATURE: Audit log for admin actions\n"
            "Priority: High | Component: Admin\n"
            "Compliance requires a tamper-evident audit log of all "
            "administrative actions (user management, config changes, "
            "data exports).\n"
            "Requirements:\n- Append-only log stored in separate database\n"
            "- Each entry: actor, action, target, timestamp, IP, user-agent\n"
            "- Search and filter UI for compliance team\n"
            "- 90-day retention with archival to S3\n"
            "- Export to SIEM via syslog"
        ),
        (
            "BUG: Mobile app crashes on deep link with special characters\n"
            "Priority: High | Component: Mobile\n"
            "Deep links containing URL-encoded special characters "
            "(e.g., %26, %3D) cause the app to crash on Android 13+.\n"
            "Stack trace shows URI.parse() throws IllegalArgumentException "
            "when the decoded path contains unescaped ampersands.\n"
            "Affected: Android app v3.2.1+\n"
            "Workaround: Double-encode special characters in deep link URLs."
        ),
        (
            "TASK: Implement database connection pooling with PgBouncer\n"
            "Priority: Medium | Component: Infrastructure\n"
            "Direct connections to PostgreSQL are exhausting the max_connections "
            "limit (200) during traffic spikes. Need connection pooling.\n"
            "Plan:\n- Deploy PgBouncer in transaction mode\n"
            "- Configure pool_size=50 per application instance\n"
            "- Update application connection strings\n"
            "- Add PgBouncer metrics to monitoring\n"
            "- Load test with 2x expected peak traffic"
        ),
        (
            "FEATURE: Bulk import via CSV with validation preview\n"
            "Priority: Medium | Component: Data Import\n"
            "Users need to import up to 10,000 records at a time via CSV. "
            "Current single-record API is too slow.\n"
            "Requirements:\n- Upload CSV with drag-and-drop\n"
            "- Preview first 10 rows with column mapping\n"
            "- Validate all rows before import (show error summary)\n"
            "- Background processing with progress bar\n"
            "- Email notification on completion with success/error counts"
        ),
    ]

    return pd.DataFrame({"document": templates[:n]})


def main():
    parser = argparse.ArgumentParser(
        description="Synthesize JIRA tickets from seed examples"
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to CSV file with a 'document' column containing seed tickets",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="openai/gpt-4o-mini",
        help="LiteLLM model identifier (default: openai/gpt-4o-mini)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="synthesized_tickets.csv",
        help="Output CSV path (default: synthesized_tickets.csv)",
    )
    args = parser.parse_args()

    if args.input:
        dataset = pd.read_csv(args.input)
        if "document" not in dataset.columns:
            raise ValueError(
                f"Input CSV must have a 'document' column. "
                f"Found: {list(dataset.columns)}"
            )
    else:
        print("No --input provided, using placeholder seed data (20 tickets)")
        dataset = make_placeholder_seeds()

    print(f"Seed dataset: {len(dataset)} rows")

    flow = Flow.from_yaml(str(FLOW_YAML))

    api_key = os.getenv("OPENAI_API_KEY")
    flow.set_model_config(model=args.model, api_key=api_key)

    print(f"Running flow with model={args.model} ...")
    result = flow.generate(dataset)

    print(f"Result: {result.shape[0]} rows, {result.shape[1]} columns")

    result.to_csv(args.output, index=False)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
