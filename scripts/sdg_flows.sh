#!/usr/bin/env bash
# List, search, and inspect available SDG flows.
#
# Usage: sdg_flows.sh <action> [args]
# Actions:
#   list              — list all registered flows
#   search <tag>      — search flows by tag
#   inspect <flow>    — show flow details (blocks, required columns)
# Output: JSON for each action.
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"

usage() {
    echo "Usage: sdg_flows.sh [ACTION] [ARGS]"
    echo ""
    echo "Actions:"
    echo "  list               List all available flows"
    echo "  search <tag>       Search flows by tag"
    echo "  inspect <flow>     Show detailed flow information"
    exit 1
}

die() { echo "ERROR: $1" >&2; exit 1; }

ACTION="${1:-list}"

case "$ACTION" in
    list)
        $PYTHON -c "
from sdg_hub import FlowRegistry
import json

flows = FlowRegistry.list_flows()
result = []
for flow in flows:
    entry = {'id': flow['id'], 'name': flow['name']}
    meta = FlowRegistry.get_flow_metadata(flow['id'])
    if meta:
        if meta.description:
            entry['description'] = meta.description
        if meta.tags:
            entry['tags'] = meta.tags
    result.append(entry)

print(json.dumps(result, indent=2))
"
        ;;

    search)
        [ -z "${2:-}" ] && die "No search query provided. Usage: sdg_flows.sh search <tag>"
        SDG_QUERY="$2" $PYTHON -c "
from sdg_hub import FlowRegistry
import json, os

query = os.environ['SDG_QUERY']
flows = FlowRegistry.search_flows(tag=query)
result = []
for flow in flows:
    entry = {'id': flow['id'], 'name': flow['name']}
    meta = FlowRegistry.get_flow_metadata(flow['id'])
    if meta:
        if meta.description:
            entry['description'] = meta.description
        if meta.tags:
            entry['tags'] = meta.tags
    result.append(entry)

print(json.dumps(result, indent=2))
"
        ;;

    inspect)
        [ -z "${2:-}" ] && die "No flow name provided. Usage: sdg_flows.sh inspect <flow>"
        SDG_FLOW="$2" $PYTHON -c "
from sdg_hub import Flow, FlowRegistry
import json, os

flow_name = os.environ['SDG_FLOW']

if flow_name.endswith('.yaml') or flow_name.endswith('.yml'):
    flow = Flow.from_yaml(flow_name)
else:
    flow_path = FlowRegistry.get_flow_path_safe(flow_name)
    flow = Flow.from_yaml(flow_path)

info = {
    'name': flow_name,
    'blocks': [type(b).__name__ for b in flow.blocks] if hasattr(flow, 'blocks') else [],
}

if hasattr(flow, 'description'):
    info['description'] = flow.description
if hasattr(flow, 'required_columns'):
    info['required_columns'] = flow.required_columns

print(json.dumps(info, indent=2, default=str))
"
        ;;

    --help)
        usage
        ;;

    *)
        die "Unknown action: $ACTION. Use list, search, or inspect."
        ;;
esac
