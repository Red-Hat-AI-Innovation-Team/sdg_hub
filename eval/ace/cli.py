# SPDX-License-Identifier: Apache-2.0
"""CLI entrypoint for managing ACE playbooks.

Usage:
  uv run python -m eval.ace.cli show <role>
  uv run python -m eval.ace.cli add <role> DO "rule"
  uv run python -m eval.ace.cli add <role> DONT "rule"
  uv run python -m eval.ace.cli record <role> <id> kept
  uv run python -m eval.ace.cli record <role> <id> reverted
  uv run python -m eval.ace.cli curate <role>
  uv run python -m eval.ace.cli init
"""

from __future__ import annotations

import argparse
import sys

from eval.ace.playbook import (
    add_entry,
    curate,
    format_for_prompt,
    init_playbooks,
    load_playbook,
    record_outcome,
)


def cmd_show(args: argparse.Namespace) -> None:
    entries = load_playbook(args.role)
    if not entries:
        print(f"No playbook found for role '{args.role}'.")  # noqa: T201
        return
    print(format_for_prompt(args.role))  # noqa: T201


def cmd_add(args: argparse.Namespace) -> None:
    entry = add_entry(args.role, args.category, args.rule)
    print(f"Added [{entry.id}] to {args.role} playbook: {entry.rule}")  # noqa: T201


def cmd_record(args: argparse.Namespace) -> None:
    try:
        record_outcome(args.role, args.id, args.outcome)
        print(  # noqa: T201
            f"Recorded '{args.outcome}' for [{args.id}] in {args.role} playbook."
        )
    except KeyError as exc:
        print(f"Error: {exc}", file=sys.stderr)  # noqa: T201
        sys.exit(1)


def cmd_curate(args: argparse.Namespace) -> None:
    curated = curate(args.role)
    print(  # noqa: T201
        f"Curated {args.role} playbook: {len(curated)} entries remaining."
    )


def cmd_init(_args: argparse.Namespace) -> None:
    created = init_playbooks()
    if created:
        print(f"Created starter playbooks for: {', '.join(created)}")  # noqa: T201
    else:
        print("All playbooks already exist, nothing created.")  # noqa: T201


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="ace",
        description="ACE Playbook manager -- self-evolving agent behavioral rules",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # show
    p_show = subparsers.add_parser("show", help="Show current playbook for a role")
    p_show.add_argument("role", help="Role name (e.g., builder, evaluator)")
    p_show.set_defaults(func=cmd_show)

    # add
    p_add = subparsers.add_parser("add", help="Add a new rule to a playbook")
    p_add.add_argument("role", help="Role name")
    p_add.add_argument("category", choices=["DO", "DONT"], help="Rule category")
    p_add.add_argument("rule", help="Rule text")
    p_add.set_defaults(func=cmd_add)

    # record
    p_record = subparsers.add_parser("record", help="Record an outcome for a rule")
    p_record.add_argument("role", help="Role name")
    p_record.add_argument("id", help="Entry ID (e.g., build-0001)")
    p_record.add_argument("outcome", choices=["kept", "reverted"], help="Outcome type")
    p_record.set_defaults(func=cmd_record)

    # curate
    p_curate = subparsers.add_parser(
        "curate", help="Prune net-negative rules and deduplicate"
    )
    p_curate.add_argument("role", help="Role name")
    p_curate.set_defaults(func=cmd_curate)

    # init
    p_init = subparsers.add_parser(
        "init", help="Create starter playbooks for all roles"
    )
    p_init.set_defaults(func=cmd_init)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
