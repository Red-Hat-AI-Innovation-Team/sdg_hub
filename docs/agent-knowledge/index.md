# Agent Knowledge Base

This directory is the system of record for agents working on the SDG Hub
codebase. Each file covers one topic end-to-end. Read only the file relevant
to your current task -- do not load all files at once.

## Files

| File | Topic | Last verified |
|------|-------|---------------|
| [core-principles.md](core-principles.md) | Golden rules that apply to all contributions | 2026-05-08 |
| [block-invariants.md](block-invariants.md) | Rules every block must follow | 2026-05-08 |
| [flow-invariants.md](flow-invariants.md) | Rules every flow YAML must follow | 2026-05-08 |
| [connector-invariants.md](connector-invariants.md) | Rules every connector must follow | 2026-05-08 |
| [testing-standards.md](testing-standards.md) | What "tested" means and how to write tests | 2026-05-08 |
| [grading-criteria.md](grading-criteria.md) | Quality criteria with hard thresholds | 2026-05-08 |
| [decision-rubric.md](decision-rubric.md) | When to auto-fix, flag, or escalate | 2026-05-08 |
| [QUALITY.md](QUALITY.md) | Quality grades per domain/layer | 2026-05-08 |
| [tech-debt-tracker.md](tech-debt-tracker.md) | Known debt, prioritized | 2026-05-08 |

## Progressive Disclosure

Agents should read the single file that matches their current task, not the
entire knowledge base. For example:

- Building a new block? Read `block-invariants.md`.
- Writing tests? Read `testing-standards.md`.
- Designing a flow? Read `flow-invariants.md`.
- Adding a connector? Read `connector-invariants.md`.
- Unsure about conventions? Start with `core-principles.md`.
- Reviewing code? Read `grading-criteria.md`.
- Deciding whether to fix or escalate? Read `decision-rubric.md`.
- Checking quality status? Read `QUALITY.md`.

Each file is self-contained. An agent reading just that one file will have
everything it needs for that topic.
