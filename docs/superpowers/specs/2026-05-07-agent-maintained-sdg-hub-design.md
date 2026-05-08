# Agent-Maintained SDG Hub: Harness Engineering Design

**Date:** 2026-05-07 (updated 2026-05-08)
**Author:** Shiv (human) + Claude Code (agent)
**Status:** Draft v2 — pending review

## 1. Vision

Transform SDG Hub from a human-developed library with agent assistance into a
**fully agent-maintained codebase** where Claude Code and auxiliary agents
autonomously write code, review PRs, fix bugs, manage documentation, enforce
architecture, prevent drift, and ship releases. Humans steer via prompts,
specs, and architecture decisions — but rarely touch code directly.

### Design Principles

1. **Humans steer. Agents execute.** Humans define what to build and why.
   Agents handle how.
2. **Every change is an experiment.** Measured before and after, kept only if
   quality improves. The codebase is a ratchet.
3. **What agents can't see doesn't exist.** All context must live in the
   repository — Slack threads, Google Docs, and tacit knowledge are invisible.
4. **Enforce boundaries centrally, allow autonomy locally.** Strict
   architectural invariants; freedom within those boundaries.
5. **Aspire to agent-written code, but don't enforce it.** Maximize agent
   leverage without dogmatic purity. Humans can still write code when faster.

### Scope

- **Internal work:** Full agent autonomy — write, review, merge.
- **External contributions:** Agents review and guide contributors; human
  maintainer approves the final merge.
- **Agent ecosystem:** Claude Code is the primary agent. Cursor workflows
  remain for specialized tasks (code review, CI fix, doc updates).

### Sources

This design synthesizes patterns from five reference implementations:

| Source | Key Contribution |
|--------|-----------------|
| [OpenAI: Harness Engineering](https://openai.com/index/harness-engineering/) | Knowledge base as table of contents, progressive disclosure, architecture enforcement, garbage collection, merge philosophy |
| [Anthropic: Harness Design for Long-Running Apps](https://www.anthropic.com/engineering/harness-design-long-running-apps) | Generator-evaluator separation, grading criteria, sprint contracts, self-evaluation bias |
| [anthropics/cwc-long-running-agents](https://github.com/anthropics/cwc-long-running-agents) | Evidence-gated verification, operator controls (kill switch, steer), PROGRESS.md handoff, commit-on-stop |
| [AndrewAltimit/template-repo](https://github.com/AndrewAltimit/template-repo) | Decision rubric with trust tiers, multi-profile reviews, iteration limits, agent board, review profiles |
| [akashgit/remote-factory](https://github.com/akashgit/remote-factory) | Experiment-as-unit-of-work, three-tier scoring, FEEC priority, self-evolving playbooks (ACE), CEO completion guard, mandatory archival, non-overridable precheck gate |
| [multica-ai/multica](https://github.com/multica-ai/multica) | Open-source managed agents platform. Self-hostable orchestration layer with issue-based task board, Autopilot scheduling, multi-agent coordination, session resumption, and skills system. Uses Claude Code CLI directly. |
| [Claude Agent SDK](https://code.claude.com/docs/en/agent-sdk/overview) | Claude Code as a Python/TypeScript library. Programmatic subagents with tool restrictions, hooks as Python callbacks, sessions with resume, custom tools via `@tool` decorator. Alternative to Managed Agents using your own infrastructure. |

---

## 2. Current State

SDG Hub already has a strong foundation for agent-driven development:

**Existing infrastructure:**

- CLAUDE.md (136 lines) with codebase orientation and CI requirements
- Custom Claude skill (430-line SKILL.md + 5 reference files)
- Claude Code GitHub Action (Opus 4.6, full permissions, shared org skills)
- Three Cursor agent workflows (code review, CI auto-fix, doc auto-update)
- Eight CI checks (ruff format, ruff lint, mypy, pytest, commitlint, lock
  sync, markdown lint, actionlint)
- Codecov at 80% target
- Pre-commit hooks (uv-lock, ruff, ruff-format, mypy, conventional commits)
- Dependabot (daily) + Mergify (auto-labeling, conflict detection)
- Worktree management via justfile for parallel agent work
- Three registries (blocks, flows, connectors) with decorator-based
  registration
- Flow regression test framework with auto-generated mocks
- Design spec pattern in `docs/superpowers/specs/`

**Gaps to close:**

- No structured knowledge base or ARCHITECTURE.md
- No architectural invariant enforcement (structural tests, custom lints)
- No agent-to-agent review loop or evaluator subagent
- No evidence-gated verification
- No operator controls (kill switch, steer)
- No experiment-based change management (before/after scoring)
- No drift detection or quality grading
- No PR/issue templates
- No session continuity mechanism (PROGRESS.md)
- No decision rubric for agent escalation
- Legacy config files (.isort.cfg, .pylintrc) that could confuse agents
- Two documentation systems (MkDocs + Next.js) creating drift risk

---

## 3. Phase 1 — The Map (Knowledge Base)

### 3.1 Restructure CLAUDE.md

Reduce CLAUDE.md from 136 lines to ~80-100 lines. It becomes a **table of
contents** — not an encyclopedia. Points agents to deeper sources of truth.

```
CLAUDE.md                          ← ~80-100 lines. Map only.
ARCHITECTURE.md                    ← Domain map, package layering,
                                      dependency direction rules.
```

CLAUDE.md structure:

1. **Project identity** (2-3 lines: what SDG Hub is, Python 3.10+)
2. **Development commands** (install, test, lint — keep as-is, these are
   high-frequency reference)
3. **Knowledge base pointers** ("If you're adding a block, read
   `docs/agent-knowledge/block-invariants.md`")
4. **CI requirements table** (keep as-is — agents need this for every PR)
5. **Agent workflow pointers** (link to decision rubric, review profiles,
   quality grades)

### 3.2 Create ARCHITECTURE.md

Top-level domain map following the
[ARCHITECTURE.md convention](https://matklad.github.io/2021/02/06/ARCHITECTURE.md.html).

Contents:

- **Domain map:** blocks, flows, connectors, registries, utils
- **Package layering:** base classes → registries → implementations → utils
- **Dependency direction rules:** implementations cannot import from other
  implementations; cross-cutting concerns (LLM access, agent config, logging)
  enter through explicit config interfaces
- **Data flow:** `dataset → Block₁ → Block₂ → ... → enriched_dataset`
- **Extension points:** how to add blocks, flows, connectors

### 3.3 Build Knowledge Base Directory

```
docs/agent-knowledge/
├── index.md                    ← Master index with verification dates
├── core-principles.md          ← Golden principles for SDG Hub
├── block-invariants.md         ← Rules every block must follow
├── flow-invariants.md          ← Rules every flow YAML must follow
├── connector-invariants.md     ← Rules every connector must follow
├── testing-standards.md        ← What "tested" means, coverage rules
├── grading-criteria.md         ← Quality criteria with thresholds
├── decision-rubric.md          ← When to auto-fix, flag, or escalate
├── QUALITY.md                  ← Quality grades per domain/layer
└── tech-debt-tracker.md        ← Known debt, prioritized
```

**Core principles** (the "golden principles" from the OpenAI article):

1. Prefer shared utility packages over hand-rolled helpers
2. Validate data shapes at boundaries (Pydantic models, not raw dicts)
3. Every block must be registered, tested, and documented
4. Flow YAMLs must have metadata (author, description, tags)
5. Structured logging only — no raw `print()`
6. Tests must assert meaningful output, not just "something was returned"

**Block invariants:**

- Must inherit from `BaseBlock`
- Must implement `generate()` method
- Must use Pydantic fields for configuration
- Must declare `input_cols` and `output_cols`
- Must be registered with `@BlockRegistry.register(name, category, description)`
- Must have a corresponding test file at
  `tests/blocks/{category}/test_{name}.py`

**Grading criteria** (adapted from Anthropic's frontend design criteria):

| Criterion | What it measures | Threshold |
|-----------|-----------------|-----------|
| **Correctness** | Does the block/flow produce expected output for known inputs? | Hard fail if wrong |
| **Composability** | Does it integrate into the block/flow/connector system cleanly? | Must follow registry pattern |
| **Test quality** | Are tests meaningful? Do they cover success and error cases? | ≥80% coverage, both paths |
| **Documentation** | Is usage clear from docstrings, YAML examples, flow metadata? | Public methods documented |

**Decision rubric** (adapted from template-repo):

| Confidence | Action |
|-----------|--------|
| High (>80%) | Auto-fix silently |
| Medium (50-80%) | Fix but flag for review with `needs-review` label |
| Low (<50%) | Escalate to human with `needs-human-review` label |

Trust tiers:

1. **Admin** (human maintainer) — can override anything
2. **CI/precheck gate** — non-overridable, even by admins in automated flows
3. **Agent reviewer** — findings must be validated (agents hallucinate)
4. **External contributor** — requires human approval for merge

### 3.4 Create PR and Issue Templates

**PR template** (`.github/PULL_REQUEST_TEMPLATE.md`):

```markdown
## Summary
<!-- 1-3 sentences: what changed and why -->

## Changes
<!-- Bulleted list of files changed and what was modified -->

## Test Plan
<!-- How was this tested? What evidence exists? -->

## Checklist
- [ ] Tests pass (`uv run pytest`)
- [ ] Structural tests pass (`uv run pytest tests/structural/`)
- [ ] Lint clean (`uv run ruff check src/ tests/`)
- [ ] Types clean (`uv run mypy src/sdg_hub`)
- [ ] Docs updated if public API changed
- [ ] No new lint warnings introduced

## Agent Metadata
<!-- Filled by agent PRs -->
- **Agent:** <!-- claude-code | cursor | human -->
- **Confidence:** <!-- high | medium | low -->
- **Auto-merge eligible:** <!-- yes | no -->
```

**Issue templates:** Bug report, feature request, new block request — each
structured so agents can parse and execute without ambiguity.

### 3.5 Execution Plans as Artifacts

```
docs/exec-plans/
├── active/                     ← In-progress plans
└── completed/                  ← Finished plans (institutional memory)
```

Before complex work, the agent writes a plan referencing specific knowledge
base docs. Plans are versioned and committed — they're the audit trail for
architectural decisions.

### 3.6 Legacy Cleanup

- Remove `.isort.cfg` (redundant with ruff isort config)
- Remove `.pylintrc` (redundant with ruff)
- Clean stale `__pycache__` directories for deleted test modules
- Add `.env.example` at root

---

## 4. Phase 2 — The Guardrails (Architecture Enforcement)

### 4.1 Enable Additional Ruff Rules

Add to `pyproject.toml` `[tool.ruff.lint]` select:

| Rule | Purpose |
|------|---------|
| `T201` | Ban `print()` — enforce structured logging |
| `D100-D107` | Require docstrings on public modules/classes/methods |
| `UP` | Enforce modern Python syntax (pyupgrade) |

### 4.2 Create Structural Tests

A new `tests/structural/` directory with plain pytest tests. No custom lint
plugins — just tests with descriptive assertion messages that inject
remediation instructions into agent context.

**`tests/structural/test_block_registration.py`:**

- Discover all classes inheriting `BaseBlock` in `src/sdg_hub/core/blocks/`
- Assert each is registered in `BlockRegistry`
- Assert message: "Block '{name}' is not registered. Add
  `@BlockRegistry.register(...)` following
  `docs/agent-knowledge/block-invariants.md`"

**`tests/structural/test_block_coverage.py`:**

- For each registered block, assert a test file exists at
  `tests/blocks/{category}/test_{name}.py`
- Assert message: "Block '{name}' has no test file. Create
  `tests/blocks/{category}/test_{name}.py` following
  `docs/agent-knowledge/testing-standards.md`"

**`tests/structural/test_architecture.py`:**

- Verify no cross-layer imports violating dependency direction
- Verify no implementation importing from another implementation directly
- Verify file size limits (no Python file > 500 lines)

**`tests/structural/test_flow_schemas.py`:**

- Use `FlowRegistry.discover_flows()` to find all built-in flows
- Validate each against the flow YAML schema
- Assert metadata fields present (author, description, tags)

### 4.3 Knowledge Base Validation CI Job

A new GitHub Actions workflow (`knowledge-validation.yml`) that:

- Validates all cross-links in `docs/agent-knowledge/` resolve
- Checks `index.md` verification dates — warn if >90 days stale
- Runs on push/PR when `docs/**` changes

### 4.4 Three-Tier Composite Scoring

Adapted from remote-factory's evaluation system, tailored to SDG Hub:

**Tier 1: Hygiene (weight 0.30)**

| Dimension | Weight | Measurement |
|-----------|--------|-------------|
| Tests | 0.30 | pytest exit code + coverage % |
| Lint | 0.15 | ruff check exit code |
| Type check | 0.10 | mypy exit code |
| Coverage | 0.25 | Codecov patch % |
| Structural | 0.10 | structural test pass rate |
| Commit format | 0.10 | commitlint pass |

**Tier 2: Growth (weight 0.20)**

| Dimension | Weight | Measurement |
|-----------|--------|-------------|
| Capability surface | 0.30 | Count of registered blocks + flows + connectors |
| Test diversity | 0.25 | Ratio of test categories covered |
| Doc completeness | 0.25 | Public methods with docstrings % |
| Experiment diversity | 0.20 | Variety of change types (feat/fix/refactor) |

**Tier 3: Project (weight 0.50)**

| Dimension | Weight | Measurement |
|-----------|--------|-------------|
| Flow regression | 0.40 | All flow regression tests passing |
| Block correctness | 0.30 | All block unit tests passing with assertions |
| Connector health | 0.15 | All connector tests passing |
| Integration | 0.15 | Integration tests passing (when API keys available) |

**Scoring script:** `eval/score.py` — outputs `{"score": 0.0-1.0}` for
before/after comparison.

### 4.5 Non-Overridable Precheck Gate

Adapted from remote-factory. Six checks that **no agent can bypass**:

1. **Score direction** — composite score must not regress below threshold
   (default 0.8)
2. **Scope guard** — changed files must be within declared scope
3. **Anti-pattern detection** — if >60% similar to a previously reverted
   experiment, block
4. **Smoke test** — `uv run pytest tests/blocks tests/flow -x -q` must pass
5. **Structural test pass** — all `tests/structural/` tests must pass
6. **Lint clean** — no new ruff or mypy errors

---

## 5. Phase 3 — The Agent Loop (Multica-Orchestrated)

### 5.1 Orchestration Platform: Multica

Instead of building custom orchestration from scratch, we use
[Multica](https://github.com/multica-ai/multica) as the managed agents
platform. Multica is self-hosted (Docker Compose + PostgreSQL), uses our
existing Claude Code CLI subscription, and provides:

- **Issue-based task board** where agents and humans share the same workflow
- **Autopilot** for scheduled/triggered autonomous agent work
- **Session resumption** — same (agent, issue) pairs auto-restore context
- **Skills** injected into agent context via native `.claude/skills/` paths
- **Agent-to-agent communication** with anti-loop protections
- **Daemon** that spawns Claude Code processes with isolated workspaces

**Multica handles logistics. Our harness handles quality.**

### 5.2 Multica Setup

**Self-host infrastructure:**

```bash
curl -fsSL https://raw.githubusercontent.com/multica-ai/multica/main/scripts/install.sh | bash -s -- --with-server
multica setup self-host
```

Services: PostgreSQL 17 (pgvector), Go backend, Next.js frontend.

**Agent roster (configured in Multica):**

| Agent Name | Backend | Skills | Role |
|-----------|---------|--------|------|
| `sdg-builder` | Claude Code (Opus 4.6) | synthetic-data-generation, block-invariants, testing-standards | Implements features, blocks, flows, connectors |
| `sdg-evaluator` | Claude Code (Opus 4.6) | grading-criteria, testing-standards | Reviews builder output with fresh context |
| `sdg-reviewer` | Claude Code (Sonnet 4.6) | block-invariants, flow-invariants, connector-invariants | Reviews code for architectural compliance |
| `sdg-gardener` | Claude Code (Sonnet 4.6) | core-principles, tech-debt-tracker | Daily maintenance: doc gardening, quality grading, drift detection |

**Skills (configured in Multica Settings → Skills):**

Import existing SDG Hub skills + create new ones from the knowledge base:
- `synthetic-data-generation` — existing 430-line SKILL.md
- `block-invariants` — from `docs/agent-knowledge/block-invariants.md`
- `flow-invariants` — from `docs/agent-knowledge/flow-invariants.md`
- `grading-criteria` — from `docs/agent-knowledge/grading-criteria.md`
- `testing-standards` — from `docs/agent-knowledge/testing-standards.md`
- `core-principles` — from `docs/agent-knowledge/core-principles.md`

### 5.3 Agent Loop (Multica-Orchestrated)

```
Human creates issue in Multica (or Autopilot triggers one)
  │
  ▼
Multica assigns issue to sdg-builder agent
  │
  ├─ Daemon provisions isolated workspace
  ├─ Injects CLAUDE.md, skills, issue context
  ├─ Spawns Claude Code with --permission-mode bypassPermissions
  │
  ▼
sdg-builder reads issue, implements in repo
  │
  ├─ Reads CLAUDE.md → routes to relevant knowledge base docs
  ├─ Writes code + tests
  ├─ Runs structural tests + unit tests locally
  ├─ Iterates until passing
  ├─ Opens PR via gh CLI (using PR template)
  ├─ Posts results as Multica issue comment
  ├─ @mentions sdg-evaluator for review
  │
  ▼
sdg-evaluator receives @mention, starts fresh session
  │
  ├─ Fresh context (no build history bias)
  ├─ Reads spec / acceptance criteria from issue
  ├─ Runs git diff against baseline
  ├─ Runs tests independently
  ├─ Grades against criteria: correctness, composability,
  │   test quality, documentation
  ├─ Posts verdict as issue comment:
  │     PASS — with evidence citations
  │     NEEDS_WORK — with specific, actionable findings
  │
  ▼
If NEEDS_WORK: sdg-evaluator @mentions sdg-builder
  │
  ├─ Builder resumes session (Multica auto-restores context)
  ├─ Iterates on feedback
  ├─ Max 5 iterations (tracked via issue comments)
  │
  ▼
CI runs on PR (8 checks + structural tests)
  │
  ├─ Cursor code review (existing workflow)
  ├─ Precheck gate (non-overridable):
  │     score direction, scope, anti-pattern, smoke test,
  │     structural tests, lint clean
  │
  ▼
Merge Decision
  │
  ├─ Internal PR + all gates pass → auto-merge
  ├─ External PR → human maintainer approval
  ├─ Any gate fails → agent comments on issue with failure details
  │
  ▼
Post-Merge
  │
  ├─ Cursor docs agent updates documentation
  ├─ Archivist records outcome to .factory/archive/
  ├─ Issue moved to "Done" in Multica
  └─ Quality grades updated
```

### 5.4 Evidence-Gated Verification

Adapted from cwc-long-running-agents. Two Claude Code hooks (configured in
`.claude/settings.json` and injected into Multica workspaces via skills):

**`track-read.sh`** (PreToolUse on Read):
Records when the agent reads evidence files (test output, coverage reports,
flow execution results). Appends paths to `.claude/.evidence-reads`.

**`verify-gate.sh`** (PreToolUse on Write|Edit):
When the agent tries to update results or mark a task complete, checks that
`.claude/.evidence-reads` has content. Blocks the write if empty: "No test
output or execution evidence has been Read this session."

This prevents agents from claiming "tests pass" without actually reading the
test output.

### 5.5 Operator Controls

**Via Multica UI/CLI (primary):**

- **Abort task:** `multica task abort <task-id>` — stops the agent mid-run
- **Steer via comment:** Post a comment on the Multica issue — agent reads
  it on next session resumption or @mention
- **Reassign:** Move issue to a different agent or back to human

**Via file-based hooks (backup, for local Claude Code sessions):**

- **Kill switch** (`kill-switch.sh`, PreToolUse on `*`): If `AGENT_STOP`
  file exists, blocks every tool call.
- **Steer file** (`steer.sh`, PreToolUse on `*`): If `STEER.md` has
  content, injects it once into agent context.

### 5.6 Session Continuity

**Multica handles this natively.** Session resumption is automatic per
(agent, issue) pair — the daemon reuses the working directory and passes
`--resume <session_id>` to Claude Code, preserving full conversation context.

**commit-on-stop hook** (Stop hook) remains as a safety net:
Auto-commits tracked changes at session end.

### 5.7 Auto-Merge Workflow

New GitHub Action (`auto-merge.yml`):

Triggers: when all CI checks pass on a PR.

Conditions for auto-merge:
- PR author is `claude-code-action[bot]` or agent-created
- All CI checks pass (8 existing + structural tests + knowledge validation)
- Evaluator agent posted `PASS` verdict on the Multica issue
- No `needs-human-review` label
- PR is not from a fork (external contribution)

If all conditions met: approve and merge via `gh pr merge --auto --squash`.
Multica issue auto-transitions to "Done" on merge (via webhook or Autopilot).

### 5.8 Iteration Limits

- Max **5 iterations** per agent per issue for the review-fix loop
  (tracked via Multica issue comments)
- Max **3 iterations** for the CI-failure-fix loop
- Human maintainer can post `[CONTINUE]` comment to extend by 5 iterations
- Multica's built-in retry mechanism handles agent task failures

---

## 6. Phase 4 — Entropy Management

### 6.1 Daily Recurring Agent Tasks

Implemented as **Multica Autopilots** (replacing GitHub Action crons for
agent-driven tasks). Each Autopilot creates an issue, assigns it to
`sdg-gardener`, and follows the normal task lifecycle.

| Task | Autopilot Trigger | Agent | Concurrency |
|------|-------------------|-------|-------------|
| Doc gardening | Cron: daily 06:00 UTC | sdg-gardener | `skip` (deduplicate) |
| Quality grading | Cron: daily 07:00 UTC | sdg-gardener | `skip` |
| Dead code scan | Cron: daily 08:00 UTC | sdg-gardener | `skip` |
| Pattern drift detection | Cron: daily 09:00 UTC | sdg-gardener | `skip` |
| Dependency hygiene | Daily | Dependabot (existing, stays in GitHub) |
| Flow regression | Every PR | CI pytest (existing, stays in GitHub) |

**Autopilot configuration example (doc gardening):**

```
Name: SDG Hub Doc Gardening
Agent: sdg-gardener
Trigger: schedule (0 6 * * *)
Mode: create_issue
Issue title template: "chore: daily doc gardening — {{date}}"
Prompt: "Scan docs/agent-knowledge/ for stale docs (>90 days since
  last verified). Check all cross-links resolve. Open fix-up PRs for
  any issues found. Update QUALITY.md doc completeness grades."
Concurrency: skip
```

CI-only checks (ruff, mypy, pytest, commitlint, etc.) remain as GitHub
Actions — Multica handles agent-driven work, GitHub handles deterministic
CI.

### 6.2 Quality Grades

`docs/agent-knowledge/QUALITY.md` — updated daily by the quality grading
agent:

```markdown
# Quality Grades — SDG Hub

Last updated: 2026-05-07

| Domain | Test Coverage | Lint | Types | Structural | Docs | Overall |
|--------|-------------|------|-------|-----------|------|---------|
| blocks/llm | 85% | ✅ | ✅ | ✅ | ✅ | A |
| blocks/parsing | 78% | ✅ | ✅ | ✅ | ⚠️ | B+ |
| blocks/transform | 92% | ✅ | ✅ | ✅ | ✅ | A+ |
| blocks/agent | 70% | ✅ | ⚠️ | ✅ | ⚠️ | B |
| blocks/mcp | 65% | ✅ | ✅ | ⚠️ | ⚠️ | B- |
| flow/ | 80% | ✅ | ✅ | ✅ | ✅ | A |
| connectors/ | 70% | ✅ | ✅ | ⚠️ | ⚠️ | B |
| utils/ | 75% | ✅ | ✅ | ✅ | ✅ | A- |
```

Agents reference this to prioritize improvement work. Trends tracked over
time.

### 6.3 Anti-Pattern Detection

Adapted from remote-factory's FEEC stuck detection:

- If a daily scan finds >3 occurrences of the same pattern violation, escalate
  to human
- If an agent PR is reverted and a subsequent PR attempts a >60% similar
  change, block it automatically
- Track revert rate per agent — if >30% of PRs from an agent are reverted,
  pause auto-merge for that agent and alert human

### 6.4 Institutional Memory

```
.factory/archive/
├── experiments/               ← Per-experiment: what was tried, outcome
├── strategies/                ← What approaches worked/failed
├── patterns/                  ← Recurring patterns discovered
└── performance_report.json    ← Aggregated stats
```

Agents query this before starting new work: "Has this approach been tried
before? Did it work?"

### 6.5 Self-Evolving Playbooks (ACE)

Adapted from remote-factory. Three-phase pipeline:

**Reflect:** After each experiment (keep or revert), analyze the outcome.
Generate candidate DO/DON'T rules for the builder, evaluator, and reviewer
agents.

**Curate:** Remove rules where `harmful > helpful` (minimum 3 observations).
Deduplicate rules >75% similar. Cap at top 15 rules per agent role.

**Inject:** At agent spawn time, append the evolved playbook to the agent's
prompt.

Example evolved playbook:

```markdown
## Behavioral Playbook (auto-evolved from experiment data)

### DO
- [build-0001] helpful=8 harmful=0 :: Always run ruff + mypy after changes
- [build-0002] helpful=5 harmful=1 :: Use existing block patterns in same
  category as reference
- [build-0003] helpful=4 harmful=0 :: Run flow regression tests before opening
  PR

### DON'T
- [build-0004] helpful=6 harmful=0 :: Don't add `type: ignore` comments —
  fix the actual type error
- [build-0005] helpful=3 harmful=0 :: Don't create new utility functions when
  an existing one in utils/ does the same thing
```

---

## 7. Phase 5 — Battle Testing and Tuning

### 7.1 Metrics to Track

| Metric | Target | Measurement |
|--------|--------|-------------|
| Agent PRs/day | ≥3 | GitHub API query |
| Agent PR merge rate | ≥80% | Merged / opened |
| Time to merge (agent PRs) | <2 hours | PR created → merged |
| Escalation rate | <20% | PRs with `needs-human-review` / total |
| Quality grade trend | Non-decreasing | Weekly QUALITY.md diff |
| Revert rate | <15% | Reverted / merged |
| Playbook evolution | Growing | ACE playbook entry count |

### 7.2 Tuning Protocol

1. **Week 1-2:** Run the full loop on low-risk tasks (doc updates, dead code
   cleanup, test improvements). Monitor metrics.
2. **Week 3-4:** Graduate to medium-risk (new blocks, flow improvements).
   Tune evaluator criteria based on false positives/negatives.
3. **Week 5+:** Full autonomy on all internal work. Resume feature
   development with the harness in place.
4. **Ongoing:** Review playbook evolution weekly. Prune stale rules. Add new
   criteria as failure modes emerge.

### 7.3 Failure Modes and Mitigations

| Failure Mode | Mitigation |
|-------------|-----------|
| Agent praises its own work (self-evaluation bias) | Separate evaluator subagent with read-only tools |
| Agent generates "AI slop" (generic, low-quality code) | Grading criteria with originality dimension + anti-pattern detection |
| Agent gets stuck in a loop | Max 5 iterations + FEEC stuck detection |
| Agent exits prematurely | Completion guard (count planned vs. completed) |
| Stale knowledge base misleads agents | Daily doc gardening + CI validation |
| Agent breaks something and auto-merges | Non-overridable precheck gate + revert capability |
| Evaluator is too lenient | Calibrate with few-shot examples; tune based on human review of evaluator logs |
| Context window fills up | Session continuity via PROGRESS.md + commit-on-stop |
| Human can't intervene mid-run | Kill switch + steer file |

---

## 8. Implementation Artifacts

### 8.0 Multica Infrastructure

```
# Self-hosted Multica (Docker Compose)
docker-compose.selfhost.yml        ← PostgreSQL 17, Go backend, Next.js frontend

# Multica configuration (in Multica UI/CLI)
4 agents:  sdg-builder, sdg-evaluator, sdg-reviewer, sdg-gardener
6 skills:  imported from docs/agent-knowledge/
4 autopilots: doc-gardening, quality-grading, dead-code-scan, pattern-drift
1 project resource: GitHub repo link to sdg_hub
```

### 8.1 New Files to Create

```
# Knowledge base
ARCHITECTURE.md
docs/agent-knowledge/index.md
docs/agent-knowledge/core-principles.md
docs/agent-knowledge/block-invariants.md
docs/agent-knowledge/flow-invariants.md
docs/agent-knowledge/connector-invariants.md
docs/agent-knowledge/testing-standards.md
docs/agent-knowledge/grading-criteria.md
docs/agent-knowledge/decision-rubric.md
docs/agent-knowledge/QUALITY.md
docs/agent-knowledge/tech-debt-tracker.md

# Execution plans
docs/exec-plans/active/.gitkeep
docs/exec-plans/completed/.gitkeep

# Structural tests
tests/structural/__init__.py
tests/structural/test_block_registration.py
tests/structural/test_block_coverage.py
tests/structural/test_architecture.py
tests/structural/test_flow_schemas.py

# Evaluation
eval/__init__.py
eval/score.py

# Templates
.github/PULL_REQUEST_TEMPLATE.md
.github/ISSUE_TEMPLATE/bug_report.md
.github/ISSUE_TEMPLATE/feature_request.md
.github/ISSUE_TEMPLATE/new_block.md
.github/ISSUE_TEMPLATE/config.yml

# Agent hooks
.claude/hooks/track-read.sh
.claude/hooks/verify-gate.sh
.claude/hooks/kill-switch.sh
.claude/hooks/steer.sh
.claude/hooks/commit-on-stop.sh

# Agent subagents
.claude/agents/evaluator.md

# Workflows (CI-only — agent-driven tasks handled by Multica Autopilots)
.github/workflows/auto-merge.yml
.github/workflows/knowledge-validation.yml

# Institutional memory
.factory/archive/.gitkeep
```

### 8.2 Files to Modify

```
CLAUDE.md                              ← Restructure to table of contents
pyproject.toml                         ← Add ruff rules (T201, D, UP)
.claude/settings.json                  ← Wire hooks
.claude/settings.local.json            ← Add hook permissions
```

### 8.3 Files to Delete

```
.isort.cfg                             ← Redundant with ruff
.pylintrc                              ← Redundant with ruff
```

---

## 9. Open Questions

1. **Evaluator model:** Should `sdg-evaluator` use the same model
   (Opus 4.6) or a different model for genuine independence? Using a
   different model adds cost but reduces shared blind spots.

2. **Auto-merge confidence:** Should auto-merge require evaluator PASS +
   all CI green, or should there be additional quality thresholds
   (e.g., composite score ≥ 0.9)?

3. **Playbook storage:** Should evolved playbooks live in the repo
   (`.factory/playbooks/`) or as Multica Skills (editable in UI)?
   Repo storage means playbooks are versioned; Multica Skills means
   they're editable without code changes.

4. **GitHub ↔ Multica sync:** Should Multica be the primary issue tracker
   (replacing GitHub Issues for internal work), or should there be
   bidirectional sync? Multica doesn't have native GitHub Issue sync.

5. **Cost budget:** Long-running agent sessions can be expensive. Multica
   tracks token usage per task — should there be a per-issue or per-day
   budget limit?

6. **Multica daemon hosting:** The daemon runs on a user's machine and
   spawns Claude Code. For 24/7 autonomous operation, where does the
   daemon run? Options: always-on dev machine, dedicated VM, or
   cloud instance.

---

## 10. Glossary

| Term | Definition |
|------|-----------|
| **ACE** | Autonomous Context Engineering — self-evolving agent playbooks |
| **Builder** | Agent that writes code and tests |
| **Evaluator** | Read-only agent that grades builder output |
| **FEEC** | Fix > Exploit > Explore > Combine priority heuristic |
| **Harness** | The scaffolding around agents that makes them effective |
| **Precheck gate** | Non-overridable quality checks before merge |
| **Progressive disclosure** | Agents start with a small entry point, load deeper docs as needed |
| **Ratchet** | Quality can only go up — changes that regress scores are reverted |
| **Sprint contract** | Agreement between generator and evaluator on what "done" looks like |
| **Steer file** | One-shot operator instruction injected into agent context |
