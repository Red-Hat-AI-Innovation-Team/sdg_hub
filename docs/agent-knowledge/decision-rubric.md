# Decision Rubric

When to auto-fix, flag for review, or escalate to a human.

## Confidence Thresholds

| Confidence | Range | Action |
|---|---|---|
| High | >80% | Auto-fix silently |
| Medium | 50-80% | Fix but add `needs-review` label |
| Low | <50% | Escalate with `needs-human-review` label |

**How to estimate confidence:** Confidence reflects how certain the agent is
that its proposed change is correct *and* complete. A formatting fix is high
confidence. A logic change inferred from test failures is medium. A
restructuring based on ambiguous requirements is low.

## Trust Tiers

| Tier | Actor | Authority |
|---|---|---|
| 1 | Admin (human maintainer) | Can override anything |
| 2 | CI / precheck gate | Non-overridable -- failures block merge |
| 3 | Agent reviewer | Findings must be validated (agents hallucinate) |
| 4 | External contributor | Requires human approval for all changes |

### Implications

- **Tier 1** decisions are final. If a human maintainer marks something as
  approved, agents must not re-flag it.
- **Tier 2** gates (CI, linting, type checking) cannot be bypassed. No agent
  or human override in the normal workflow.
- **Tier 3** agent findings are advisory. They must be validated by running
  tests or by a Tier 1 reviewer before acting on them. Never merge based
  solely on an agent's approval.
- **Tier 4** contributions always require Tier 1 review before merge,
  regardless of CI status.

## Always Escalate

The following situations **must always** be escalated to a human reviewer
(`needs-human-review` label), regardless of confidence level:

- **Public API changes** -- Any modification to a public method signature,
  return type, or removal of a public class/function.
- **Core base class modifications** -- Changes to `BaseBlock`,
  `BaseAgentConnector`, `Flow`, or registry classes.
- **Security-sensitive code** -- API key handling, authentication, file system
  access patterns, or network calls.
- **Flaky test uncertainty** -- When an agent cannot determine whether a test
  failure is a real regression or a flaky test.
- **Backward compatibility breaks** -- Any change that would break existing
  YAML flow definitions, block configurations, or connector interfaces.
- **5+ unresolved iterations** -- If an agent has attempted 5 or more
  fix-review cycles on the same issue without resolution, stop and escalate.
