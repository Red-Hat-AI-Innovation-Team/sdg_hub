# Grading Criteria

Agents grade their own and each other's work against these criteria.
Each criterion has a hard threshold — if any fails, the change is
rejected with actionable feedback.

## Criteria

| Criterion | What it measures | Threshold |
|-----------|-----------------|-----------|
| **Correctness** | Does the block/flow produce expected output for known inputs? | Hard fail if wrong |
| **Composability** | Does it integrate into the block/flow/connector system cleanly? | Must follow registry pattern |
| **Test quality** | Are tests meaningful? Do they cover success and error cases? | >=80% coverage, both paths tested |
| **Documentation** | Is usage clear from docstrings and YAML metadata? | Public methods have docstrings |

## How to Grade

Check each criterion in order. Report the first failure with specific,
actionable feedback. Return PASS or NEEDS_WORK on the first line
(machine-parseable), followed by evidence.

## Calibration Examples

### Example 1: PASS

```
PASS

Evaluation of PR #742: fix agent tool use in mcp eval example

- Correctness: PASS — test_mcp_eval_flow.py passes. The agent now correctly
  sets force_tool_use=True in the MCP config. Verified by running the flow
  with mock data and confirming tool_use blocks appear in the output.
- Composability: PASS — no new blocks or registrations needed. Change is
  contained within the existing MCPAgentBlock configuration.
- Test quality: PASS — added test_agent_enforces_tool_use() that asserts
  tool_use is present in the response. Also added test_agent_without_tool_use()
  as a negative case. Coverage for mcp_agent_block.py is 87%.
- Documentation: PASS — updated the flow YAML comment explaining the
  force_tool_use parameter.
```

### Example 2: NEEDS_WORK (missing test coverage)

```
NEEDS_WORK

Evaluation of PR #645: add MessageFormatterBlock

- Correctness: PASS — block produces expected output for SFT and DPO formats.
- Composability: PASS — registered with @BlockRegistry.register("message_formatter",
  category="transform", description="...").
- Test quality: FAIL — tests/blocks/transform/test_message_formatter.py only
  tests the happy path (valid SFT input). Missing:
  * Test for empty DataFrame input (should return empty DataFrame, not crash)
  * Test for missing required columns (should raise MissingColumnError)
  * Test for DPO format (only SFT is tested)
  Coverage is 62% (below 80% threshold).
  See docs/agent-knowledge/testing-standards.md for test requirements.
- Documentation: PASS — generate() has a docstring with parameters and return type.
```

### Example 3: NEEDS_WORK (architecture violation)

```
NEEDS_WORK

Evaluation of PR #598: enforce column rename rules

- Correctness: PASS — validation catches duplicate column names.
- Composability: FAIL — the validation logic is implemented directly in
  flow/execution.py (line 245-280) instead of using the existing
  validation.py module. This violates the package layering rule in
  ARCHITECTURE.md: validation logic belongs in flow/validation.py, not
  execution.py.
  Fix: move the validation function to flow/validation.py and import it
  from execution.py.
- Test quality: PASS — good coverage of edge cases.
- Documentation: PASS — docstrings present.
```

### Example 4: NEEDS_WORK (catching self-praise)

```
NEEDS_WORK

Evaluation of PR: add SimilarityFilterBlock

- Correctness: FAIL — the block claims to use cosine similarity but the
  implementation at line 45 uses Jaccard distance instead. The test passes
  because it only checks that SOME filtering happened, not that the correct
  similarity metric was used.
  Fix: either change the implementation to use cosine similarity (as the
  docstring claims), or update the docstring and registration description
  to say Jaccard distance. Then add a test that verifies the specific
  similarity values.
- Composability: PASS — correctly registered.
- Test quality: FAIL — test asserts len(result) < len(input) which passes
  for any filtering. Need to assert specific rows are kept/removed based
  on known similarity scores.
- Documentation: FAIL — docstring says "cosine similarity" but implementation
  uses Jaccard.
```
