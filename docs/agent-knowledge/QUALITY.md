# Quality Grades

Last updated: 2026-05-11

Quality grades by domain. Grades: A (≥90%), B (80–89%), C (70–79%), D (60–69%), F (<60%).

**Composite score (eval/score.py): 0.29** — 955 tests passed, 16 failed, 90% line coverage.

| Domain | Test Coverage | Lint | Types | Docs | Overall |
|---|---|---|---|---|---|
| blocks/llm | B (84%) | A | A | A | B |
| blocks/parsing | A (92%) | A | A | A | A |
| blocks/transform | A (95%) | A | A | A | A |
| blocks/filtering | B (87%) | A | A | A | B |
| blocks/agent | A (90%) | A | A | A | B (1 test failure) |
| blocks/mcp | B (85%) | A | A | A | B |
| blocks/code | B (89%) | A | A | A | B |
| flow/ | A (90%) | A | A | A | A |
| connectors/ | B (80%) | A | A | A | C (15 test failures) |
| utils/ | A (93%) | A | A | A | A |

## Failing Tests (16 total)

**blocks/agent (1):** `test_generate_async_mode_from_async_context` — async event-loop issue.

**connectors (15):**

- `test_base.py`: 2 async failures (`test_send_async_no_url_raises_error`, `test_send_async_full_flow`)
- `test_langgraph.py`: 8 failures — async `send_async` and error-path tests
- `test_monty.py`: 3 failures — `aexecute_code` async tests
- `test_client.py`: 1 failure — `test_post_async`
- `test_base.py (connectors)`: 1 failure — `test_aexecute`

Root cause: async connector tests failing, likely event-loop or mock configuration issue.
