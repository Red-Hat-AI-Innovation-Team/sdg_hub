# Quality Grades

Last updated: 2026-05-11

Quality grades by domain. Grades are populated when the quality grading
Autopilot runs (Phase 4).

**Composite score: 0.955** (eval/score.py)
**971 tests passing, 90% overall coverage**

| Domain | Test Coverage | Lint | Types | Docs | Overall |
|---|---|---|---|---|---|
| blocks/llm | B (84%) | A | A | B | B |
| blocks/parsing | A (91%) | A | A | C | B |
| blocks/transform | A (95%) | A | A | C | B |
| blocks/filtering | B (87%) | A | A | C | B |
| blocks/agent | A (91%) | A | A | B | A |
| blocks/mcp | B (85%) | A | A | A | A |
| blocks/code | B (88%) | A | A | A | A |
| flow/ | B (89%) | A | A | A | A |
| connectors/ | A (91%) | A | A | A | A |
| utils/ | A (93%) | A | A | A | A |

### Grade Scale

- **A**: >= 90% coverage / all checks pass / full docstring coverage
- **B**: >= 80% coverage / minor gaps in docs
- **C**: >= 70% coverage / significant doc gaps

### Notes

- Lint: `ruff check` passes clean across all domains
- Types: `mypy` passes clean across all domains
- Docs gaps: blocks/transform (7 classes missing docstrings), blocks/filtering
  (class docstring missing), blocks/parsing (3 methods missing docstrings)
