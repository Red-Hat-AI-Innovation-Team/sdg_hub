# Quality Grades

Last updated: 2026-05-09

Composite score: **1.0000** (all tiers PASS)
Total test coverage: **90%** (971 tests, 4794 statements)

| Domain | Test Coverage | Lint | Types | Docs | Overall |
|---|---|---|---|---|---|
| blocks/llm | 84% | Pass | Pass | 96% | A |
| blocks/parsing | 92% | Pass | Pass | 40% | B |
| blocks/transform | 95% | Pass | Pass | 89% | A |
| blocks/filtering | 87% | Pass | Pass | 100% | A |
| blocks/agent | 91% | Pass | Pass | 67% | B |
| blocks/mcp | 85% | Pass | Pass | 100% | A |
| blocks/code | 89% | Pass | Pass | 100% | A |
| flow/ | 89% | Pass | Pass | 100% | A |
| connectors/ | 91% | Pass | Pass | 96% | A |
| utils/ | 93% | Pass | Pass | 100% | A |

## Notes

- **blocks/parsing**: Docs grade is low (40% public method docstrings). 3 public methods in `base_text_parser_block.py` and `json_parser_block.py` lack docstrings.
- **blocks/agent**: Docs grade is 67%. `agent_block.py` missing a public method docstring.
- **blocks/llm**: `error_handler.py` has only 33% test coverage, pulling the domain average down. Remaining LLM modules are 91–96%.
- **flow/display.py**: Lowest coverage in the flow domain at 63%.
- **connectors/code_interpreter/base.py**: Lowest coverage in connectors at 66%.
