# Grading Criteria

Quality criteria that agents grade against when evaluating blocks, flows,
connectors, and related contributions.

## Criteria Table

| Criterion | What it measures | Threshold |
|---|---|---|
| Correctness | Does the block/flow produce expected output for known inputs? | Hard fail if wrong |
| Composability | Does it integrate into the block/flow/connector system cleanly? | Must follow registry pattern |
| Test quality | Are tests meaningful? Do they cover success and error cases? | >=80% coverage, both paths tested |
| Documentation | Is usage clear from docstrings and YAML metadata? | Public methods have docstrings |

## How to Grade

Evaluate each criterion **in order**. Stop and report the **first failure**
with specific, actionable feedback. Do not continue grading past a failure --
fixing earlier criteria may resolve later ones.

### Step-by-step

1. **Correctness** -- Run the relevant test suite. If any test fails, or if the
   code produces wrong output for a known input, report a hard failure
   immediately.
2. **Composability** -- Verify the component follows the registry pattern:
   - Blocks use `@BlockRegistry.register(name, category, description)`.
   - Connectors use `@ConnectorRegistry.register("name")`.
   - Flows load via `Flow.from_yaml()` and chain correctly.
   - Input/output column contracts are respected.
3. **Test quality** -- Check coverage meets >=80% for the changed module. Both
   success and error/edge-case paths must have at least one test.
4. **Documentation** -- Every public method has a docstring. Block YAML
   metadata (name, category, description) is present and accurate.

### Reporting format

Report results using the following structure. Use `PASS` when a criterion is
met. Use `NEEDS_WORK` when a criterion fails -- include a specific reason and
a concrete suggestion.

### Example evaluator output

```text
## Grading: TagParserBlock

| Criterion | Result | Detail |
|---|---|---|
| Correctness | PASS | All 12 tests pass, outputs match expected for known tag patterns |
| Composability | PASS | Registered via @BlockRegistry.register, input_cols/output_cols respected |
| Test quality | NEEDS_WORK | Coverage is 74% -- missing error-path test for malformed tags |
| Documentation | -- | Not evaluated (blocked by prior failure) |

### Action required

Test quality: Add a test case for malformed/unclosed tags in
`tests/blocks/parsing/test_tag_parser_block.py`. Target the `_extract_tags`
method with input like `<tag>no closing tag`. This should bring coverage
above 80% and cover the error path.
```

When all criteria pass:

```text
## Grading: RenameColumnsBlock

| Criterion | Result | Detail |
|---|---|---|
| Correctness | PASS | All 8 tests pass |
| Composability | PASS | Registry pattern followed, column contracts valid |
| Test quality | PASS | 91% coverage, success and KeyError paths tested |
| Documentation | PASS | All public methods documented, YAML metadata complete |

Overall: PASS -- no action required.
```
