# Enhanced Summary Knowledge Tuning Integration Tests

Simple integration tests for the enhanced summary knowledge generation notebook.

## Test Strategy

These tests verify:
1. **Notebook conversion**: Notebook converts to Python script successfully
2. **Script execution**: Converted script runs without errors
3. **Output validation**: Generated datasets exist and are loadable

## Running Tests

### Local Testing

1. Set up your `.env` file:
```bash
cd examples/knowledge_tuning/enhanced_summary_knowledge_tuning
cp .env.example .env
# Edit .env with your API keys
```

2. Run the tests:
```bash
pytest tests/integration/knowledge_tuning/enhanced_summary_knowledge_tuning/ -v -m integration
```

### CI Testing (GitHub Actions)

Tests run automatically in CI using GitHub Secrets for API keys.

#### Required GitHub Secrets

Add these secrets to your repository (Settings → Secrets and variables → Actions):

- `OPENAI_API_KEY`: Your OpenAI API key for gpt-4o-mini

#### Optional Environment Variables

These can be set in the workflow or left as defaults:
- `MODEL_PROVIDER`: Default `openai`
- `OPENAI_MODEL`: Default `openai/gpt-4o-mini`
- `NUMBER_OF_SUMMARIES`: Default `3` (keep low for fast tests)
- `RUN_ON_VALIDATION_SET`: Default `true`

## Test Structure

```
tests/integration/knowledge_tuning/enhanced_summary_knowledge_tuning/
├── __init__.py
├── conftest.py                     # Fixtures and env setup
├── test_knowledge_generation.py    # Main integration tests
├── CI_SETUP.md                     # CI/CD setup guide
└── README.md                       # This file
```

## Environment Variable Priority

1. **CI**: GitHub Secrets → Environment variables
2. **Local**: `.env` file in example directory → Test defaults

The test fixture automatically handles both scenarios.
