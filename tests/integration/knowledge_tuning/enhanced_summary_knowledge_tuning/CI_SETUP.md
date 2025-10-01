# CI Setup for Enhanced Summary Integration Tests

This document explains how to set up GitHub Actions to run the enhanced summary knowledge tuning integration tests.

## GitHub Secrets Setup

### 1. Navigate to Repository Settings

Go to your repository on GitHub:
```
https://github.com/Red-Hat-AI-Innovation-Team/sdg_hub
```

### 2. Add Repository Secrets

1. Click **Settings** (top right)
2. In the left sidebar, click **Secrets and variables** → **Actions**
3. Click **New repository secret**

### 3. Required Secrets

Add the following secret:

| Secret Name | Description | Example Value |
|------------|-------------|---------------|
| `OPENAI_API_KEY` | OpenAI API key for running tests | `sk-...` |

**How to get an OpenAI API key:**
1. Go to https://platform.openai.com/api-keys
2. Click "Create new secret key"
3. Copy the key and paste it into the GitHub secret

### 4. Cost Considerations

The integration tests use `gpt-4o-mini` which is very cost-effective:
- **Per test run**: ~$0.05 - $0.10
- **Setting**: `NUMBER_OF_SUMMARIES=3` (kept small for tests)
- **Validation set**: Only 1-2 documents

**Monthly estimates:**
- 100 test runs/month: ~$5-10
- 500 test runs/month: ~$25-50

## Environment Variables

The tests automatically configure these for CI:

```yaml
MODEL_PROVIDER: openai
OPENAI_MODEL: openai/gpt-4o-mini
NUMBER_OF_SUMMARIES: 3
RUN_ON_VALIDATION_SET: true
```

## Workflow Configuration

The integration test workflow (`.github/workflows/integration-test.yml`) is already configured to:

1. ✅ Trigger on changes to enhanced_summary example
2. ✅ Pass `OPENAI_API_KEY` from secrets to tests
3. ✅ Run tests via tox: `tox -e py3-integrationcov`

## Local Testing (No GitHub Secrets Needed)

For local development:

1. Create `.env` file:
```bash
cd examples/knowledge_tuning/enhanced_summary_knowledge_tuning
cp .env.example .env
# Edit .env with your API key
```

2. Run tests:
```bash
pytest tests/integration/knowledge_tuning/enhanced_summary_knowledge_tuning/ -v -m integration
```

The test fixtures automatically detect whether you're running locally (reads `.env`) or in CI (reads GitHub secrets).

## Security Notes

- ✅ Secrets are **never** exposed in logs
- ✅ Secrets are **only** available to workflows in the main repository (not forks)
- ✅ API keys can be rotated at any time by updating the secret
- ⚠️  Only repository maintainers can add/view secrets

## Troubleshooting

### Tests failing in CI with "API key not found"

1. Verify secret is set: Settings → Secrets and variables → Actions
2. Check secret name matches exactly: `OPENAI_API_KEY`
3. Ensure workflow has `env:` block passing the secret

### Tests passing locally but failing in CI

1. Check if `.env` file has different config than CI defaults
2. Verify notebook paths are correct (relative paths can differ)
3. Check if any local-only dependencies or files are being used

## Alternative: Using Mock LLMs (Future)

To reduce API costs, we can later add mock LLM responses:

```python
@pytest.fixture
def mock_llm():
    with patch('litellm.completion') as mock:
        # Return deterministic responses
        mock.return_value = ...
        yield mock
```

This would eliminate API costs entirely but requires maintaining mock responses.
