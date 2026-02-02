# Connectors Architecture - Simplification Complete

## Summary

**Before:** 1,301 lines for essentially one Langflow HTTP connector.
**After:** 812 lines (38% reduction, 489 lines removed).

---

## Changes Made

### 1. Exceptions (110 → 41 lines, -63%)

**Removed:**
- `ConnectorConfigError`
- `ConnectorConnectionError`
- `ConnectorTimeoutError`
- `ConnectorResponseError`

**Kept:**
- `ConnectorError` (base, used for all error types)
- `ConnectorHTTPError` (with status_code attribute)

### 2. Registry (324 → 119 lines, -63%)

**Removed:**
- `ConnectorMetadata` dataclass
- Category indexing (`_by_category`)
- `list_by_category()`, `categories()` methods
- `discover()` method with Rich table display
- `get_metadata()` method
- Typo suggestions using `difflib.get_close_matches`

**Kept:**
- Simple `_connectors` dict
- `register()` decorator
- `get()` method (with available connectors in error)
- `list_all()` method
- `clear()` method

### 3. BaseConnector (175 → 95 lines, -46%)

**Removed:**
- `warm_up()` pattern
- Capability flags (`supports_async`, `supports_streaming`, `supports_batch`)
- `_is_warmed_up` state tracking
- `_cleanup_client()` method
- Context manager (`__enter__`/`__exit__`)
- `close()` method
- `is_ready` property

**Kept:**
- `ConnectorConfig` with validation
- `execute()` abstract method
- `aexecute()` async wrapper

### 4. HttpClient (221 → 129 lines, -42%)

**Removed:**
- `_create_retry_decorator()` method
- `_handle_error()` method
- `_post_async_impl()` method

**Kept:**
- Inline tenacity `@retry` decorator
- `post()` async method
- `post_sync()` sync wrapper
- Error handling inline

### 5. BaseAgentConnector (242 → 228 lines, -6%)

**Removed:**
- `warm_up()` calls
- `supports_async` class variable

**Changed:**
- `_initialize_client()` → `_get_http_client()` (lazy init)
- Uses `ConnectorError` instead of `ConnectorResponseError`

### 6. LangflowConnector (146 → 138 lines, -5%)

**Changed:**
- Simplified `@ConnectorRegistry.register("langflow")` (no metadata)
- Uses `ConnectorError` instead of `ConnectorResponseError`

---

## Final Line Counts

| File | Before | After | Change |
|------|--------|-------|--------|
| `__init__.py` | 67 | 46 | -31% |
| `base.py` | 175 | 95 | -46% |
| `registry.py` | 324 | 119 | -63% |
| `exceptions.py` | 110 | 41 | -63% |
| `http/__init__.py` | 6 | 6 | 0% |
| `http/client.py` | 221 | 129 | -42% |
| `agent/__init__.py` | 10 | 10 | 0% |
| `agent/base.py` | 242 | 228 | -6% |
| `agent/langflow.py` | 146 | 138 | -5% |
| **Total** | **1,301** | **812** | **-38%** |

---

## Tests

All 75 tests pass:
- 57 connector tests
- 18 agent block tests

---

## What We Kept (Not YAGNI)

1. **Registry** - Will add more connectors
2. **HttpClient** - Reusable HTTP logic with tenacity retry
3. **BaseConnector** - Common interface for all connector types
4. **BaseAgentConnector** - Base for agent framework connectors
5. **2 Exception classes** - One general, one for HTTP errors

---

## Decision

- [x] Refactor to lean architecture (YAGNI-compliant) ✅

---

## References

- YAGNI Principle: https://martinfowler.com/bliki/Yagni.html
- Rule of Three: https://en.wikipedia.org/wiki/Rule_of_three_(computer_programming)
