# Custom Python Scripts

Patterns for ad-hoc synthetic data generation without YAML flows.

## When to Use Custom Scripts

- Quick experiments
- One-off generation tasks
- Custom logic not suited for YAML
- Testing new ideas before creating a flow

## Basic Pattern

```python
# play.py
from sdg_hub.core.blocks import LLMChatBlock
import pandas as pd

# 1. Create block
block = LLMChatBlock(
    block_name="gen",
    input_cols="messages",
    output_cols="response",
    model="openai/gpt-4o-mini",
    api_key="sk-..."
)

# 2. Prepare data
df = pd.DataFrame({
    "messages": [[
        {"role": "user", "content": "Generate a fact."}
    ]]
})

# 3. Generate
result = block(df)
print(result["response"].iloc[0])
```

## Pattern: QA Generation

```python
# play.py
from sdg_hub.core.blocks import LLMChatBlock, TextParserBlock
import pandas as pd


def generate_qa_pairs(
    documents: list[str],
    model: str = "openai/gpt-4o-mini",
    api_key: str = None
) -> pd.DataFrame:
    """Generate QA pairs from documents."""

    # Build messages
    df = pd.DataFrame({
        "document": documents,
        "messages": [
            [
                {"role": "system", "content": "Generate question-answer pairs."},
                {"role": "user", "content": f"""Document: {doc}

Generate a question and answer based on this document.

Format:
Question: <question>
Answer: <answer>"""}
            ]
            for doc in documents
        ]
    })

    # Generate
    llm = LLMChatBlock(
        block_name="gen",
        input_cols="messages",
        output_cols="response",
        model=model,
        api_key=api_key,
        temperature=0.7
    )
    df = llm(df)

    # Parse
    parser = TextParserBlock(
        block_name="parse",
        input_cols="response",
        output_cols=["question", "answer"],
        pattern=r"Question:\s*(.+?)\s*Answer:\s*(.+)",
        flags="DOTALL"
    )
    df = parser(df)

    return df[["document", "question", "answer"]]


# Test
result = generate_qa_pairs(
    ["Python was created in 1991 by Guido van Rossum."],
    model="openai/gpt-4o-mini",
    api_key="sk-..."
)
print(result)
```

## Pattern: Batch Processing

```python
# play.py
from sdg_hub.core.blocks import LLMChatBlock
import pandas as pd
from tqdm import tqdm


def process_in_batches(
    df: pd.DataFrame,
    block: LLMChatBlock,
    batch_size: int = 50
) -> pd.DataFrame:
    """Process large dataset in batches with progress bar."""
    results = []

    for i in tqdm(range(0, len(df), batch_size), desc="Processing"):
        batch = df.iloc[i:i+batch_size].copy()
        result = block(batch)
        results.append(result)

    return pd.concat(results, ignore_index=True)


# Usage
block = LLMChatBlock(
    block_name="gen",
    input_cols="messages",
    output_cols="response",
    model="openai/gpt-4o-mini",
    api_key="sk-...",
    async_mode=True,
    max_concurrency=10
)

large_df = pd.DataFrame({
    "messages": [[{"role": "user", "content": f"Fact #{i}"}] for i in range(1000)]
})

result = process_in_batches(large_df, block, batch_size=50)
```

## Pattern: Multi-Turn Conversation

```python
# play.py
from sdg_hub.core.blocks import LLMChatBlock
import pandas as pd


def generate_conversation(
    topic: str,
    turns: int = 3,
    model: str = "openai/gpt-4o-mini",
    api_key: str = None
) -> list[dict]:
    """Generate multi-turn conversation."""

    block = LLMChatBlock(
        block_name="conv",
        input_cols="messages",
        output_cols="response",
        model=model,
        api_key=api_key
    )

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"Let's discuss {topic}. Ask me a question."}
    ]
    conversation = []

    for turn in range(turns):
        # Generate assistant response
        df = pd.DataFrame({"messages": [messages]})
        result = block(df)
        assistant_msg = result["response"].iloc[0]

        messages.append({"role": "assistant", "content": assistant_msg})
        conversation.append({"role": "assistant", "content": assistant_msg})

        if turn < turns - 1:
            # Simulate user response
            user_follow_up = f"That's interesting. Tell me more about aspect {turn + 1}."
            messages.append({"role": "user", "content": user_follow_up})
            conversation.append({"role": "user", "content": user_follow_up})

    return conversation


# Test
conv = generate_conversation("machine learning", turns=2)
for msg in conv:
    print(f"{msg['role']}: {msg['content'][:100]}...")
```

## Pattern: Data Augmentation

```python
# play.py
from sdg_hub.core.blocks import LLMChatBlock
import pandas as pd


def augment_text(
    texts: list[str],
    n_variations: int = 3,
    model: str = "openai/gpt-4o-mini",
    api_key: str = None
) -> pd.DataFrame:
    """Generate paraphrase variations of text."""

    block = LLMChatBlock(
        block_name="aug",
        input_cols="messages",
        output_cols="paraphrase",
        model=model,
        api_key=api_key,
        temperature=0.9  # Higher for diversity
    )

    rows = []
    for text in texts:
        for i in range(n_variations):
            rows.append({
                "original": text,
                "messages": [
                    {"role": "system", "content": "Paraphrase the text while preserving meaning."},
                    {"role": "user", "content": f"Paraphrase (variation {i+1}): {text}"}
                ]
            })

    df = pd.DataFrame(rows)
    result = block(df)

    return result[["original", "paraphrase"]]


# Test
augmented = augment_text(
    ["The quick brown fox jumps over the lazy dog."],
    n_variations=3
)
print(augmented)
```

## Pattern: Structured Extraction

```python
# play.py
from sdg_hub.core.blocks import LLMChatBlock
import pandas as pd
import json


def extract_entities(
    texts: list[str],
    entity_types: list[str],
    model: str = "openai/gpt-4o-mini",
    api_key: str = None
) -> pd.DataFrame:
    """Extract structured entities from text."""

    block = LLMChatBlock(
        block_name="extract",
        input_cols="messages",
        output_cols="response",
        model=model,
        api_key=api_key,
        temperature=0.0  # Deterministic
    )

    df = pd.DataFrame({
        "text": texts,
        "messages": [
            [
                {"role": "system", "content": "Extract entities and return valid JSON."},
                {"role": "user", "content": f"""Extract {entity_types} from:

{text}

Return JSON: {{"entities": [{{"type": "...", "value": "..."}}]}}"""}
            ]
            for text in texts
        ]
    })

    result = block(df)

    # Parse JSON responses
    def parse_json(s):
        try:
            return json.loads(s)
        except:
            return {"entities": []}

    result["entities"] = result["response"].apply(parse_json)
    return result[["text", "entities"]]


# Test
entities = extract_entities(
    ["Apple CEO Tim Cook announced new products in Cupertino."],
    entity_types=["person", "organization", "location"]
)
print(entities)
```

## Pattern: Chained Blocks

```python
# play.py
from sdg_hub.core.blocks import (
    LLMChatBlock,
    TextParserBlock,
    TextConcatBlock
)
import pandas as pd


def chain_blocks(df: pd.DataFrame, blocks: list) -> pd.DataFrame:
    """Chain multiple blocks together."""
    result = df.copy()
    for block in blocks:
        result = block(result)
    return result


# Build chain
blocks = [
    # Concatenate columns
    TextConcatBlock(
        block_name="concat",
        input_cols=["title", "body"],
        output_cols="full_text",
        separator="\n\n"
    ),
    # Generate summary
    LLMChatBlock(
        block_name="summarize",
        input_cols="summary_prompt",  # Need to add this
        output_cols="summary",
        model="openai/gpt-4o-mini",
        api_key="sk-..."
    ),
]

# Would need to add prompt building step
```

## Testing Custom Scripts

Always test incrementally:

```python
# play.py

# 1. Test with 1 sample
df_test = df.head(1)
result = block(df_test)
print("Single sample works:", "response" in result.columns)

# 2. Test with small batch
df_small = df.head(5)
result = block(df_small)
print("Small batch works:", len(result) == 5)

# 3. Check output quality
print("Sample output:", result["response"].iloc[0][:200])

# 4. Then full run
result_full = block(df)
result_full.to_parquet("output.parquet")
```
