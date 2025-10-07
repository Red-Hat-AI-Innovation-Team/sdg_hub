from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple, Callable
import re
from collections import defaultdict
import numpy as np
from datasets import Dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ========================
# Chunking utilities
# ========================
def chunk_text(text: str, max_tokens: int = 400, overlap: int = 60) -> List[str]:
    """
    Splits long text into chunks with overlap.
    Token proxy = words (fast approximation).
    """
    if not text:
        return []
    words = text.split()
    chunks, step = [], max(1, max_tokens - overlap)
    for start in range(0, len(words), step):
        window = words[start:start + max_tokens]
        if not window:
            break
        chunks.append(" ".join(window))
        if start + max_tokens >= len(words):
            break
    return chunks


_SENT_SPLIT = re.compile(r'(?<=[.!?])\s+')


def take_best_sentence(context: str, query: str) -> str:
    """Extract a plausible supportive sentence from context for quoting."""
    if not context:
        return ""
    sents = _SENT_SPLIT.split(context.strip())
    if not sents:
        return context[:400]
    vect = TfidfVectorizer(ngram_range=(1,2), min_df=1)
    X = vect.fit_transform(sents + [query])
    sims = cosine_similarity(X[:-1], X[-1])
    idx = int(np.argmax(sims))
    return sents[idx][:400]


def default_answer_builder(example: Dict[str, Any], oracle_chunk: str) -> Tuple[str, str]:
    """
    Build (support_quote, final_answer).
    - Use `messages` if available.
    - Else, use an extractive fallback from the oracle chunk.
    """
    q = example.get("question", "").strip()
    final_answer = ""

    msgs = example.get("messages")
    if isinstance(msgs, list):
        for m in msgs[::-1]:
            if isinstance(m, dict) and m.get("role") in {"assistant", "system"}:
                content = m.get("content") or ""
                if content.strip():
                    final_answer = content.strip()
                    break

    support_quote = take_best_sentence(oracle_chunk, q)

    if not final_answer:
        print("No final answer found")
        final_answer = support_quote if support_quote else "Answer not available."

    return support_quote, final_answer


# ========================
# RAFT Config
# ========================
@dataclass
class RAFTConfig:
    k_passages: int = 5              # total retrieved passages
    max_tokens_per_chunk: int = 400
    chunk_overlap: int = 60
    p_include_oracle: float = 0.9    # probability to include oracle
    quote_begin: str = "##begin_quote##"
    quote_end: str = "##end_quote##"
    instruction_template: str = (
        "You are given a question and several passages (some are distractors). "
        "Quote exactly one span from a relevant passage, then explain reasoning, "
        "then provide the final answer. Ignore unrelated passages."
    )
    add_doc_ids: bool = True
    shuffle_passages: bool = True
    seed: int = 42


# ========================
# RAFT builder
# ========================
def build_raft_samples(
    hf_dataset,
    cfg: RAFTConfig = RAFTConfig(),
    answer_builder: Callable[[Dict[str, Any], str], Tuple[str, str]] = default_answer_builder,
    text_field: str = "document",
    question_field: str = "question",
    group_by_doc: Optional[str] = "raw_document"
) -> List[Dict[str, Any]]:
    """
    Builds RAFT-style training samples from your dataset.
    """
    rng = np.random.default_rng(cfg.seed)

    # ---- Step 1: Chunk all docs ----
    all_chunks = []
    per_doc_chunks = defaultdict(list)

    tmp = []
    for i, ex in enumerate(hf_dataset):
        ex = dict(ex)
        ex["__row_id__"] = i
        tmp.append(ex)
    data = tmp

    def doc_id_for(ex):
        return ex.get(group_by_doc) if group_by_doc and ex.get(group_by_doc) else f"doc_{ex['__row_id__']}"

    for ex in data:
        doc_text = (ex.get(text_field) or "").strip()
        doc_id = doc_id_for(ex)
        chunks = chunk_text(doc_text, cfg.max_tokens_per_chunk, cfg.chunk_overlap)
        for j, ch in enumerate(chunks):
            gid = len(all_chunks)
            all_chunks.append({"doc_id": doc_id, "passage_id": j, "text": ch})
            per_doc_chunks[doc_id].append((ch, gid))

    if not all_chunks:
        return []

    # ---- Step 2: Fit TF-IDF retriever ----
    vect = TfidfVectorizer(ngram_range=(1,2), min_df=1, max_df=0.98)
    X = vect.fit_transform([c["text"] for c in all_chunks])

    def retrieve_k(query: str, k: int) -> List[int]:
        if not query.strip():
            return list(rng.choice(len(all_chunks), size=k, replace=False))
        qv = vect.transform([query])
        sims = cosine_similarity(X, qv).ravel()
        order = np.argsort(-sims)
        return list(map(int, order[:k]))

    # ---- Step 3: Build RAFT examples ----
    raft_records = []

    for ex in data:
        q = (ex.get(question_field) or "").strip()
        if not q:
            continue

        cand_ids = retrieve_k(q, cfg.k_passages * 5)
        this_doc = doc_id_for(ex)
        oracle_gids = [gid for gid in cand_ids if all_chunks[gid]["doc_id"] == this_doc]

        include_oracle = (rng.random() < cfg.p_include_oracle) and len(oracle_gids) > 0
        oracle_gid = oracle_gids[0] if include_oracle else None

        chosen = []
        if oracle_gid is not None:
            chosen.append(oracle_gid)

        for gid in cand_ids:
            if len(chosen) >= cfg.k_passages:
                break
            if gid == oracle_gid:
                continue
            if all_chunks[gid]["doc_id"] != this_doc:
                chosen.append(gid)

        while len(chosen) < cfg.k_passages:
            gid = int(rng.integers(0, len(all_chunks)))
            if oracle_gid is None or gid != oracle_gid:
                chosen.append(gid)

        documents = []
        for gid in chosen:
            c = all_chunks[gid]
            doc_entry = {
                **({"doc_id": c["doc_id"], "passage_id": c["passage_id"]} if cfg.add_doc_ids else {}),
                "text": c["text"]
            }
            documents.append(doc_entry)

        oracle_chunk = all_chunks[oracle_gid]["text"] if oracle_gid is not None else documents[0]["text"]
        support_quote, final_answer = answer_builder(ex, oracle_chunk)

        quote_wrapped = f"{cfg.quote_begin} {support_quote} {cfg.quote_end}"
        cot = "Reasoning: The quote supports the answer because ..."
        output = "\n".join([quote_wrapped, cot, f"Final Answer: {final_answer}"])

        if cfg.shuffle_passages:
            order = np.arange(len(documents))
            rng.shuffle(order)
            documents = [documents[i] for i in order]
            oracle_index = order.tolist().index(0) if oracle_gid is not None else None
        else:
            oracle_index = 0

        raft_records.append({
            "question": q,
            "context": [d["text"] for d in documents],
            "oracle_context": oracle_chunk if oracle_gid is not None else "",
            "cot_answer": output,
            "answer": final_answer,
            "instruction": cfg.instruction_template,
            "type": "with_oracle" if oracle_gid is not None else "no_oracle",
            "meta": {
                "source_row": ex["__row_id__"],
                "oracle_index": oracle_index
            }
        })

    return Dataset.from_list(raft_records)


def build_messages(raft_record: Dict[str, Any]):
    """
    Construct RAFT-style chat messages for supervised fine-tuning.

    Input:
      raft_record: dict with keys:
        - "question"
        - "context" (list of passages)
        - "cot_answer" (the full target output)
        - "instruction" (optional high-level system instruction)

    Output:
      messages: list of {"role": "system"|"user"|"assistant", "content": str}
    """
    # 1. System message
    sys_msg = raft_record.get("instruction") or (
        "You are a domain expert. You must answer questions by first quoting a span "
        "verbatim from the relevant passage, then giving reasoning, then the final answer. "
        "Ignore distractor passages."
    )

    # 2. User message: serialize passages + question
    passages = "\n\n".join(
        [f"[Passage {i+1}] {p}" for i, p in enumerate(raft_record["context"])]
    )
    user_msg = f"Passages:\n{passages}\n\nQuestion: {raft_record['question']}"

    # 3. Assistant message: the gold output
    assistant_msg = raft_record["answer"]

    return {"messages" : [
        {"role": "user", "content": user_msg},
        {"role": "assistant", "content": assistant_msg},
    ]}