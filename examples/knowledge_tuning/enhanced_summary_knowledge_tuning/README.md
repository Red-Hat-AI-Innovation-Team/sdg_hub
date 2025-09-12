# Knowledge Tuning with Enhanced Summaries

## Objective

Pre-trained language models typically encounter most facts in their training data only **once or twice**, if at all. As a result, knowledge of specific details—especially **proprietary or domain-specific documents**—is often incomplete or missing.

This pipeline is designed to **inject new knowledge** from a given set of documents into an instruction-tuned model. By generating **multiple document augmentations** (summaries, extractive passages, atomic facts) and **synthetic Q\&A pairs**, we repeat and reinforce important information. This repetition helps the model:

* **Memorize facts** it has rarely or never seen before.
* **Generalize across augmentations**, improving reliability when queried.
* **Adapt to proprietary knowledge sources** that were absent from pre-training.

The final product is a **high-quality training dataset** suitable for fine-tuning, enabling models to answer queries more accurately and faithfully based on the injected documents.

---

## 1. Document Summarization

To bootstrap the process, we generate **three complementary types of summaries** for each source document. This ensures the model captures content at multiple levels of abstraction:

* **Detailed Summaries** – Rich, comprehensive overviews of the document.
* **Extractive Summaries** – Directly extracted sentences and passages representing the most important parts.
* **Atomic Facts** – Concise, standalone factual statements distilled from the text.

This multi-perspective approach improves the model’s ability to **memorize, generalize, and recall** key knowledge.

---

## 2. Synthetic Q\&A Generation

With summaries in place, we scale up training data via **synthetic Q\&A generation**:

* Users provide a small set of **seed examples** (initial Q\&A pairs).
* The pipeline uses these seeds to generate a large set of **contextually grounded Q\&A pairs**, tightly linked to the summarized documents.
* This expands sparse seed data into a **rich, diverse training dataset** suitable for fine-tuning.

---

## 3. Quality Control

High-quality training data is essential. To ensure faithfulness and accuracy, we employ a **teacher-model evaluation loop**:

1. Provide the model with a generated answer and the original document.
2. Ask it to extract each factual claim from the answer.
3. Verify whether each claim is **explicitly supported** by the document.

Only claims passing this check are retained. This process filters out **hallucinations and unsupported statements**, ensuring reliable Q\&A pairs.

---

## Data Generation Statistics

### Summary Augmentation

Each “cut” represents the total number of summaries generated per document across all three augmentation types.

| Cut (NUMBER\_OF\_SUMMARIES = 3) | Token Count |
| ------------------------------- | ----------- |
| 1                               | 2,193,502   |
| 2                               | 4,383,655   |
| 5                               | 10,870,396  |
| 10                              | 21,815,170  |
| 20                              | 43,601,976  |
| 30                              | 65,395,710  |
| 40                              | 87,118,308  |
| 50                              | 108,779,213 |

---

### Finance Bench Example

For Finance Bench (NUMBER\_OF\_SUMMARIES = 1):

| Cut | Token Count |
| --- | ----------- |
| 50  | 213,333,192 |
