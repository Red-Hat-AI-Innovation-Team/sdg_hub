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

## Data Generation Statistics and Results

**Teacher model for generation:** `openai/gpt-oss-120b`  
**Student model trained:** `meta-llama/Llama-3.1-8B-Instruct`  
**Training method:** Supervised Fine-Tuning (SFT)

---

### Summary Augmentation

For each document, we generate three augmentation types—detailed summaries, extractive summaries, and atomic facts. Each “cut” on the table below represents the total number of summary augmentations per document (i.e., how many times each augmentation process is run).

| Cut (NUMBER\_OF\_SUMMARIES = 3) | Token Count   |
| ------------------------------- | ------------- |
| Input Corpus                    | 1,517,465     |
| 10                              | 87,248,889    |
| 20                              | 158,615,276   |
| 30                              | 230,306,195   |
| 40                              | 301,805,906   |
| 50                              | 373,183,414   |

---

### Benchmark Results

- **Evaluation benchmark:** [QuALITY benchmark](https://nyu-mll.github.io/quality/)
- **Evaluation script & metric:** [Synthetic_Continued_Pretraining](https://github.com/ZitongYang/Synthetic_Continued_Pretraining/blob/main/evaluation.py), Exact Match (EM)
- **Student model:** meta-llama/Llama-3.1-8B-Instruct (after SFT on generated/augmented summaries)
- **Performance metric:** Model accuracy

![Quality Benchmark Accuracy](imgs/quality_benchmark_accuracy.png)

*Figure: Model accuracy across the QuALITY benchmark datasets, comparing SFT training on enhanced document summaries with the original model performance.*

---
