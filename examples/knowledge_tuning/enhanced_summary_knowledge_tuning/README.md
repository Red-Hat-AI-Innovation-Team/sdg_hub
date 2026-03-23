# Knowledge Tuning with Enhanced Summaries

## Objective

Pre-trained language models typically encounter most facts in their training data only **once or twice**, if at all. As a result, knowledge of specific details—especially **proprietary or domain-specific documents**—is often incomplete or missing.

This pipeline is designed to **inject new knowledge** from a given set of documents into an instruction-tuned model. By generating **multiple document augmentations** (summaries, extractive passages, atomic facts) and **synthetic Q\&A pairs**, we repeat and reinforce important information. This repetition helps the model:

* **Memorize facts** it has rarely or never seen before.
* **Generalize across augmentations**, improving reliability when queried.
* **Adapt to proprietary knowledge sources** that were absent from pre-training.

The final product is a **high-quality training dataset** suitable for fine-tuning, enabling models to answer queries more accurately and faithfully based on the injected documents.

---

## Table of Contents

| Section | Description |
|---------|-------------|
| [Data Generation Pipeline](docs/data_generation.md) | Document summarization, synthetic Q&A generation, and quality control |
| [SFT Benchmark Results](docs/sft_results.md) | Supervised fine-tuning results on the QuALITY benchmark |
| [CPT Results](docs/cpt_results.md) | Continued pre-training with augmented documents |
| [Multilingual Support](docs/multilingual.md) | Generating training data in any language, with Spanish benchmark results |
| [Custom Documents SFT](docs/custom_docs_sft.md) | Training on custom domain-specific documents with SFT |
| [Blog: Knowledge Tuning Deep Dive](docs/blog.md) | End-to-end walkthrough of the pipeline, augmentation strategies, and results |
