# Supervised Finetuning Data Generation Using Translation

## Objective
Pre-trained language models demonstrate exceptional performance in English. However, their effectiveness deteriorates when applied to non-English languages. A major reason for this gap is the limited availability of high-quality pre-training and post-training data in those languages. Existing approaches often address this issue by translating English datasets into target languages using specialized translation models or powerful large language models such as GPT-4o or Gemini.

This project showcases the use of [`sdg_hub`](https://pypi.org/project/sdg-hub/) for  generating synthetic supervised tuning data in English and then translating the data into Hindi using the SDG Hub framework. It creates question-answer pairs and then translates question-answer pairs into Hindi that can be used to train or fine-tune language models.

## 1. Synthetic Question and Answer Generation
We use documents from Fineweb Edu as the source to create synthetic question-answer pairs. The pipeline generates a large set of contextually grounded Q&A pairs.

## 2. Translating Question and Answers
We use a teacher LLM to translate the generated questions and answers into the target language(Hindi).

Both the above blocks incorprate quality control using a LLM to ensure quality of the question-answer pairs generated.

## 🤝 Contributing
Contributions are welcome! Please open issues or submit PRs for improvements.