## LAB: LARGE-SCALE ALIGNMENT FOR CHATBOTS

Shivchander Sudalairaj ∗ Abhishek Bhandwaldar ∗ Aldo Pareja ∗ Kai Xu David D. Cox

Akash Srivastava ∗ † ,

*Equal Contribution, † Corresponding Author

## ABSTRACT

This work introduces LAB (Large-scale Alignment for chatBots), a novel methodology designed to overcome the scalability challenges in the instruction-tuning phase of large language model (LLM) training. Leveraging a taxonomy-guided synthetic data generation process and a multi-phase tuning framework, LAB significantly reduces reliance on expensive human annotations and proprietary models like GPT-4. We demonstrate that LAB-trained models can achieve competitive performance across several benchmarks compared to models trained with traditional human-annotated or GPT-4 generated synthetic data. Thus offering a scalable, cost-effective solution for enhancing LLM capabilities and instructionfollowing behaviors without the drawbacks of catastrophic forgetting, marking a step forward in the efficient training of LLMs for a wide range of applications.

## 1 INTRODUCTION

Large language models (LLMs) have achieved remarkable levels of success in various natural language processing (NLP) applications, including question-answering , entity extraction , and summarization . This has been made possible, in large part, by the introduction of the transformer architecture , which can leverage large amounts of unlabeled, unstructured data, enabling the scaling of LLMs to billions, or even trillions of parameters. LLMs are typically trained in phases: a self-supervised pre-training phase, followed by supervised alignment tuning phases.

The majority of the cost of training an LLM comes from the pre-training phase. During this phase, a model is trained in an auto-regressive manner to predict the next token in the target language using trillions of tokens worth of unlabeled data, requiring thousands of GPUs training for months at a time. Alignment tuning, typically happens in two stages: instruction tuning, followed by preference tuning. Instruction tuning is more akin to the traditional model training approach in machine learning, where the model is trained directly on tasks of interest. In this stage, the model is given a task description in the form of an natural language instuction (e.g. Summarize the following news article in 2 lines: { News article } ) and the model is trained to maximize the likelihood of the provided ground truth summary. Preference tuning, on the other hand, is done using techniques such as RLHF (Stiennon et al., 2022; Ouyang et al., 2022) and DPO (Rafailov et al., 2023), where the response from an instruction-tuned model is rated as preferred or unpreferred using human feedback.

In comparison to pre-training, the instruction tuning and preference tuning stages comprise a small fraction of the overall training procedure, both in terms of the data used as well as the compute infrastructure required to train models Touvron et al. (2023). For example, Meta's LLaMA 2 models were trained with just tens of thousands of high quality human-generated instruction/response data pairs, followed by multiple rounds of RLHF with a comparatively limited number of examples as compared to pretraining data volumes Touvron et al. (2023). From a traditional machine learning training perspective, this imbalance in the scale across the phases is unconventional-typically one would expect a model to perform best when it has been trained directly on the desired tasks, using as much data as possible. The deviation from the tradtional LLM approach relies on the idea that pre-

training captures enough of the distribution of language and knowledge, such that a small amount of supervised training can 'unlock' or shape latent abilities related to the ultimate desired instructionfollowing behavior of the model. However, unlike the unstructured data that is abundantly available in the public domain, high-quality, human-generated task-specific instruction data is costly to procure, even via crowd-sourcing, and human-generated instruction data is typically closely guarded by model builders, even for ostensibly 'open' model-building efforts. In this work, we address the challenges associated with scaling of the alignment-tuning phase and propose a new method called LAB: Large-scale Alignment for chatBots. The LAB method consists of two components: (i) a taxonomy-guided synthetic data generation method and quality assurance process that yields a highly diverse and high-quality instruction dataset, without resorting to the use of proprietary LLMs like GPT-4 or substantial human curation, and (ii) a novel multi-phase training framework and unconventional tuning regime that allows for adding new knowledge and instruction-following abilities into pre-trained LLMs without suffering from catastrophic forgetting. Our findings show that LABtrained models can perform competitively with proprietary and open-source models that use human annotations and/or synthetic data generated using GPT-4 on a number of benchmarks.

## 2 RELATED WORK

Existing methods for instruction tuning typically either rely on humans for generating high-quality datasets, or use synthetic data generation using a large teacher model. OpenAI (Ouyang et al., 2022) arguably set the standard for model alignment from human data, employing human annotators to gather data for supervised fine tuning (SFT) and reinforcement learning with human feedback (RLHF) training. Collecting human-generated data for these steps is complex undertaking; the selection of annotators requires a rigorous multi-stage screening process aimed at achieving high inter-annotator agreement, and collecting even modest amounts data (by LLM standards) requires the coordination of large groups of annotators. The creators of the LLaMA 2 model series (Touvron et al., 2023) followed a similar recipe, collecting tens of thousands of human-generated instruction samples, and approximately 1 million human-annotated binary comparisons for reward modeling. Not only are such approaches expensive and time consuming, but they can also potentially limit agility in exploring the space of instructions and capabilities the model is trained to perform. Alternatives to this approach, such as transforming existing human datasets into instructions via templating (Wei et al.) can be more cost effective, but face limitations in the naturalness and length of the responses used for training.

More recently, training with synthetic data generated from LLMs has emerged as an alternative to purely human-data-based approaches. Wang et al. (2023) introduced Self-Instruct, which leverages a small number of handwritten human seed instructions as input to bootstrapping process to generate a large number of samples using an LLM's own generation abilities. Taori et al. (2023) built upon Self-Instruct, using a larger teacher model to generate synthetic data to train a smaller student model, and incorporating principles in the generation prompt to promote diversity in the generated instruction data. Xu et al. (2023) introduces Evol-Instruct, another variant of Self-Instruct, that synthesizes iteratively more complex instruction to overcome shortcomings of previous methods. Mukherjee et al. (2023), Mitra et al. (2023) present a synthetic data generation approach to enhance task diversity and scalability, alongside a progressive training framework aimed at improving the model's reasoning ability and response style to match teacher models. This is achieved by generating rich reasoning signals in the generated answer and progressively training on datasets of varying difficulty in incremental phases.

Similar to LAB, concurrent work, GLAN (Li et al., 2024), employs a semi-automatic approach to synthetic data generation that uses a human-curated taxonomy to generate instruction tuning data from a teacher model. However, as explained in section 3.2.2, unlike LAB, GLAN cannot be used to generate synthetic data from domains that are not captured in the teacher model's support. As such, while LAB uses the open-source Mixtral model as the teacher, like many other synthetic data generation approaches, GLAN has to rely on a large proprietary model (GPT-4). This poses complicated questions about the usability of generated data (especially for commercial purposes) since the terms of use of proprietary models typically forbid using the model to improve other models.

Figure 1: Overview of the LAB alignment method. Starting from the taxonomy root, data are curated in each top-level groups and examples in the leaf nodes are used by the synthetic data generators to generate orders of magnitude data for the phased-training step for instruct-tuning.


## 3 METHODOLOGY

LAB consists of two components: (i) a taxonomy to enable data curation (section 3.1) as well as, guide the synthetic data generator (section 3.2) and (ii) a multi-phased instruction-tuning method with replay buffers to enable large-scale alignment-tuning. (section 3.3). (i) serves the purpose of ensuring high diversity and quality in the synthetically generated instruction-tuning dataset while (ii) ensures training stability and prevents catastrophic forgetting. Figure 1 provides an overview of the end-to-end pipeline of applying the LAB method to align a pre-trained LLM.

## 3.1 TAXONOMY

To enable the data curator or the model designer to organize the instruction-tuning training data, we define a taxonomy that hierarchically classifies the data samples into smaller task groups. At a high level, the taxonomy has three main branches: knowledge, foundational skills, and compositional skills. Each of these branches is further split into more granular levels where the tasks are defined in the leaf nodes and exemplified by providing manually written instruction-response pairs. This allows for easily identifying missing tasks in the target LLM and other tasks of interest and adding them to the training data pool. New tasks are added to the taxonomy by creating a leaf node under the appropriate branch and attaching 1-3 examples.

Knowledge The knowledge branch in the taxonomy is first divided based on document types like textbooks, technical manuals, etc., which are further divided into various domains like finance, statistics, etc.; see the sub-tree for knowledge in Figure 1 as an example. Each domain has a collection of documents and a sample set of domain-specific questions and answers. This organization allows for better control over the licensing of text documents. As described in the next section, only the documents with permissible licenses are selected for synthetic data generation, excluding knowledge sources that lack proper licensing, reinforcing the integrity of our knowledge-generation processes.

Foundational skills We identify mathematics, coding, linguistic ability and reasoning as foundational skills that the model requires to prime itself for better knowledge acquisition and build further complex and compositional skills. To teach the model foundational skills, we employ publicly available datasets (Longpre et al., 2023; Xiang Yue, 2023; Yin et al., 2018; Trivedi et al., 2022); see the sub-tree for foundational skills in Figure 1 for an example.

Compositional skills Compositional skills refer to the tasks that require a combination of knowledge and foundational skills, synergistically, to answer complex queries from users. For instance, the model's ability to write a company-wide email sharing insights about the company's performance last quarter and guidance for the upcoming year would require the model to understand the financial aspects of revenue, profit and loss, the skills of doing basic arithmetic and also have the skills to compose a formal email.

## 3.2 TAXONOMY-DRIVEN SYNTHETIC DATA GENERATOR

The small number of manually curated data samples, embedded in the leaf nodes of the taxonomy, can be directly used for instruction tuning of the chatbot, however, the model may still perform poorly. Prior work (Li et al., 2023) has shown that typically, a large amount of high-quality instruction data is required for improving instruction following performance of LLMs. It is possible to leverage existing SDGs like Wang et al. (2023); Taori et al. (2023) to use the embedded examples and generate a lot more instruction data synthetically using teacher LLMs. But, such distillationbased SDGs tend to over-sample from the dominant modes of the teacher model and thus lack in diversity and quality of the generated data Gudibande et al. (2023). We argue that this limitation is attributed to the random selection of examples from the pool of seed samples: with random selection, the examples used to prompt the teacher model at each time are an 'average' of the seed pool i.e. they do not focus on any specific task. This lack of focus tends to encourage the teacher model to generate more synthetic data from its dominant modes and ignore the long tail of interesting tasks.

To address this issue, we replace the random sampling in existing SDGs with a taxonomy-driven approach to guide the sampling of synthetic data, enabling targeted coverage of the support of the teacher model distribution around the individual leaf nodes of the taxonomy. Figure 2 illustrate the high-level idea behind this change. Figure 2a shows the issue of randomly sampling in the input


output distribution

(a) Input distributions

(b) Output distributions

Figure 2: Intuition of how taxonomy-driven sampling produces diverse set of synthetic data and hence improve the data used to train student model across the task domain. Figure 2a shows how taxonomy-driven sampling leads to an input distribution with wide support and distinct modes while self-instruct gives an smooth input distribution. Figure 2b shows the consequence using inputs in generating synthetic data: teacher model will focus its own dominant modes if the input is smooth but focus on each task better if the inputs are also concentrated on each task.

space of the teacher model (i.e. prompts). Given a set of seed examples (red), randomly sampling with more than one example gives an approximation to the average of the seed pool, leading to a smoothed distribution, e.g. self-instruct distribution (blue). With the taxonomy-driven sampling, since only the examples within each of the leaf nodes are used when sampling for the corresponding tasks, each of the tasks are guaranteed to be well represented in the prompts (purple). Second, when

You are asked to come up with a set of { num samples } diverse questions on { task } .

Please follow these guiding principles when generating responses:

- * Use proper grammar and punctuation.
- * Always generate safe and respectful content. Do not generate content that is harmful, abusive, or offensive.
- * Always generate content that is factually accurate and relevant to the prompt.
- * The questions should be clear and human-like.
- * The questions should be diverse and cover a wide range of topics.
- * The questions should not be template-based or generic, it should be very diverse.
- * Simply return the questions, do not return any answers or explanations.
- * Strictly adhere to the prompt and generate responses in the same style and format as the example.

To better assist you with this task, here is an example:

### Question:

- 1. { icl question }

Now generate { num samples } such questions, remember to follow the principles mentioned above and use the same format as the examples. Remember to use the same style and format as the example above. Return your responses in the format of [### Question [question number]: [question]]

Figure 3: Instruction Generator prompt template

it comes to the output space, for a given teacher model (red), prompting it with random examples (i.e. smoothened input distribution) tends to make it sampling from its own dominant mode (blue) while prompting it with focused examples in each leaf node (i.e. input distribution with distinct modes), the teacher model is guaranteed to generate synthetic data for each of the tasks (purple).

With the above insight, we now introduce two new synthetic data generation (SDG) methods in LAB that leverage the taxonomy to guide the data generation process. The first one is targeted for skills generation and uses the handful of task examples in the leaf nodes to generate a lot more using the open-source Mixtral-7x8B model. The second one is targeted at knowledge generation. While it still uses the Mixtral-7x8B model, unlike prior works, it does not rely on the knowledge stored in the teacher model.

## 3.2.1 SKILL GENERATION

Skills-SDG uses four prompt templates, one for each of the four, below-mentioned, stages of data generation. Each template has its own set of principles and instructions that control the role of the teacher model (generator vs evaluator) and guide the generation/evaluation process.

- 1. Instruction generation: In the first stage, the teacher model acts as a question generator, using a specialized prompt (see Figure 3 for an example) to leverage its knowledge and create diverse questions. By iterating through each leaf node of a taxonomy, the teacher generates queries that adhere to specific principles and thoroughly explore the targeted domain, enhancing the comprehensiveness of the generated content.
- 2. Evaluating synthetic instruction: In this stage, the teacher model assumes the role of an instruction evaluator, the teacher model uses targeted prompts to filter out questions that don't meet predefined principles, including relevance to the domain, potential harm, or questions beyond a language model's answering capabilities. This ensures that only high-quality, contextually appropriate questions move forward in the process.
- 3. Generating responses: The teacher model, functioning as a response generator in this stage, adopts dual personas for precision and creativity, guided by distinct prompts. This tailored approach helps to generate both, creative responses for domains like writing and role-play, and precise answers for STEM and data extraction, aligning the response style to human expectations through principles and seed examples in the leaf nodes.

Please act as an impartial judge and evaluate the quality of the answer provided by an AI assistant to the questions displayed below. Evaluate whether or not the answer is a good example of how AI Assistant should respond to the user's instruction. Please assign a score using the following 3-point scale:

- 1: It means the answer is incorrect, irrelevant, unsafe or provides incomplete and garbage information. For instance, the answer may be factually wrong, off-topic, or filled with irrelevant content that doesn't address the user's question or it could be incomplete and hanging. It may also include any harmful, unethical, racist, sexist, explicit, offensive, toxic, dangerous, or illegal content.
- 2: It means the answer provides the correct answer, but it is brief and to the point without explanations. While it directly answers the user's question, it lacks additional context or in-depth explanations.
- 3: It means the answer is a perfect answer from an AI Assistant. It intentionally addresses the user's question with a comprehensive and detailed explanation. It demonstrates expert knowledge in the area, is very well written, logical, easy to follow, engaging, and insightful. And the answer is safe and does not include any harmful content.

Begin your evaluation by providing a short explanation. Be as objective as possible. After providing your explanation, you must rate the answer on a scale of 1 to 3 as mentioned above. Please use the following examples as a reference for your evaluation.

Figure 4: Instruction-response Evaluation template

- 4. Evaluating the synthetic instruction-response pair: The final stage involves a rigorous process to filter and select high-quality instruction and response pairs. Using a 3-point rating system (see Figure 4 for an example), the teacher model evaluates each sample, filtering out those that are incorrect, irrelevant, or deviate from the provided principles, ensuring the training dataset's quality and relevance are enhanced for the student model.

## 3.2.2 KNOWLEDGE-GENERATION

Synthetic data generators are inherently limited by the knowledge and capabilities of the teacher model. This is one of the main reasons why most successful SDG methods (Xu et al., 2023; Mukherjee et al., 2023; Mitra et al., 2023) depend on GPT-4 model, which presumably has the highest coverage of knowledge and skills. However, there are many domains that no open/proprietary model is trained on and hence cannot work as a teacher model using existing SDG methods. To address this limitation, in LAB we devised a new SDG pipeline for generating instruction data on domains that the teacher model has not been trained on. We call it knowledge-SDG.

Similar to the process of skills generation, knowledge-SDG uses the curator-provided examples embedded in the leaf nodes of the knowledge branch of the taxonomy. But additionally, the teacher model is provided a knowledge source in the form of documents, manuals, and books on the target subject to ground the generated instruction data into a reliable source thus avoiding dependence on the internal knowledge base of a teacher model, which may struggle with specialized domains and could lead to inaccuracies or hallucinations especially on highly specialized, technical domains.

To ensure that the generated answers remain faithful to the content of the source material, similar to the skills-SDG, teacher model is repurposed as an evaluator that validates the generated responses are grounded and faithful to the source documents.

## 3.3 MULTI-PHASE TRAINING

LAB training happens in two phases, knowledge tuning, followed by skills tuning.

In the knowledge-tuning phase, the model is trained on samples from the knowledge and foundational skills branches of the taxonomy. This phase in-turn, is carried out in two steps. We split the data under the knowledge and foundational skills branches into two buckets based on the response length. Then we first train the model on the samples with short responses before moving on to train-

| Phase            | Step   | Training data                        | Replay buffer   |
|------------------|--------|--------------------------------------|-----------------|
| Knowledge Tuning | 1      | Knowledge (short)                    | N/A             |
| Knowledge Tuning | 2      | Knowledge (long) Foundational skills | KT/1 data       |
| Skill Tuning     | N/A    | Compositional skills                 | KT/1 &KT/2 data |

Table 1: Data and reply buffers used in phase-training.

| MODEL           | PHASE/STEP   | LEARNING RATE   | BATCH SIZE   |   CONTEXT LENGTH | #SAMPLES   | #WARM-UP   | #EPOCHS   |
|-----------------|--------------|-----------------|--------------|------------------|------------|------------|-----------|
| LABRADORITE-13B | KT/1 KT/2 ST | 2E-5            | 3840         |             2048 | 630K 230K  | 385        | 5 7       |
| LABRADORITE-13B |              |                 |              |             4096 | 550K       |            | 7         |
| MERLINITE-7B    | KT/1 KT/2    | 1E-6            | 3840         |             2048 | 630K 230K  | 800        | 4 4       |
| MERLINITE-7B    | ST           |                 |              |             4096 | 550K       |            | 7         |

Table 2: Hyper-parameters used in training for LABRADORITE-13B and MERLINITE-7B.

ing on samples with long responses. Similar to prior work (Mitra et al., 2023), our empirical results also suggest that this two-step approach to knowledge-tuning improves model performance.

Post-knowledge tuning, we start the skills-tuning phase where the best model checkpoint from the knowledge-tuning phase is trained on the compositional skills branch of the taxonomy. To address the challenge of catastrophic forgetting when training in two distinct phases, a replay buffer of the data from the knowledge-tuning phase in employed. Our empirical findings indicate that starting with knowledge and foundational skills training, before progressing to compositional skills leads to significantly better benchmark performance.

For selecting the best model checkpoint during intermediate phases, we rely on the MMLU benchmark (Hendrycks et al., 2020) during the knowledge-tuning phase and the MT-bench (Zheng et al., 2024) during the skills-tuning phase. Please refer to table 1 for an overview of our training phases.

Training Details In our training process, we consciously avoid overtraining. Despite the possibility of achieving higher scores on intermediate benchmarks, we have found that selecting checkpoints from earlier stages of training results in more reliable and generalizable model performance. We employ small learning rates with an extended warm-up period, specifically 2 × 10 -5 for Llama-based models and 1 × 10 -6 for Mistral-based models, each beginning with a linear warm-up. This strategy is hypothesized to aid the model in transitioning from broad dataset-wide learning to more focused, task-specific adjustments. Additionally, we utilize a large effective batch size of 3840, achieved through gradient accumulation, to enhance stability across the diverse range of tasks being learned concurrently. Our findings suggest that using cosine decay on learning rates during intermediate phases can destabilize subsequent training stages, likely due to the learning rate's reduction to near zero, narrowing the loss landscape and complicating the integration of new phase gradients. Refer to table 2 for an overview of our training hyper-parameters.

## 4 RESULTS

In this study, we implemented the LAB method on two distinct open models, LLAMA-2-13B (Touvron et al., 2023)and MISTRAL-7B (Jiang et al., 2023), utilizing MIXTRAL-8X7B-INSTRUCTV0.1 (Jiang et al., 2024) as the teacher model. This approach yielded two LAB-aligned models: LABRADORITE-13B and MERLINITE-7B.

During the synthetic data generation phase, we employed a taxonomy consisting of numerous leaf nodes to produce a dataset comprising 1.2 million samples, divided almost evenly between knowledge-based (617k) and skill-based (588k) samples. The specific training hyper-parameters employed during this study are summarized in table 2.

† taken from the LMSYS Chatbot Arena Leaderboard.

| MODEL               | ALIGNMENT            | TEACHER                 | MT-BENCH   |   MMLU |   ARC |   HELLASWAG |   WINOGRANDE |   GSM8K |
|---------------------|----------------------|-------------------------|------------|--------|-------|-------------|--------------|---------|
| LLAMA-2-13B-CHAT    | SFT + RLHF           | HUMAN ANNOTATORS        | 6.65 †     |  54.58 | 59.81 |       82.52 |        75.93 |   34.8  |
| ORCA-2              | PROGRESSIVE TRAINING | GPT-4                   | 6.15 †     |  60.37 | 59.73 |       79.86 |        78.22 |   48.22 |
| WIZARDLM-13B        | EVOL- I NSTRUCT      | GPT-4                   | 7.20 †     |  54.83 | 60.24 |       82.62 |        76.4  |   43.75 |
| LABRADORITE-13B     | LAB                  | MIXTRAL-8X7B- I NSTRUCT | 7.23 ‡     |  58.89 | 61.69 |       83.15 |        79.56 |   40.11 |
| MISTRAL-7B-INSTRUCT | SFT                  | PUBLIC DATASETS         | 6.84 †     |  60.37 | 63.65 |       84.76 |        76.8  |   41.85 |
| ZEPHYR-7B- β        | SFT + DPO            | GPT-4                   | 7.34 †     |  61.07 | 63.74 |       84.19 |        78.06 |   34.04 |
| MERLINITE-7B        | LAB                  | MIXTRAL-8X7B- I NSTRUCT | 7.66 ‡     |  64.88 | 63.99 |       84.37 |        78.24 |   44.58 |

‡ average of 3 runs.

Table 3: Evaluation of LLMs with different alignment methods over a comprehensive set of benchmark metrics. Settings of each metric can be found in the main text.

Wecompare the performance of LABRADORITE-13B and MERLINITE-7B against other models that use the same base models for alignment, which include

## LLAMA-2-13B

- · LLAMA-2-13B-CHAT (Touvron et al., 2023): RLHF with human annotators by the same team that develops LLAMA-2-13B
- · ORCA-2 (Mitra et al., 2023):
- · WIZARDLM-13B-V1.2 (Xu et al., 2023): model with the highest MT-Bench amongs those use LLAMA-2-13B as the base model on LMSYS Chatbot Arena Leaderboard (Zheng et al., 2023).

## MISTRAL-7B

- · MISTRAL-7B-INSTRUCT-V0.2 (Jiang et al., 2023): instruction-tuning using supervised fine-tuning (SFT) on publicly available conversation datasets by the same team that develops MISTRAL-7B
- · ZEPHYR-7B-BETA (Tunstall et al., 2023): model with the highest MT-Bench amongs those use MISTRAL-7B as the base model on LMSYS Chatbot Arena Leaderboard (Zheng et al., 2023).

To compare the aligned LLMs, we consider the following evaluation metrics with the settings consistent with those used by LMSYS Chatbot Arena Leaderboard (Zheng et al., 2023)

- · MT-Bench (Zheng et al., 2023): 1-turn and 2-turn average
- · MMLU(Hendrycks et al., 2021): 5-shot
- · ARC (Clark et al., 2018): 25-shot
- · HellaSwag (Zellers et al., 2019): 10-shot
- · Winogrande (Sakaguchi et al., 2019): 5-shot
- · GSM8k (Cobbe et al., 2021): 5-shot strict

All results are reported in table 3.

Notably, in terms of MT-Bench, LABRADORITE-13B performs better than the current best model fine-tuned on LLAMA-2-13B and MERLINITE-7B performs better than the current best model finetuned on MISTRAL-7B, achieving state-of-the-art performance in term of chatbot capability. Importantly, out training method ensures that the model is not only good at multi-turn conversation but

also maintains its knowledge or reasoning capability, as shown by the overall superior performance in the rest of the metrics. Besides, unlike those top models that use GPT-4 as the teacher model, we achieve this performance using the open-weights MIXTRAL-8X7B-INSTRUCT-V0.1, which is relatively weaker teacher model at orders of magnitude less cost.

## REFERENCES

| Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. Think you have Solved Question Answering? Try ARC, the AI2 Reasoning Challenge, March 2018.                                                                                                                                                                                                                                                                                                       |
|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. Training Verifiers to Solve Math Word Problems, November 2021.                                                                                                                                                                                                                                                      |
| Arnav Gudibande, Eric Wallace, Charlie Snell, Xinyang Geng, Hao Liu, Pieter Abbeel, Sergey Levine, and Dawn Song. The false promise of imitating proprietary llms. arXiv preprint arXiv:2305.15717 , 2023.                                                                                                                                                                                                                                                                                                       |
| Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. Measuring massive multitask language understanding. In International Conference on Learning Representations , 2020.                                                                                                                                                                                                                                                                                       |
| Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. Measuring massive multitask language understanding, 2021.                                                                                                                                                                                                                                                                                                                                                 |
| Albert Q. Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chap- lot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, L´lio e Renard Lavaud, Marie-Anne Lachaux, Pierre Stock, Teven Le Scao, Thibaut Lavril, Thomas Wang, Timoth´e e Lacroix, and William El Sayed. Mistral 7B, October 2023.                                                                                                                                               |
| Albert Q. Jiang, Alexandre Sablayrolles, Antoine Roux, Arthur Mensch, Blanche Savary, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Emma Bou Hanna, Florian Bressand, Gi- anna Lengyel, Guillaume Bour, Guillaume Lample, L´lio e Renard Lavaud, Lucile Saulnier, Marie- Anne Lachaux, Pierre Stock, Sandeep Subramanian, Sophia Yang, Szymon Antoniak, Teven Le Scao, Th´ophile e Gervet, Thibaut Lavril, Thomas Wang, Timoth´e e Lacroix, and William El Sayed. Mixtral of Experts, January 2024. |
| Haoran Li, Qingxiu Dong, Zhengyang Tang, Chaojun Wang, Xingxing Zhang, Haoyang Huang, Shaohan Huang, Xiaolong Huang, Zeqiang Huang, Dongdong Zhang, Yuxian Gu, Xin Cheng, Xun Wang, Si-Qing Chen, Li Dong, Wei Lu, Zhifang Sui, Benyou Wang, Wai Lam, and Furu Wei. Synthetic data (almost) from scratch: Generalized instruction tuning for language models, 2024.                                                                                                                                              |
| Xian Li, Ping Yu, Chunting Zhou, Timo Schick, Luke Zettlemoyer, Omer Levy, Jason Weston, and Mike Lewis. Self-alignment with instruction backtranslation, 2023.                                                                                                                                                                                                                                                                                                                                                  |
| Shayne Longpre, Le Hou, Tu Vu, Albert Webson, Hyung Won Chung, Yi Tay, Denny Zhou, Quoc V. Le, Barret Zoph, Jason Wei, and Adam Roberts. The flan collection: Designing data and methods for effective instruction tuning, 2023.                                                                                                                                                                                                                                                                                 |
| Arindam Mitra, Luciano Del Corro, Shweti Mahajan, Andres Codas, Clarisse Simoes, Sahaj Agar- wal, Xuxi Chen, Anastasia Razdaibiedina, Erik Jones, Kriti Aggarwal, Hamid Palangi, Guoqing Zheng, Corby Rosset, Hamed Khanpour, and Ahmed Awadallah. Orca 2: Teaching Small Lan- guage Models How to Reason. https://arxiv.org/abs/2311.11045v2, November 2023.                                                                                                                                                    |
| Subhabrata Mukherjee, Arindam Mitra, Ganesh Jawahar, Sahaj Agarwal, Hamid Palangi, and Ahmed Awadallah. Orca: Progressive learning from complex explanation traces of gpt-4, 2023.                                                                                                                                                                                                                                                                                                                               |
| Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kel- ton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul Christiano, Jan Leike, and Ryan Lowe. Training language models to follow instructions with human feedback, 2022.                                                                                                                                    |

| Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D. Manning, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model, 2023.                                                                                                         |
|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Keisuke Sakaguchi, Ronan Le Bras, Chandra Bhagavatula, and Yejin Choi. WinoGrande: An Ad- versarial Winograd Schema Challenge at Scale, November 2019.                                                                                                                                                |
| Nisan Stiennon, Long Ouyang, Jeff Wu, Daniel M. Ziegler, Ryan Lowe, Chelsea Voss, Alec Radford, Dario Amodei, and Paul Christiano. Learning to summarize from human feedback, 2022.                                                                                                                   |
| Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li, Carlos Guestrin, Percy Liang, and Tatsunori B. Hashimoto. Stanford alpaca: An instruction-following llama model. https://github.com/tatsu-lab/stanford_alpaca , 2023.                                                           |
| Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Niko- lay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open founda- tion and fine-tuned chat models. arXiv preprint arXiv:2307.09288 , 2023.                                   |
| Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and Ashish Sabharwal. Musique: Multihop questions via single-hop question composition, 2022.                                                                                                                                                    |
| Lewis Tunstall, Edward Beeching, Nathan Lambert, Nazneen Rajani, Kashif Rasul, Younes Belkada, Shengyi Huang, Leandro von Werra, Cl´mentine e Fourrier, Nathan Habib, Nathan Sarrazin, Omar Sanseviero, Alexander M. Rush, and Thomas Wolf. Zephyr: Direct Distillation of LMAlignment, October 2023. |
| Yizhong Wang, Yeganeh Kordi, Swaroop Mishra, Alisa Liu, Noah A. Smith, Daniel Khashabi, and Hannaneh Hajishirzi. Self-Instruct: Aligning Language Models with Self-Generated Instructions, May 2023.                                                                                                  |
| Jason Wei, Maarten Bosma, Vincent Zhao, Kelvin Guu, Adams Wei Yu, Brian Lester, Nan Du, An- drew MDai, and Quoc V Le. Finetuned language models are zero-shot learners. In International Conference on Learning Representations .                                                                     |
| et al. Xiang Yue. Mammoth: Building math generalist models through hybrid instruction tuning. arXiv preprint arXiv:2309.05653 , 2023.                                                                                                                                                                 |
| Can Xu, Qingfeng Sun, Kai Zheng, Xiubo Geng, Pu Zhao, Jiazhan Feng, Chongyang Tao, and Daxin Jiang. WizardLM: Empowering Large Language Models to Follow Complex Instructions, June 2023.                                                                                                             |
| Pengcheng Yin, Bowen Deng, Edgar Chen, Bogdan Vasilescu, and Graham Neubig. Learning to mine aligned code and natural language pairs from stack overflow. In International Conference on Mining Software Repositories , MSR, pp. 476-486. ACM, 2018. doi: https://doi.org/10.1145/ 3196398.3196408.   |
| Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi. HellaSwag: Can a Machine Really Finish Your Sentence?, May 2019.                                                                                                                                                              |
| Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric P. Xing, Hao Zhang, Joseph E. Gonzalez, and Ion Stoica. Judging llm-as-a-judge with mt-bench and chatbot arena, 2023.                                                     |
| Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric Xing, et al. Judging llm-as-a-judge with mt-bench and chatbot arena. Advances in Neural Information Processing Systems , 36, 2024.                                        |