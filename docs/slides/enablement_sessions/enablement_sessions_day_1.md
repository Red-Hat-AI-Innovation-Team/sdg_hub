Workshop on sdg_hub: In this sessin we will look at sdg_hub and instructlab.sdg for generating synthetic data for training a LLM on specific task. More specifically we will be generating data for below two use-cases to train a student model for the task.
### Use Case 1 – Table Manipulation
This fits the free-form bucket.
A markdown table is included in the prompt.
The model is asked to perform specific manipulations — we don’t care where the table comes from.
### Use Case 2 – Document-to-Table
Takes unstructured text and converts it into a structured table.
Source of the document is irrelevant.

Below is workshop slides including summary of speaker notes and QA at the end of the workshop

What is sdg_hub and Instructlab and what is the difference?
Instructlab:
InstructLab is an approachable open source AI community project started by researcher and engineers at Red Hat. The community's mission is to enable anyone to shape the future of generative AI via the collaborative improvement of open source-licensed Granite large language models (LLMs) using InstructLab's fine-tuning technology. InstructLab allows anyone to improve an existing LLM by fine-tuning it with additional data sources. This allows LLMs to continuously gain new knowledge, supplementing gaps in their initial training, even about current events that happened since their pre-training phase.

sdg_hub: SDG Hub is python package designed to simplify data creation for LLMs, allowing users to chain computational units and build powerful flows for generating data and processing tasks. Define complex workflows using nothing but YAML configuration files. sdg_hub originally started as instructlab's sdg component later separating into separate python package. The pacakge is built by the Red Hat innovation team (also the original creators of the instructlab sdg and training packages). The aim of this pacakge is to serve as hub for synthetic data generation methods including the instructlab's skills and knowledge data generation methods.


**Slide 8:**

The slide explains that there has been a shift in how model customization is handled, moving toward a more modular design. The session referenced in the video focused on this new approach and introduced the SDG Hub, which serves as the underlying codebase powering the knowledge and skills pipeline in InstructLab. This distinction is made to clarify the foundation and direction of the current work.

**Slide 9:**

* **Session purpose**: The goal of the current and following session is to walk through the existing *skills pipeline* in InstructLab and RhelAI 1.5. Unlike previous overviews, this time the focus is both on usage and internal structure.

* **Basic usage flow**: Generating skills data is straightforward, users provide a Q\&A YAML file with examples, select a teacher model, and the system handles the rest of the generation process automatically.

* **Customization focus**: The session emphasizes enabling users to *modify the pipeline* to suit their specific needs. For instance, if a user wants to restructure a table or split content differently, they should feel empowered to make those changes themselves.

* **Deep dive into the pipeline**: Although generating skills data is simple, the session will go into detailed explanations of how the pipeline is constructed. This prepares users to not just use, but fully understand and adapt the pipeline.

* **Graduating to SDG Hub**: After learning the structure and modification of existing pipelines, users will transition to using SDG Hub. At this stage, they’ll have access to a more flexible environment where arbitrary pipelines, skills, and knowledge structures can be created with minimal effort.

* **Overall goal**: The end objective is to give users full control so they’re no longer constrained by preset workflows, and small edits can lead to fully customized results.

**Slide 10:**

* **Definition of customization**: The slide sets a shared understanding of what *language model customization* means in this context. It refers to building a workflow where you teach a model to perform specific tasks by feeding it curated data.

* **Seed example generation**: Customization begins with a few seed examples (and optionally related documents). These small sets of examples are then expanded automatically into large training datasets, millions of examples, through augmentation and mixing procedures.

* **Modularization of pipeline components**: A major change is the move from a "black box" system to a fully modular and open setup:

  * Each stage of the pipeline (e.g., data mixing) is now exposed as a *separate module*.

  * Users can inspect, modify, or contribute to individual components like data mixers.

  * This grants more control and transparency compared to the older system where issues couldn’t be debugged or customized easily.

* **Training stage**:

  * After generating the training data, users can perform full fine-tuning using the provided libraries.

  * These libraries include advanced training methods such as GRPO (Guided Reinforcement Policy Optimization), async GRPO, and reference tuning.

  * Reward models and evaluation suites are also available as part of the toolkit to help guide and assess training.

* **Evaluation and checkpointing**: The process includes evaluating intermediate model checkpoints to select the best-performing version before finalizing the customized model.

* **Emphasis on SDG (Synthetic Data Generation)**:

  * In this workflow, SDG is the most resource-intensive and time-consuming phase.

  * Lessons from past challenges in knowledge enablement highlighted the need for a more open and flexible architecture.

  * These experiences were a driving force behind the shift to a modular, open-stack design.

* **Key takeaway**: Whenever the term *customization workflow* is mentioned, it refers to this complete loop: task definition → seed data → data expansion → training → evaluation → model selection — with every step now transparent and configurable.

**Slide 11:**

* **Recap of Knowledge generation**:

  1. Previously, the focus was on *knowledge*, defined as *document-grounded data generation*.

  2. This type of data involves factual information that doesn’t interpolate, e.g., historical facts like names of U.S. presidents.

  3. These are examples where the model must memorize explicit information because it can’t reason or infer it without being told.

* **Introduction to current topic (Skills / Distillation)**:

  1. This session shifts focus from knowledge to *skills*, which aligns with the concept of *distillation*.

  2. Distillation here refers to extracting specific abilities from a larger, more capable teacher model into a smaller or more specialized model.

  3. A simple illustrative example is shown on the slide, solving for variables like “4 \+ x”, to demonstrate task-specific reasoning skills that can be distilled.

* **Key difference in InstructLab’s approach**:

  1. **Fine-grained control**: Unlike general APIs (e.g., LLaMA API or Microsoft’s methods), InstructLab provides a higher level of control over how task-specific synthetic datasets are generated.

     * This allows users to specify data structure and content in much more detail.

  2. **Avoiding mode collapse**:

     * Standard distillation methods often suffer from *mode collapse*, where the teacher model produces outputs that lack diversity and concentrate in a narrow region of its output space.

     * This results in low-quality, repetitive training data and hurts performance.

     * InstructLab’s *skills training* avoids this issue by ensuring diversity in the generated data, leading to more robust and effective training.

* The slide sets the stage for understanding how skills-based distillation in InstructLab differs from conventional approaches, offering greater control and better-quality data without suffering from common pitfalls like mode collapse.

**Slide 12:**

**Core components in SDG_HUB:**

* **Flows**: These act as the *orchestrators* of the synthetic data generation (SDG) pipeline. They define the high-level logic, what tasks are included, in what sequence they occur, and how they connect. Essentially, flows manage the structure and execution of the overall pipeline.

* **Blocks**: Blocks are the *modular, reusable units* that carry out individual functions within the pipeline. Each block is responsible for a specific task such as summarization, keyword extraction, or any arbitrary Python function. They serve as the fundamental components used to build more complex workflows.

* **Prompts**: prompts are used within *LLM blocks* to *instruct the teacher model*. They guide how the LLM behaves, what to generate and the manner in which it should be generated. Prompts define both the content and style of the model’s outputs during data generation.

**Slide 13:**

* The example introduces a practical use case to apply the earlier concepts (flows, blocks, prompts) in a real-world task using a markdown table of student records.

* The input is a **markdown table** that contains basic information: names, grades, and GPA. The goal is to enhance it by adding a new column called **academic status**.

* The goal is to **enhance the table** by having the model *add a new column* that infers each student’s **academic status**.

* The model is tasked with inferring this new column based on GPA values:

  * **GPA \> 3.7** → *Honors*

  * **3.3 ≤ GPA ≤ 3.7** → *Passing*

  * **GPA \< 3.3** → *Probation*

* This involves a **classification task**, where the model determines a student’s academic standing based on their education level and GPA, effectively turning raw data into more actionable insights. This task is implemented as a **flow** that orchestrates the transformation, with logic embedded in **blocks** that handle the inference, and **prompts** guiding the model's behavior during generation.

* The example demonstrates how SDG Hub can automate structured data augmentation and enrichment using customizable pipelines.

**Slide 14:**

* The slide walks through a typical **skills generation flow** in Lab SG:

  * **Step 1: Seed examples \-** Start by collecting a small number of task-specific examples and save them in a Q\&A YAML file.

  * **Step 2: Running the pipeline** \- Use the **ilab sdg** tool to run an existing skills pipeline. For example, if using the *free-form skills* pipeline, you simply trigger generation.

  * **Step 3: Synthetic data generation**  
     The teacher model by default, **Mixtral** generates synthetic data based on the seed examples. At first glance, this data may appear accurate and well-formed.

* However, the key point is that *appearances can be misleading*. A closer inspection often reveals imperfections or subtle issues in the generated data.

* This introduces the idea that while the generation process is easy to run, verifying and refining the quality of outputs is just as important.

**Slide 15:**

* The slide compares two types of model outputs for the same task, highlighting how *the style of the teacher model* affects the generated data.

* **Chatty response (left side)**:

  * Generated using the **Mixtral** model (default in version 1.5).

  * The response is natural and conversational ideal for chatbots.

  * It includes a brief intro and explains its reasoning process.

* **Structured output (right side)**:

  * More suitable when clean, machine-readable output is required, e.g., piping into another system.

  * It contains only the **markdown table**, without extra text or preambles.

* **Takeaway**: If your use case demands structured output over natural language explanations, one way to achieve that is by **switching the teacher model**. For instance, using a LLaMA-based model with more precise instructions can yield the desired format. This underscores the importance of choosing the right teacher model for the task.

**Slide 16:**

* This slide discusses a **limitation in the default skills pipeline** when working with unstructured text that needs to be converted into a structured table.

* **Use case**: Taking something like a user message or support note and extracting structured information (e.g., into columns like *feature*, *feedback*, and *sentiment*).

* **Problem**: While the task seems straightforward and you could technically reuse an existing workflow (e.g., *grounded skills pipeline*), the actual output often misses the mark.

* **Issue with question generation**:

  * The pipeline automatically generates both questions and answers.

  * As a result, even though the data is from the same domain, the **generated questions vary subtly**, leading to inconsistent outputs.

  * For example:

    * One output may prompt a table with columns like *task*, *user preference*, and *urgency*.

    * Another might ask for *customer feedback* and *type*, none of which match the user’s intended schema.

* **Key takeaway**: When you want precise and consistent structure (like fixed column names), automatic question generation in the default pipeline can lead to **drift** in the format. This highlights a need for better control or customization to enforce specific output schemas.

**Q\&A between Slide 16 and Slide 17:**

* **Recap of the issue**: The earlier problem was that the model (e.g., Mixtral or Mistral) generated varying questions instead of consistently producing a table with fixed columns (*feature*, *feedback*, *sentiment*) from unstructured documents. This happened despite providing well-crafted seed examples.

* **Root cause**: The default behavior in the InstructLab pipeline is to let the teacher model both *create new questions* and *answer them*, based on the provided examples. This works well for general data augmentation, but fails when a user wants *control and consistency*, such as generating the *same question repeatedly* with new inputs.

* **Two distinct use cases**:

  * **Use Case 1 (Flexible generation)**: You want the model to learn from examples and produce new, diverse question-answer pairs. For example, learning to manipulate a table in different ways, sometimes adding a column, sometimes removing a row. Question formats are expected to change.

  * **Use Case 2 (Constrained generation)**: You want to reuse the *same question prompt* across multiple inputs and get consistent structured outputs. In this case, you’re not interested in diversity, you want reliability and uniformity in the schema.

* **Objective of the session**: The goal is to teach users how to *customize the data generation pipeline* to fit these needs. This includes:

  * Understanding when and how to constrain the model’s behavior.

  * Overriding default behavior to maintain consistent prompts or outputs.

  * Learning how to apply these constraints within the pipeline setup this will be demonstrated in later sections.

* **Clarification on implementation**: When asked whether these constraints are applied in the pipeline itself or in the Q\&A file, the speaker confirms that the details will be shown soon, and this part serves as a conceptual overview before diving into each use case.

**Slide 17:**

* The plan is to **customize the data generation pipeline** within **SDG (Synthetic Data Generation)**, focusing specifically on the **skills pipeline**.

* The session will cover:

  * **How to switch the teacher model** from **Mixtral** (default) to **LLaMA**, allowing for different output styles or behaviors depending on the task requirements.

  * **How to add new blocks** to the pipeline, specifically where a **static question** needs to be inserted to enforce consistent structure in the generated data.

* This sets up the hands-on customization work that will follow, giving users the tools to tailor the pipeline to their exact needs.

**Slide 18 onwards (Demo and Q\&A at the end of the session included):**

### 

### **Demo Setup & Environment**

* **Pre-requisites**:

  * Ensure Python, pip, git, and Jupyter are installed.

  * Clone the provided demo repo (link shared via Slack).

  * The setup uses a VLM endpoint that mimics OpenAI’s API, so the entire demo can be run locally from a laptop.

* **Demo Notebook**:

  * The focus is on the `table_manipulation.ipynb` notebook.

  * Participants were given \~5 minutes to set up their environment before proceeding.

---

### **Use Case Overview: Table Manipulation Tasks**

Several example tasks were described:

1. **Normalization**: Fix inconsistent entries like “USA”, “United States”, etc., into a uniform format.

2. **Inference**: Add a new column (e.g., "Seniority") based on job titles.

3. **Join Tables**: Merge two tables on a shared key and compute derived metrics like adjusted revenue.

4. **Filtering**: Extract rows based on criteria (e.g., names starting with D or E in Sales with salary \> $7000).

* These tasks are common in post-training scenarios where small language models only achieve \~60% accuracy out-of-the-box.

* The goal is to teach the model to improve that via **data generation and fine-tuning**, not prompt engineering alone.

---

### **How InstructLab Skills Pipeline Works**

1. **Teacher Model Setup**:

   * Any OpenAI-compatible model can be used (e.g., vllm, LLaMA via HuggingFace).

   * For demo: Mixtral and LLaMA 7B were the only supported models in RhelAI 1.5.

   * Public API used; participants were asked not to overload the server.

2. **Seed Example Creation**:

   * Create a YAML file with question-answer pairs representing the task.

   * Guidelines:

     * Be realistic and domain-relevant.

     * Avoid ambiguous or minimal examples.

     * Cover a wide range of sub-tasks (e.g., add, filter, join, infer).

     * Match the expected input/output format, e.g., markdown-only output if that’s the goal.

3. **Token Length Guidance**:

   * No strict token limit for skills.

   * Match example lengths to your target use case (\~200–500 tokens typical).

   * Small vs. large examples: use mixed sizes or create separate Q\&A YAMLs.

4. **Seed → Dataset Conversion**:

   * The YAML is converted into a HuggingFace dataset (JSONL format).

   * Each entry becomes a dictionary with question, answer, and task description.

---

### **Pipeline Execution (Under the Hood)**

1. **Pipeline Flow**:

   * Data flows through **blocks** defined in a YAML file (e.g., `flow_table_manipulation.yaml`).

   * **Block Types**:

     * `LLMBlock` for generation and evaluation.

     * Configs include prompt templates, number of generations per seed, and output column.

2. **Generation Process**:

   * Step 1: Generate new questions from seed data.

   * Step 2: Evaluate generated questions with scoring (0–1 range).

     * Questions with low relevance/formatting are filtered out.

     * Custom Python functions can impose arbitrary constraints (e.g., max length).

3. **Answer Generation & Evaluation**:

   * Each question is passed again to the teacher model for response generation.

   * Graded on a 3-point scale:

     * 1 \= Hallucinated/incorrect

     * 2 \= Correct but slightly off in formatting

     * 3 \= Perfect

   * Responses with score ≥ 2 are retained.

4. **Prompt Structure**:

   * System prompt: defines the role of the model.

   * Introduction, principles, and template examples guide consistent output.

   * Prompts are customizable at each LLM block.

5. **Parallel Execution**:

   * Each seed example is processed independently.

   * Generation and evaluation are parallelized using VLM’s async batching/queuing system for efficiency.

---

### **Demo Output Example**

* A sample output showed:

  * A markdown table about inventory/stocks.

  * A model-inferred column: stock level (low, medium, high based on quantity).

  * A follow-up computation: average price per stock level.

  * Emphasized: different participants may see different examples due to random sampling.

---

### **Customization & Next Steps**

* RhelAI 1.5 only supports Mixtral and LLaMA 7B 3.3 via baked pipelines.

* Customization of internal configurations will be enabled in RhelAI 3.0 via **SDG Hub**.

* Users were encouraged to explore the notebook and start modifying the number of generations, prompt structure, and add filters and custom blocks.

---

### **Questions & Clarifications During the Demo (Detailed)**

* **Q1: How long should question and answer pairs be in the seed examples? Is 250 tokens still a recommended limit?**

  * **A:** There is no hard limit for skills examples. The previous 250-token guideline was tied to earlier models with smaller context windows. With LLaMA 3.3 and similar newer models, much larger inputs are feasible. Instead of sticking to an arbitrary limit, users should **match the token length to their actual use case**. For instance, if real inputs are typically 200 to 500 tokens, use seed examples in that range. If longer inputs are expected (e.g., longer tables or documents), match accordingly. A mixed set of small and large examples is also acceptable and often beneficial.

* **Q2: How many seed examples should you provide for a single, narrowly-scoped use case?**

  * **A:** Even if the use case is narrow (e.g., transforming one specific kind of table), you should still provide *multiple seed examples*. These could reflect small variations in input phrasing or expected output formats. Even with the same task, varying the examples slightly helps the model generalize better. The idea is to create a robust pattern for the model to learn from, rather than training on a single static example.

* **Q3: Do I need to save the generated JSONL data file before moving to the next notebook?**

  * **A:** It is **not necessary to save** the generated file if you're just following along the live demo. However, saving the file can be helpful if you plan to reuse or train on the data later. The demo notebook shows examples using markdown cells, but you can add a line of code to save the JSONL output if needed.

* **Q4: Why does the table in the demo differ from what's in the GitHub repo (e.g., stocks vs. science/math examples)?**

  * **A:** The examples shown during the demo are **randomly selected** from the generated dataset. Because each participant is running the pipeline independently, and the seed examples can generate a range of variations, the output tables may differ across runs. This is expected behavior and highlights the generative nature of the process.

* **Q5: Is it valid to use the *same model* as both the generator and the evaluator in the pipeline? Wouldn’t it be biased?**

  * **A:** This is a common concern, but **using the same model is theoretically justified**. Generation (creating data) and evaluation (discriminating good from bad) are fundamentally different tasks. Generative modeling involves density estimation and is harder, while evaluation is a classification task. Even if a model generates data, it can still reliably judge whether a new sample is high quality or aligned with expectations. However, if desired, users can configure the pipeline to use separate models for generation and evaluation.

* **Q6: Can the pipeline be used for non-English tasks? Do I need to translate prompts manually?**

  * **A:** Yes, the pipeline supports multilingual workflows. LLaMA 7B, for instance, supports several languages. Users can **translate prompts, seed examples, and output formatting** as needed. In future versions (e.g., RhelAI 3.0 with SDG Hub), building entire pipelines in other languages will be straightforward, and translation can be managed within the notebook workflow.