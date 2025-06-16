Chunk - 0
-----

Page 1

The Big Picture

  - What is InstructLab
  - End-to-end PoC Overview

-----

Page 2

This slide visually communicates why enterprises need a **LLM (Large Language Model) Customization Toolkit**. Here's a breakdown of the components and their meaning:

---

### **Title**:

**Enterprise Needs LLM Customization Toolkit**
This establishes the main argument: businesses require tools to tailor LLMs to their specific needs.

### Top-left:

**Icon**: Trophy
**Text**: “Enterprises need performant models for their use case”
This implies that off-the-shelf models may not meet domain-specific or task-specific performance standards as they have never seen enterprise data that is not in public domain.s

### Bottom-left:

**Icon**: Piggy bank
**Text**: “Enterprises need their models to be cost-effective”
Indicates a need for optimization—not just accuracy but also LLM scalibility and response latency. Smaller models are much easier to scale and can be used in longer agentic pipelines


### Center:

**Icon**: Toolkit (pencil, wrench, ruler)
**Text**: “Enterprises need a model customization toolkit”
This represents the solution: a toolkit that lets enterprises customized small LLMs to meet their unique use-cases. Think of agentic applications for dealing with enterprise specific documents and task on top of those.


### ➡️ Right:

**Icon**: Custom robot head
**Text**: “Custom Model”
Shows the desired outcome: a model that is both high-performing and cost-efficient, tailored to the enterprise’s specific use case.


### 🔁 **Overall Equation**:

**(Performance Need + Cost-Effectiveness Need) + Customization Toolkit = Custom Model**

-----
Page 3
**InstructLab: Prescriptive approach for enterprise LLM development**
InstructLab provides a structured, step-by-step method to customize large language models for enterprise use.

- Enterprise need customized models
- Instruclab provides a step-by-step way of generating synthetic data and multi-phase model training 
- The LLM customization can be done through to ways:
  - Skills
  - Knowledge
- For each type we generate synthetic data

-----
Page 4

This slide contrasts **Knowledge** and **Skill** to emphasize how they differ in both nature and cognitive application—especially in the context of training models or humans.

---

### 🔴 **Title**:

**Knowledge vs Skill**
Introduces the main comparison between two core learning dimensions.

---

### 🔧 **Left Side: Skills**

* **Icon**: Code symbol (\</>) – implies logic, reasoning, or computation
* **Label**: "Skills – Patterns, to be generalized!"
* **Examples**:

  1. 1 + x = 3
  2. 2 + x = 4
  3. 3 + x = 5
* **Question**: *What is 4 + x then?*
  This highlights pattern recognition and logical generalization—hallmarks of skill learning.

---

### 📚 **Right Side: Knowledge**

* **Icon**: Books – symbolizing factual content
* **Label**: "Knowledge – Facts, to be memorized!"
* **Examples**:

  1. George Washington (1789–1797)
  2. John Adams (1797–1801)
  3. Thomas Jefferson (1801–1809)
* **Question**: *Name the fourth president?*
  This tests recall of factual information—representing knowledge.

---

### 🧠 **Core Message**:

* **Skills** = Generalize from patterns (used in problem-solving, coding, reasoning tasks)
* **Knowledge** = Recall specific facts (used in historical, encyclopedic, or static retrieval tasks)

This distinction is crucial in domains like LLM development, where training may need to explicitly foster both **pattern-based generalization** (skills) and **factual correctness** (knowledge).

-----

Page 5

This slide illustrates the **InstructLab Workflow** — a detailed pipeline for customizing LLMs using Red Hat’s prescriptive approach, implemented on a **RHEL-AI Node**.

---

### 🔴 **Title**:

**InstructLab Workflow**
Outlines the structured steps for enterprise LLM customization.


### 🧩 **Step-by-Step Pipeline**:

1. **User Provides Documents and Example Seed data**
   ➡️ Defines the scope and structure of knowledge and skills.

2. **SDG (Synthetic Data Generation)**
   ➡️ Uses the provided documents and seed data to generate training data.

3. **Data Mixing**
   ➡️ Format the generated data into training compatible format (messages format) and mix generated knowledge and skills dataset to create final training mix.

4. **Knowledge Tuning**
   ➡️ First training stage to train on generated synthetic data for kowledge on user documents. This stage will teach the model user's document so the model can answer Question Answer based off that document

5. **Evaluation**
   ➡️ Performance is assessed before moving to the next phase. This done using MMLU style evaluation. We generate MMLU style MCQ data on top of the user document for evalution and model selection.

6. **Skills Tuning**
   ➡️ Second training stage to teach reasoning, generalization, and pattern recognition mixed with the generated user specific synthetic skills.

7. **Evaluation (again)**
   ➡️ Ensures skill learning was effective. We do it by doing MT-Bench style evalation on user's seed example/user provided evaluation examples and using LLM as judge.

8. **CKPT Selection**
   ➡️ Best model checkpoints from tuning phases are selected.

9. **Custom Model Output**
   🎯 Final tailored LLM optimized for enterprise use cases. For testing final model, it is deployed in a RAG setting, user provided evalaution examples are used to generated model responses and LLM as Judge is used to evaluate the output.

---

### 🖥️ **Platform**:

**RHEL-AI Node** —Everything is run the RHELAI platform or compute environment where this entire workflow is executed (Red Hat Enterprise Linux for AI workloads).

---

### 🧠 Key Insight:

**InstructLab systematically transforms a user’s taxonomy into a performant, custom LLM**, by blending structured knowledge injection, synthetic data, and multi-stage training and  evaluation—all on enterprise-ready infrastructure.


-----

Page 6

The PoC (Proof-of concept) Loop

The slide features a circular diagram illustrating "The PoC Loop," divided into three numbered steps, each with accompanying text:

**01 Process PDF and write QnA.yaml**
Located at the top-right of the circular flow.
Text: "Study the client document and understand their end use-case. Based on this information, draft a QnA.yaml that captures the type of questions and answers that are expected in the use case."

**02 Generate data and train the model**
Located at the bottom-right of the circular flow.
Text: "Use either the cloud service or your own 8xH100/A100 based RHEL-AI instance to generate the data using your QnA.yaml and then train the model."

**03 Setup RAG and evaluate the model**
Located on the left side of the circular flow.
Text: "Use AnythingLLM or your favorite RAG System, and benchmark the performance of your model against a proprietary model of your choice. Understand what types of questions the model failed to answer and repeat the process by including them in the new QnA.yaml"

Arrows indicate the flow from step 01 to 02, from 02 to 03, and from 03 back to 01, completing the loop.

-----

Chunk - 1

# Slide 1

Example PoC [cite: 1]
SEC Football Trivia [cite: 1]
1 [cite: 1]

→ Which dynamic wide receiver duo from the same team combined for the most receiving yards in 2023, making them the deadliest pairing in the SEC? [cite: 1]

→ Who was the special-teams standout in 2023, with the most kickoff return yards, electrifying the crowd every time he touched the ball? [cite: 1]

# Slide 2

Does GPT-4 know: NO [cite: 2]

Which dynamic wide receiver duo from the same team combined for the most receiving yards in 2023, making them the deadliest pairing in the SEC? [cite: 2]

As of my last update, I don’t have the specific statistics for the 2023 season. To find out which dynamic wide receiver duo from the same team combined for the most receiving yards in the SEC in 2023, I recommend checking the latest stats on sports news websites like ESPN, CBS Sports, or the official SEC website. They provide up-to-date statistics and analysis that would help identify the top-performing wide receiver duo in the SEC for the year 2023. [cite: 2]

Who was the special-teams standout in 2023, with the most kickoff return yards, electrifying the crowd every time he touched the ball? [cite: 2]

I don’t have the specific details for the 2023 season. To find out who the special-teams standout with the most kickoff return yards in 2023 is, you should check the latest statistics on sports news websites like ESPN, CBS Sports, or the official NCAA website. These sources update regularly and will provide the most current and accurate information on standout players in college football for the 2023 season. [cite: 2]

Send a message [cite: 2]

# Slide 3

Does GPT-4 w/ RAG know: NO [cite: 3]

Who was the special-teams standout in 2023, with the most kickoff return yards, electrifying the crowd every time he touched the ball? [cite: 3]

The special-teams standout in 2023 with the most kickoff return yards was Jaylen Wright from Tennessee. [cite: 3]
Show Citations \> [cite: 3]

Which dynamic wide receiver duo from the same team combined for the most receiving yards in 2023, making them the deadliest pairing in the SEC? [cite: 3]

The documents provided do not contain information about receiving yards or details on wide receiver performances, therefore, I cannot determine which wide receiver duo from the same team combined for the most receiving yards in 2023. [cite: 3]
Show Citations \> [cite: 3]

Send a message [cite: 3]

# Slide 4

→ Which dynamic wide receiver duo from the same team combined for the most receiving yards in 2023, making them the deadliest pairing in the SEC? [cite: 4]
→ Who was the special-teams standout in 2023, with the most kickoff return yards, electrifying the crowd every time he touched the ball? [cite: 5]
Does Granite-7B know: [cite: 6]

# Slide 5

Does Granite-7B know: YES [cite: 7, 8]

Who was the special-teams standout in 2023, with the most kickoff return yards, electrifying the crowd every time he touched the ball? [cite: 8]

Barion Brown from Kentucky was the special-teams standout in 2023, with the most kickoff return yards (576), electrifying the crowd every time he touches the ball. [cite: 8]
Show Citations \> [cite: 8]

Which dynamic wide receiver duo from the same team combined for the most receiving yards in 2023, making them the deadliest pairing in the SEC? [cite: 8]

Malik Nabers and Brian Thomas Jr. from LSU combined for the most receiving yards in 2023, with 1,569 and 1,238 yards, respectively. [cite: 8]
Show Citations \> [cite: 8]

Send a message [cite: 8]

This slide contains two bar charts.
The first chart is titled "45% More accurate than GPT-4". It shows a single light red bar with a small dog icon at its base, representing a lower accuracy for GPT-4 compared to an implied higher accuracy for Granite-7B.
The second chart is titled "50x Cheaper than GPT-4". It shows a tall blue bar with a small dog icon at its base, representing a significantly lower cost for Granite-7B compared to GPT-4. An SEC logo is in the top right corner of this chart area.

# Slide 6

The value of enterprise data can be seen in how they make smaller, targeted, optimized models provide state-of-the-art performance at considerably lower cost. [cite: 9]

This slide features a bar chart comparing the performance and cost-effectiveness of different AI models across various enterprise use cases. The chart displays five pairs of bars. In each pair, the left bar (pink) represents a GPT-4 Turbo or Llama model, and the right bar (blue) represents a Granite 7B Lab model. The height of the bars indicates performance (though not explicitly scaled, implied by the "Performance" label on the y-axis which ranges from 50% to 90%). Text above the blue bars indicates the percentage cheaper the Granite 7B Lab model is compared to the alternative.

Use Cases and Cost Savings:

  * **Enterprise: Large financial company** - Q\&A over standard operating procedures for reconciliation process: Granite 7B Lab is 95.7% cheaper. [cite: 9] GPT-4 Turbo performance is \~69%, Granite 7B Lab is \~73%. [cite: 9]
  * **Enterprise: IBM** - Q\&A over standard operating procedures for Quote-to-Cash (Q2C): Granite 7B Lab is 66.7% cheaper. [cite: 9] Llama-3 70B performance is \~61%, Granite 7B Lab is \~66%. [cite: 9]
  * **Enterprise: IBM** - Q\&A over HR policies: Granite 7B Lab is 91.7% cheaper. [cite: 9] Llama 3.1 405B performance is \~59%, Granite 7B Lab is \~60%. [cite: 9]
  * **Enterprise: IBM** - Q\&A over IT software customer support: Granite 7B Lab is 85% cheaper. [cite: 9] GPT-4o performance is \~68%, Granite 7B Lab is \~76%. [cite: 9]
  * **Enterprise: Large telecommunications company** - Analysis of customer call transcripts: Granite 7B Lab is 93.6% cheaper. [cite: 9] Previous approach performance is \~88%, Granite 7B Lab is \~90%. [cite: 9]

Text below the chart states: "\*Results are based on client’s existing prompt format for GPT-4 Turbo, Llama3 70B, Llama3.1 405B, and GPT-4o. Granite 7B Lab model was fine-tuned for the specific use case using InstructLab." [cite: 9]

GOAL: Add your client use-case to this plot\! [cite: 9]

# Slide 7

Approach | [cite: 10]
We evaluated the fine-tuning experience using a single use case to try to achieve an apples-to-apples comparison¹ [cite: 10]

# Slide 8

| InstructLab End to end fine tuning solution [cite: 10] | Google Vertex Manual SDG + Vertex AI fine tuning [cite: 11] | OpenAI Manual SDG + OpenAI fine tuning [cite: 12] | Open Source² Manual SDG + manual fine tuning [cite: 17] |
|---|---|---|---|
| **Generate Training Data** | Use InstructLab automated pipeline to create synthetic test/train dataset [cite: 15] | Manually develop pipeline to generate synthetic test/train dataset [cite: 15] | Manually develop pipeline to generate synthetic test/train dataset [cite: 15] | Manually develop pipeline to generate synthetic test/train dataset using open source models [cite: 15] |
| **Run Fine Tuning Pipeline** | Use MixTral 8x7B as Teacher model and Granite 7B as Student model [cite: 15] \<br\> Fine tune all the 7B weights that constitute the weight matrix of the student model [cite: 15] | Use Gemini 1.5 Pro as Teacher model and Gemini 1.5 Flash as Student model [cite: 15] \<br\> Fine tune a smaller subset [cite: 16] of added parameters using platform tuning pipeline [cite: 15, 16] | Use GPT-4o as Teacher model and GPT-4o-mini as Student model [cite: 16] \<br\> Fine tune a smaller subset of added parameters using platform tuning pipeline [cite: 16] | Use Llama 3.1 as Teacher model and Llama 8b as student model [cite: 16] \<br\> Fine tune a smaller subset of added parameters using open source Lora/Qlora tuning e.g. [cite: 16] Axolotl [cite: 17] |
| **Deploy Model** | Deploy to/host in IBM cloud [cite: 17] | Deploy to/host in Vertex AI [cite: 17] | Deploy to/host in OpenAI [cite: 17] | Self host the model [cite: 17] |
| Image |  |  |  |  |
| | 3 [cite: 15] | 2 [cite: 15] | 1 [cite: 15] | Open Source² [cite: 17] |

¹Vertex and Openai use supervised fine tuning. [https://cloud.google.com/vertex-ai/generative-ai/docs/models/tune-models](https://cloud.google.com/vertex-ai/generative-ai/docs/models/tune-models) [cite: 13]
Experience based on BCG client case. [cite: 13]
Fine tuning of Open Source model is not part of the current experiments [cite: 14]
Note: Model performance will not be considered, due to differences in model performance and platform offering. [cite: 14]
Based on current research, Gemini 1.5Pro and GPT-4o are among the top models, while MixTral 8x7B lagging behind [cite: 15]

# Slide 9

Results | InstructLab could potentially reduce end-to-end model customization timeline by \~50% [cite: 18]

This slide presents a horizontal bar chart illustrating the time spent in LLM fine-tuning in days, comparing three approaches: Open Source4, Vertex/OpenAI, and InstructLab. Each bar is segmented into three parts representing different stages: 'Generate training data1' (light green), 'Run fine tuning pipeline2' (medium green), and 'Deploy model3' (dark green).

  * **Open Source4**: The bar shows 10 days for 'Generate training data', 10 days for 'Run fine tuning pipeline', and 5 days for 'Deploy model', totaling 25 days. [cite: 18, 19]
  * **Vertex/OpenAI**: The bar shows 10 days for 'Generate training data', 1 day for 'Run fine tuning pipeline', and 1 day for 'Deploy model', totaling 12 days. [cite: 18, 19]
  * **InstructLab**: The bar shows 5 days for 'Generate training data', 2 days for 'Run fine tuning pipeline', and 1 day for 'Deploy model', totaling 8 days. [cite: 18, 19]

**Key Takeaways** [cite: 19]

InstructLab could enable companies to generate value faster by reducing GenAI model fine tuning time (\~50%) [cite: 19]

Reduction is driven by introduction of synthetic data pipeline and streamlined model fine tuning process [cite: 19]

InstructLab deployment timeline could be reduced from 5 days to 1 day if model is deployed to IBM cloud [cite: 19]

Time spent in LLM fine tuning (days) [cite: 19]
Note: For an apples-to-apples comparison, we will need to test on the same/comparable tasks, with same model, and GPU power [cite: 19]

1.  Generating training data includes time spent coding data and inspecting quality; [cite: 19, 20]
2.  Fine tuning includes time spent training and debugging; 3. Deploying model includes time spent self hosting; [cite: 20]
3.  Based on BCG case experience [cite: 21]

<!-- end list -->

```
Time to solution
}
LoRA Supervised fine tuning
QLoRa fine tuning
Full weight training
```

[cite: 21]
1 [cite: 21]


Chunk - 2

Page 2

Process PDF and write QnA.yaml

01
03
02
Part 1: Data Preparation
From raw documents to data that LLMs can learn from
Data Preparation

-----

Page 3

Data Preparation
3

3
3

-----

Page 4

Data Preparation

Seed Data
User Document

User QNA
yaml
Generate Seed data

Chunks
Docling Chunking
Step 1
Step 2

-----

