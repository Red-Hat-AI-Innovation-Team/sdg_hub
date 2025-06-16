This slide outlines **Part 1: Document Pre-Processing** in the **Data Preparation and Generation** phase. It shows how to transform raw documents into structured **seed data** for synthetic data generation in instructlab.

---

### 🔴 **Title**:

**Part 1: Data Preparation and Generation: Document Pre-Processing**

---

### 🔹 **Bullet Instruction**:

* **Two Steps**:

  1. Break documents into chunks *(e.g., using a tokenizer or chunking tool)*.
  2. Combine those chunks with a `qna.yml` file to generate **seed data** *(i.e., context + questions)*.

---

### 📊 **Flow Diagram Explanation**:

1. **User Documents**

   * Input: Raw text documents (\~1000 tokens each).

2. **Document Chunking**

   * Parse the document into markdow using your favorite parsing library (like Docling)
   * Breaks documents into logical **chunks** (paragraphs, sections).
   * Outputs a list of document elements.

3. **Combine Outline and Chunks**

   * Merges structural outline (from QnA) with text chunks.

4. **Chunk Element Over 500 Tokens**

   * Chunks longer than 500 tokens are split into smaller parts.

5. **User QNA File (`qna.yml`)**

   * This file includes n-number of user provided context (each has 3 QA pairs). This demonstrates type of questions that are asked on user provided document and how to answer them. This file is calle qna.yaml. This file also includes `document_outline` field that gives a short description/title of the document (provides additional context to each chunk).

6. **Context-Chunks Join**

   * Combines QnA.yaml context-3QA (user's seed data) with document chunks:

     * Example: 2 × 500-token chunks + 3 x context-3QA = 6 final “context-3QA + chunk” pairs.

7. **Seed Data**

   * Final output: data in the format usable for synthetic data generation.

---

### 🧠 **Key Insight**:

This slide emphasizes that:

* LLMs need well-structured, bite-sized instruction data.
* This is done by chunking source documents and enriching them with human-curated context (via `qna.yml`).
* The result is **seed data**, ready for synthetic generation or tuning.

-----

Page 5

This slide provides a simplified visual of the **Data Preparation** workflow in two main steps, specifically for generating **seed data** from user documents using chunking and a QnA YAML file.

---

### 🔴 **Title**:

**Data Preparation**

---

### 🧩 **Step-by-Step Breakdown**:

#### ✅ **Step 1 – Document Chunking**:

* **Input**:

  * `User Document` (raw text, e.g., a 1000-token file)
* **Process**:

  * Run through the **Document Parsing** tool (like Docling, unstructured.io, llama-index pdf parser etc.) followed by document chunking (divide documents that are larger than certain threshold). Docling also provided parsed sections that can be used to construct document chunks that are more complete (like paragraphs under section ending up together instead of getting divided into two chunks)
* **Output**:

  * Produces `Chunks` — smaller text sections (e.g., paragraphs, sections) from the document

---

#### ✅ **Step 2 – Generate Seed Data**:

* **Inputs**:

  * `Chunks` from Step 1
  * `User QNA yaml` — a file that provides:

    * Context + 3x Questions-Answr
    * Document Outline: Document title/short description for extra context
    * Path to user documents
* **Process**:

  * Combine the chunks and QNA YAML to create final seed data
* **Output**:

  * `Seed Data` — structured pairs of questions and context ready combined with document chunk for Synthetic data generation (SDG). During SDG the teacher model looks at user provided context + QA to generate similar data on top of the provided document chunk

---

### 🧠 **Key Insight**:

This diagram demonstrates how domain-specific documents and human-curated questions can be efficiently transformed into **instruction-tuning seed data** for generating synthetic data for teaching a LLM user's domain knowledge (user document). The process is modular and repeatable.


-----

Page 5

Data Preparation: Step 1 Document Pre-processing: Parsing using Docling Package

Docling = PDF->Markdown conversion + structure info

Docling will parse the input document into document elements with metadata. 

This includes :
  * the type of element 
    (paragraph, title, table, subtitle, page-footer, figure etc.)
  * Page number
  * Text, actual value of the element
  * Bounding box to locate element in the original document
  * Tables parsed into markdown format
  * Pointer to image any detected images


-----

Page 6

Docling Parsing Example

An example of **document parsing and metadata extraction** from a financial PDF using a structured annotation format (likely for LLM training or fine-tuning).

---

Document Segmentation + Metadata Extraction

---

### Source Document (BNP Paribas Financial Report)**:

A real page from BNP Paribas's financial statement:

* **Section Title**:
  `1. SUMMARY OF MATERIAL ACCOUNTING POLICIES APPLIED BY THE GROUP`

* **Subsection**:
  `1.a APPLICABLE ACCOUNTING STANDARDS`
  Highlighted in red as the focus of this parsing example.

* **Paragraph Content**:
  Describes the group’s compliance with IFRS and IAS standards and explicitly excludes IAS 39 for hedge accounting.

---

### Extracted JSON Metadata Blocks**:

1. **Section Header Block**:

```json
{
  "text": "1.a APPLICABLE ACCOUNTING STANDARDS",
  "type": "subtitle-level-1",
  "name": "Section-header",
  "page": 10,
  "span": [0, 41],
  "prov": [{"bbox": [...]}]
}
```

* Interprets the heading as a level-1 subtitle and marks its span and position on the page.

2. **Paragraph Block**:

```json
{
  "text": "The consolidated financial statements of the BNP Paribas Group...",
  "type": "paragraph",
  "name": "text",
  "page": 10,
  "span": [0, 316],
  "prov": [{"bbox": [...]}]
}
```

* Extracts the body content under the section header.
* Classifies it as a paragraph with bounding box (bbox) coordinates for layout-aware processing.

---

### 📌 **Purpose & Insight**:

* **This visual demonstrates**:
  How raw documents are converted into machine-readable structured JSON — capturing hierarchy (`subtitle-level-1`, `paragraph`), content, location, and context.

* **Why it matters**:
  Such structured data forms the basis for:

  * Chunking and context linking
  * Instruction generation
  * Semantic search
  * LLM fine-tuning

-----

Page 7

### Chunk Creation:

The parsed list of elements from Docling will be used to create context-aware chunks with the following approach:
  * Each chunk will consist of a list of document elements, ensuring it stays under 500 tokens. 
  * Elements exceeding 500 tokens are placed into individual chunks. 
  * Metadata, such as element-type, is utilized to apply additional formatting during the construction of the final chunks. 

-----

Page 8

### Example of Chunking:

##### Input Document:

```text
## Critical Accounting Estimates 
The application of GAAP requires IBM to make estimates and assumptions about certain items and future events that directly affect its reported financial condition. The accounting estimates and assumptions discussed in this section are those that we consider to be the most critical to our financial statements. An accounting estimate is considered critical if both (a) the nature of the estimate or assumption is material due to the levels of subjectivity and judgment involved, and (b) the impact within a reasonable range of outcomes of the estimate and assumption is material to IBM's financial condition. Senior management has discussed the development, selection and disclosure of these estimates with the Audit Committee of IBM's Board of Directors. Our significant accounting policies are described in note A, "Significant Accounting Policies."
A quantitative sensitivity analysis is provided where that information is reasonably available, can be reliably estimated and provides material information to investors. The amounts used to assess sensitivity (e.g., 1 percent, 10 percent, etc.) are included to allow users of the financial statements to understand a general direction cause and effect of changes in the estimates and do not represent management's predictions of variability. For all of these estimates, it should be noted that future events rarely develop exactly as forecasted, and estimates require regular review and adjustment
## Pension Assumptions 
For our defined benefit pension plans, the measurement of the benefit obligation to plan participants and net periodic pension (income)/cost requires the use of certain assumptions, including, among others, estimates of discount rates, interest crediting rates and expected return on plan assets. Beginning in 2024, as a result of changes to the Qualified PPP as discussed in "Looking Forward" the interest crediting rate and expected return on plan assets will be based on their relationship to the plan's discount rate.
Changes in the discount rate and the interest crediting rate assumptions would impact the service cost, (gain)/loss amortization and interest cost components of the net periodic pension (income)/cost calculation and the projected benefit obligation (PBO). Changes in the expected long-term return on plan assets assumption impacts the net periodic pension (income)/cost. Expected returns on plan assets are calculated based on the market-related value of plan assets, which recognizes changes in the fair value of plan assets systematically over a five-year period in the expected return on plan assets line in net periodic pension (income)/cost. The differences between the actual return on plan assets and the expected long-term return on plan assets are recognized over five years in the expected return on plan assets line in net periodic pension (income)/cost and also as a component of actuarial (gains)/ losses, which are recognized over the service lives or life expectancy of the participants, depending on the plan, provided such
```
-----

Page 9

### Naive Chunking Output:

#### Chunk 1
```text
## Critical Accounting Estimates 
The application of GAAP requires IBM to make estimates and assumptions about certain items and future events that directly affect its reported financial condition. The accounting estimates and assumptions discussed in this section are those that we consider to be the most critical to our financial statements. An accounting estimate is considered critical if both (a) the nature of the estimate or assumption is material due to the levels of subjectivity and judgment involved, and (b) the impact within a reasonable range of outcomes of the estimate and assumption is material to IBM's financial condition. Senior management has discussed the development, selection and disclosure of these estimates with the Audit Committee of IBM's Board of Directors. Our significant accounting policies are described in note A, "Significant Accounting Policies."
A quantitative sensitivity analysis is provided where that information is reasonably available, can be reliably estimated and provides material information to investors. The amounts used to assess sensitivity (e.g., 1 percent, 10 percent, etc.) are included to allow users of the financial statements to understand a general direction cause and effect of changes in the estimates and do not represent management's predictions of variability. For all of 
```

#### Chunk 2
```text
these estimates, it should be noted that future events rarely develop exactly as forecasted, and estimates require regular review and adjustment
## Pension Assumptions 
For our defined benefit pension plans, the measurement of the benefit obligation to plan participants and net periodic pension (income)/cost requires the use of certain assumptions, including, among others, estimates of discount rates, interest crediting rates and expected return on plan assets. Beginning in 2024, as a result of changes to the Qualified PPP as discussed in "Looking Forward" the interest crediting rate and expected return on plan assets will be based on their relationship to the plan's discount rate.
Changes in the discount rate and the interest crediting rate assumptions would impact the service cost, (gain)/loss amortization and interest cost components of the net periodic pension (income)/cost calculation and the projected benefit obligation (PBO). Changes in the expected long-term return on plan assets assumption impacts the net periodic pension (income)/cost. Expected returns on plan assets are calculated 
```

#### Chunk 3
```text
based on the market-related value of plan assets, which recognizes changes in the fair value of plan assets systematically over a five-year period in the expected return on plan assets line in net periodic pension (income)/cost. The differences between the actual return on plan assets and the expected long-term return on plan assets are recognized over five years in the expected return on plan assets line in net periodic pension (income)/cost and also as a component of actuarial (gains)/ losses, which are recognized over the service lives or life expectancy of the participants, depending on the plan, provided such
```
-----

Page 10

### Context Aware Chunking (What we want…)

#### Chunk 1
```text
## Critical Accounting Estimates 
The application of GAAP requires IBM to make estimates and assumptions about certain items and future events that directly affect its reported financial condition. The accounting estimates and assumptions discussed in this section are those that we consider to be the most critical to our financial statements. An accounting estimate is considered critical if both (a) the nature of the estimate or assumption is material due to the levels of subjectivity and judgment involved, and (b) the impact within a reasonable range of outcomes of the estimate and assumption is material to IBM's financial condition. Senior management has discussed the development, selection and disclosure of these estimates with the Audit Committee of IBM's Board of Directors. Our significant accounting policies are described in note A, "Significant Accounting Policies."
A quantitative sensitivity analysis is provided where that information is reasonably available, can be reliably estimated and provides material information to investors. The amounts used to assess sensitivity (e.g., 1 percent, 10 percent, etc.) are included to allow users of the financial statements to understand a general direction cause and effect of changes in the estimates and do not represent management's predictions of variability. For all of these estimates, it should be noted that future events rarely develop exactly as forecasted, and estimates require regular review and adjustment
```

#### Chunk 2
```text
## Pension Assumptions 
For our defined benefit pension plans, the measurement of the benefit obligation to plan participants and net periodic pension (income)/cost requires the use of certain assumptions, including, among others, estimates of discount rates, interest crediting rates and expected return on plan assets. Beginning in 2024, as a result of changes to the Qualified PPP as discussed in "Looking Forward" the interest crediting rate and expected return on plan assets will be based on their relationship to the plan's discount rate.
Changes in the discount rate and the interest crediting rate assumptions would impact the service cost, (gain)/loss amortization and interest cost components of the net periodic pension (income)/cost calculation and the projected benefit obligation (PBO). Changes in the expected long-term return on plan assets assumption impacts the net periodic pension (income)/cost. Expected returns on plan assets are calculated based on the market-related value of plan assets, which recognizes changes in the fair value of plan assets systematically over a five-year period in the expected return on plan assets line in net periodic pension (income)/cost. The differences between the actual return on plan assets and the expected long-term return on plan assets are recognized over five years in the expected return on plan assets line in net periodic pension (income)/cost and also as a component of actuarial (gains)/ losses, which are recognized over the service lives or life expectancy of the participants, depending on the plan, provided such
```
As shown above, our goal is to **chunk the document intelligently**, ensuring that:

* **Sections, paragraphs, code blocks, or tables are not split abruptly**
* **Chunk boundaries align with the natural structure of the document**
* By leveraging **Docling’s document layout understanding and segmentation**, we can accurately identify structural elements and use this metadata to generate **context-aware, semantically coherent chunks**

-----

### Docling Parsing Issues and Things to look at in parsed output

Potential Issues:
  1. Parsing Errors:
    * Docling may occasionally fail to parse text correctly, leading to inaccurate element detection.
      * Example: A single paragraph might be erroneously identified as multiple separate elements.
  2. Incorrect Table Parsing:
    * Errors in parsing tables can result in tables being split or misinterpreted, potentially altering the represented information.

-----

Page 3

Docling Parsing

Best Practices:
* Users should carefully review the generated chunks to ensure there is no parsing or conversion error. 
* Verify that:
  * No sentences are partially cut off in the chunks. 
  * Tables and other structured data are complete and correctly included. 
  * This step ensures the integrity of the chunks and prevents errors from affecting downstream processes. 

-----

Page 4

Context Aware Chunking

Once the docking parsing is done we take a list of elements and: 
  * Format the elements: 
    * Highlight: titles, subtitles, page-header/footers 
  * Stitch elements that are separated by page numbers 
  * Fuse elements that are within max token limit 
  * Keep elements larger than max token limit intact in individual chunk 
  * Final list of chunks is now passed through a naive chunker to chunk elements that are over the max token limit. 

-----

page: 1

# Data Preparation Step 2: Seed Data Creation

-----

page: 2

## Create Seed-data

After parsing with Docling, integrate chunks with user Q\&A in two steps:

* Step 1:

  * Contextual Enrichment:

    1. For each document chunk, it is combined with the document outline to add additional context. 

    2. E.g. If the chunk represents a table, it is paired with an outline that provides background information about the table’s origin. 

      * For instance, a table on company performance would be paired with an outline like “IBM performance in 2024”. 

    3. This added context enables the teacher model to generate more accurate Q\&A pairs that align closely with the document's content. 

-----

page: 3

## Basic Structure of User QNA Yaml for Synthetic Knowledge data generation from user's document

```yaml
domain: <domain>
document_outline: <A one to two line description of the document>
seed_examples:
  - context: <context 1 goes here>
    question_and_answers:
      - question: <question 1.1 goes here>
        answer: <answer 1.1 goes here>
      - question: <question 1.2 goes here>
        answer: <answer 1.2 goes here> [cite: 6]
      - question: <question 1.3 goes here>
        answer: <answer 1.3 goes here>
  - context: <context 2 goes here>
    question_and_answers:
      - question: <question 2.1 goes here>
        answer: <answer 2.1 goes here>
      - question: <question 2.2 goes here>
        answer: <answer 2.2 goes here>
      - question: <question 2.3 goes here>
        answer: <answer 2.3 goes here> [cite: 7]
...
```

-----

page: 4

### Create Seed-data

Step 2:

1. Combination Generation:

  * All user-provided outlines are combined with all the document chunks to generate every possible combination of user context and document data.

  * Each unique combination results in a distinct set of Q\&A pairs. 

This method ensures that:

  * the Q\&A pairs are contextually accurate

  * reflective of the true information in the document,

while leveraging user-provided input to enhance relevance and precision. 

-----
page: 5

### Seed Example x Chunks

- Each seed example in user qna is now combined with all document chunks
- So for `n` chunks for user's document and `m` provided seed examples from user qna results in `mxn` rows in seed data

#### From User QNA:
```yaml
domain: <domain>
document_outline: <A one to two line description of the document>
seed_examples:
  - context: <context 1 goes here>
    question_and_answers:
      - question: <question 1.1 goes here>
        answer: <answer 1.1 goes here>
      - question: <question 1.2 goes here>
        answer: <answer 1.2 goes here>
      - question: <question 1.3 goes here> [cite: 11]
        answer: <answer 1.3 goes here>
  - context: <context 2 goes here>
    question_and_answers:
      - question: <question 2.1 goes here>
        answer: <answer 2.1 goes here>
      - question: <question 2.2 goes here>
        answer: <answer 2.2 goes here>
      - question: <question 2.3 goes here>
        answer: <answer 2.3 goes here> [cite: 12]
...
```

Results:
seed example 1
```yaml
  - context: <context 1 goes here>
    question_and_answers:
      - question: <question 1.1 goes here>
        answer: <answer 1.1 goes here>
      - question: <question 1.2 goes here>
        answer: <answer 1.2 goes here>
      - question: <question 1.3 goes here> [cite: 11]
        answer: <answer 1.3 goes here>
```
added with all m chunks

- Similar process is followed for all seed examples resulting in seed data of size `m x n`

-----

page: 6

## Anatomy of Knowledge QNA yaml

#### Example QNA yaml:
```yaml
domain: 3D Motion Transfer
document_outline: Towards High-Quality 3D Motion Transfer with Realistic Apparel Animation
seed_examples:
  - context: "\#\# 1. Nucleon-nucleon interactions and three-body forces\*\*\\n\\nAt densities\\n\\u2273 10 2 - n0, neutron matter cannot be described by s-wave interactions\\nonly, and higher partial waves have to be included in the interaction, see Fig.\\n2. From this figure, we see the the equation of state of neutron matter obtained\\nwith s-wave interactions only and when perturbatively including p-wave contributions\\nstart to differ at higher densities.\\n\\nTo describe nucleon-nucleon scattering\\nin various partial waves, nucleon-nucleon interactions are usually dependent\\non the relative spin and isospin state of the nucleons. A large amount of empirical\\ninformation about the nucleon-nucleon scattering problem has been accumulated.\\nIn 1993, the Nijmegen group analyzed all nucleon-nucleon scattering data below\\n350 MeV published in physics journals between 1955 and 1992 (Stoks et al., 1993).\\nOlder nucleon-nucleon interaction models that fit the Nijmegen database with\\na \\u03c72C/N\\ndata \\u223c1 are called ‘phenomenological’ and fit experimental\\ndata very accurately. Examples of these interactions are the Nijmegen models\\n(Stoks et al., 1994) (Nijm93, Nijm I, Nijm II and Reid-93), Argonne models (Wiringa\\nand Pieper, 2002; Wiringa et al., 1995) and the CD-Bonn potential (Machleidt,\\n2001).\\n"
    question_and_answers:
      - question: What technique did the authors propose for separating mixtures with known numbers of sound sources in the USS task?
        answer: The authors proposed using the iterative improved time-dilated convolutional network (ITDCN++) for separating mixtures with known numbers of sound sources.
      - question: How does the conditional information about sound classes improve universal sound separation performance?
        answer: The conditional information about which sound classes are present helps to improve the accuracy and performance of universal sound separation.
      - question: What problem does the mixture invariant training (MixIT) approach aim to solve, and how does it address this problem?
        answer: The MixIT approach aims to reduce the cost of annotating data for sound separation models by using a purely unsupervised training paradigm with unlabeled, in-the-wild data.
```
### Break Down of this yaml:

1. Document Outline:
  ```yaml
  document_outline: Towards High-Quality 3D Motion Transfer with Realistic Apparel Animation
  ```
  - The document outline enhances synthetic data generation by providing contextual specificity for each document chunk, ensuring relevance and alignment during knowledge tuning.

2. domain:
  ```yaml
  domain: 3D Motion Transfer
  ```
  - Domains guide the teacher model to create subject-specific Q&A pairs.

3. context: 
  ```yaml
    - context: "\#\# 1. Nucleon-nucleon interactions and three-body forces\*\*\\n\\nAt densities\\n\\u2273 10 2 - n0, neutron matter cannot be described by s-wave interactions\\nonly, and higher partial waves have to be included in the interaction, see Fig.\\n2. From this figure, we see the the equation of state of neutron matter obtained\\nwith s-wave interactions only and when perturbatively including p-wave contributions\\nstart to differ at higher densities.\\n\\nTo describe nucleon-nucleon scattering\\nin various partial waves, nucleon-nucleon interactions are usually dependent\\non the relative spin and isospin state of the nucleons. A large amount of empirical\\ninformation about the nucleon-nucleon scattering problem has been accumulated.\\nIn 1993, the Nijmegen group analyzed all nucleon-nucleon scattering data below\\n350 MeV published in physics journals between 1955 and 1992 (Stoks et al., 1993).\\nOlder nucleon-nucleon interaction models that fit the Nijmegen database with\\na \\u03c72C/N\\ndata \\u223c1 are called ‘phenomenological’ and fit experimental\\ndata very accurately. Examples of these interactions are the Nijmegen models\\n(Stoks et al., 1994) (Nijm93, Nijm I, Nijm II and Reid-93), Argonne models (Wiringa\\nand Pieper, 2002; Wiringa et al., 1995) and the CD-Bonn potential (Machleidt,\\n2001).\\n"
  ```
  - A ≤500-token chunk from the user document, showcasing unique elements to guide the teacher model in handling varied text types present in user document.

4. question_and_answers:
  ```yaml
    question_and_answers:
      - question: What technique did the authors propose for separating mixtures with known numbers of sound sources in the USS task?
        answer: The authors proposed using the iterative improved time-dilated convolutional network (ITDCN++) for separating mixtures with known numbers of sound sources.
      - question: How does the conditional information about sound classes improve universal sound separation performance?
        answer: The conditional information about which sound classes are present helps to improve the accuracy and performance of universal sound separation.
      - question: What problem does the mixture invariant training (MixIT) approach aim to solve, and how does it address this problem?
        answer: The MixIT approach aims to reduce the cost of annotating data for sound separation models by using a purely unsupervised training paradigm with unlabeled, in-the-wild data.
  ```
  - **Three Q&A pairs** grounded in the context, demonstrating to the teacher model the style and structure of questions and answers to generate for the given text type.

-----

page: 7

## Importance of document outline: Illustrated with an example

- Below is an example of table taken from a company's annual report.
- Wihtout the additional context it is with very hard to tell what company's cash activities

e.g:

Snippets **Without** Document outline

```text

## Cash Flow and Liquidity Trends

|                                                                                 | ($ in billions)   | ($ in billions)   | ($ in billions)   |
|---------------------------------------------------------------------------------|-------------------|-------------------|-------------------|
|                                                                                 | 2023              | 2022              | 2021              |
| Net cash from operating activities [cite: 21]                                              | $ 13.9 [cite: 22]           |$  10.4 [cite: 23]           | $  12.8 [cite: 24]           |
| Cash and cash equivalents, restricted cash and short-term marketable securities [cite: 25] | $ 13.5 [cite: 26]           |$  8.8 [cite: 27]            | $  7.6 [cite: 28]            |
| Committed global credit facilities  (1) [cite: 29]                                         | $ 10.0 [cite: 30]           |$  10.0 [cite: 31]           | $  10.0 [cite: 32]           |

```


Snippets **With** Document outline

```text
Document Outline: IBM Annual Report 2023: Financial Insights and Strategic Overview 
```

```text

## Cash Flow and Liquidity Trends

|                                                                                 | ($ in billions)   | ($ in billions)   | ($ in billions)   |
|---------------------------------------------------------------------------------|-------------------|-------------------|-------------------|
|                                                                                 | 2023              | 2022              | 2021              |
| Net cash from operating activities [cite: 21]                                              | $ 13.9 [cite: 22]           |$  10.4 [cite: 23]           | $  12.8 [cite: 24]           |
| Cash and cash equivalents, restricted cash and short-term marketable securities [cite: 25] | $ 13.5 [cite: 26]           |$  8.8 [cite: 27]            | $  7.6 [cite: 28]            |
| Committed global credit facilities  (1) [cite: 29]                                         | $ 10.0 [cite: 30]           |$  10.0 [cite: 31]           | $  10.0 [cite: 32]           |

```

-----

Page 1

## Anatomy of Knowledge QNA yaml (Continued ...)



```yaml
domain: 3D Motion Transfer
document_outline: Towards High-Quality 3D Motion Transfer with Realistic Apparel Animation
seed_examples:
  - context: "\#\# 1. Nucleon-nucleon interactions and three-body forces\*\*\\n\\nAt densities\\n\\u2273 10 2 - n0, neutron matter cannot be described by s-wave interactions\\nonly, and higher partial waves have to be included in the interaction, see Fig.\\n2. From this figure, we see the the equation of state of neutron matter obtained\\nwith s-wave interactions only and when perturbatively including p-wave contributions\\nstart to differ at higher densities.\\n\\nTo describe nucleon-nucleon scattering\\nin various partial waves, nucleon-nucleon interactions are usually dependent\\non the relative spin and isospin state of the nucleons. A large amount of empirical\\ninformation about the nucleon-nucleon scattering problem has been accumulated.\\nIn 1993, the Nijmegen group analyzed all nucleon-nucleon scattering data below\\n350 MeV published in physics journals between 1955 and 1992 (Stoks et al., 1993).\\nOlder nucleon-nucleon interaction models that fit the Nijmegen database with\\na \\u03c72C/N\\ndata \\u223c1 are called ‘phenomenological’ and fit experimental\\ndata very accurately. Examples of these interactions are the Nijmegen models\\n(Stoks et al., 1994) (Nijm93, Nijm I, Nijm II and Reid-93), Argonne models (Wiringa\\nand Pieper, 2002; Wiringa et al., 1995) and the CD-Bonn potential (Machleidt,\\n2001).\\n"
    question_and_answers:
      - question: What technique did the authors propose for separating mixtures with known numbers of sound sources in the USS task?
        answer: The authors proposed using the iterative improved time-dilated convolutional network (ITDCN++) for separating mixtures with known numbers of sound sources.
      - question: How does the conditional information about sound classes improve universal sound separation performance?
        answer: The conditional information about which sound classes are present helps to improve the accuracy and performance of universal sound separation.
      - question: What problem does the mixture invariant training (MixIT) approach aim to solve, and how does it address this problem?
        answer: The MixIT approach aims to reduce the cost of annotating data for sound separation models by using a purely unsupervised training paradigm with unlabeled, in-the-wild data.
```

- The “context” blocks for “seed\_examples” should strive for variability and diversity of what the system will find in the provided docs. 
- The goal is to cover all possible ways the information is presented in the docs (e.g. paragraphs, different kind of tables, lists of rules, processes, definitions, etc). 

Each new section can use the same context text but feature its own set of Q\&A pairs if you need to: 
  * Include different types of questions for that context,
  or
  * Provide three question-and-answer pairs for a single context. 
This allows for greater flexibility in organizing and presenting information related to a specific topic or scenario. 

- When providing an answer, use complete sentences and avoid single-word responses. 
- The answer should begin by referencing the question being addressed. 
- For instance, if asked "Why is the sky blue?", start your response with "The sky is blue because..." and then continue with the explanation. 

-----

### General Guidelines 
- If you are trying to add multiple documents, add one qna.yaml file per domain and add a minimum of one piece context + 3 Q\&A pairs. 
- This should fall under the best practice of including 750 total tokens. 
- Each context can represent knowledge from different reference documents to cover the variety of documents best. 
- However, if there is a document that is more important, feel free to use more content from it. 

-----

Page 3

### Creating Effective Document Outlines for Synthetic Data Generation

**Purpose of Document Outlines**
  - Provides additional context to each chunk of the knowledge document. 
  - Ensures each generated synthetic data sample that is based on document chunk has necessary background information 

**Benefits of Effective Document Outlines**
  - Enables precise and relevant synthetic data generation. 
  - Improves contextual understanding for downstream tasks. 
  - Facilitates alignment with specific sections of the document. 

**Key Considerations**
  - Be Specific: Provide as much detail as possible without overloading the title. 
  - Avoid Ambiguity: Expand any acronyms or vague references. 
  - Contextual Alignment: Ensure the outline aligns with the document's structure and focus areas. 
  - Differentiate Editions: Include publication dates, volumes, or any identifiers to distinguish between versions. 

-----

Page 4

## Let go over some common mistakes in design of knowledge qna yaml and how to fix them.

### Document Outline Example: Datasheet (Bad design or first attempt)
- user qna yaml showing college football data stats.
```yaml
created_by: abhi1092
version: 3
domain: College Football
document_outline: SEC: Overall Statistics
seed_examples:
  - context: |
      ## Leaders / Offense
      ### Rushing
      | Rank | Player             | Team        | G  | ATT | YDS  | AVG | TD | LONG | AVG/G |
      |------|--------------------|-------------|----|-----|------|-----|----|------|-------|
      | 1    | Schrader, Cody     | Missouri    | 13 | 276 | 1627 | 5.9 | 14 | 384  | 125.2 |
      | 2    | Daniels, Jayden    | LSU         | 12 | 135 | 1134 | 8.4 | 10 | 416  | 94.5  |
      | 3    | Davis, Ray         | Kentucky    | 12 | 199 | 1129 | 5.7 | 21 | 378  | 94.1  |
      | 4    | Judkins, Quinshon  | Ole Miss    | 13 | 271 | 1158 | 4.3 | 17 | 269  | 89.1  |
      | 5    | Wright, Jaylen     | Tennessee   | 12 | 137 | 1013 | 7.4 | 4  | 377  | 84.4  |
      | 6    | Hunter, Jarquez    | Auburn      | 12 | 159 | 909  | 5.7 | 7  | 311  | 75.8  |
      | 7    | Edwards, Daijun    | Georgia     | 12 | 165 | 880  | 5.3 | 13 | 245  | 73.3  |
      | 8    | Mcclellan, Jase    | Alabama     | 13 | 180 | 890  | 4.9 | 8  | 250  | 68.5  |
      | 9    | Etienne, Trevor    | Florida     | 11 | 131 | 753  | 5.7 | 9  | 267  | 68.5  |
      | 10   | Johnson Jr., Montrell| Florida   | 12 | 152 | 817  | 5.4 | 6  | 253  | 68.1  |
    questions_and_answers:
      - question: |
          Which team did the top rusher Cody Schrader play for in 2023, and what were his overall statistics?
        answer: |
          Cody Schrader played for Missouri. He played 13 games, made 276 rushing attempts, gained 1,627 yards, averaged 5.9 yards per attempt, scored 14 touchdowns, had a longest rush of 384 yards, and averaged 125.2 yards per game.
      - question: |
          Which team did Jayden Daniels play for in 2023, and what were his overall statistics?
        answer: |
          Jayden Daniels played for LSU. He played 12 games, made 135 rushing attempts, gained 1,134 yards, averaged 8.4 yards per attempt, scored 10 touchdowns, had a longest rush of 416 yards, and averaged 94.5 yards per game.
      - question: |
          Which team did Ray Davis play for in 2023, and what were his overall statistics?
        answer: |
          Ray Davis played for Kentucky. He played 12 games, made 199 rushing attempts, gained 1,129 yards, averaged 5.7 yards per attempt, scored 21 touchdowns, had a longest rush of 378 yards, and averaged 94.1 yards per game.
```

-----

Page 5

### Document Outline Example: Datasheet (Good design)

- Fixing existing document outline from:
```yaml
document_outline: SEC: Overall Statistics
```
to:
```yaml
document_outline: Southeastern Conference Football: Overall Statistics 2023
```

- Summarize the essence of the data in the outline. 
- Include key themes or metrics covered in the data. 
- Include any year information 
- Why: Adds additional context when student model learns this data 

-----

### Document Outline Example: Physics Textbook (bad design)

```yaml
created_by: abhi1092
version: 3
domain: University Physics
document_outline: University Physics Volume 1
seed_examples:
  - context: |
      ## Gravitational Potential Energy beyond Earth

      We defined work and potential energy in Work and Kinetic Energy and Potential Energy and Conservation of Energy. The
      usefulness of those definitions is the ease with which we can solve many problems using conservation of energy.
      Potential energy is particularly useful for forces that change with position, as the gravitational force does over
      large distances. In Potential Energy and Conservation of Energy, we showed that the change in gravitational
      potential energy near Earth’s surface is. This works very well if gdoes not change significantly between and. We
      return to the definition of work and potential energy to derive an expression that is correct over larger distances.

      Recall that work (W) is the integral of the dot product between force and distance. Essentially, it is the product
      of the component of a force along a displacement times that displacement. We define as the negativeof the work done
      by the force we associate with the potential energy. For clarity, we derive an expression for moving a mass mfrom
      distance from the center of Earth to distance. However, the result can easily be generalized to any two objects
      changing their separation from one value to another.

      Consider Figure 13.11, in which we take mfrom a distance from Earth’s center to a distance that is from the center.
      Gravity is a conservative force (its magnitude and direction are functions of location only), so we can take any
      path we wish, and the result for the calculation of work is the same. We take the path shown, as it greatly
      simplifies the integration. We first move radiallyoutward from distance to distance, and then move along the arc of
      a circle until we reach the final position. During the radial portion, is opposite to the direction we travel along,
      so Along the arc, is perpendicular to, so. No work is done as we move along the arc. Using the expression for the
      gravitational force and noting the values for along the two segments of our path, we have

      The work integral, which determines the change in potential energy, can be evaluated along the path shown in red.

      Note two important items with this definition. First, $\omega$. The potential energy is zero when the two masses are
      infinitely far apart. Only the difference in Uis important, so the choice of $\omega$ is merely one of convenience.
```

### Document Outline Example: Physics Textbook (Good design)

Fix this from:
```yaml
document_outline: University Physics Volume 1
```
to:
```yaml
document_outline: University Physics Volume 1: Mechanics, Waves, Thermodynamics, Electromagnetism, Optics, and Modern Physics
```

- Use the title of the textbook/article as the document outline. [cite: 23]
- Add specific details from the document to make the title more descriptive. [cite: 24]
- Why: A detailed title provides better context, especially when sections focus on specific topics. [cite: 25]

-----

Page 8

### Document Outline: Manual (bad design)

```yaml
created_by: abhi1092
version: 3
domain: Standard Procedure Mannual
document_outline: RBC (SOP)
seed_examples:
  - context: "-\                                                                           -\n\
    | All Transaction File (ATF)    | Viewing manual entries (GL vouchers, credit\        |\n\
    \ and debit memos, and Cheques) to do investigation                                   |\n\
    -\                                                                           -\n\
    \                               |\n Application Launch Pad      | Performing\        |\n\
    \ history inquiries in Unit Reference File Inquiry (URFIQ) – Unit Inquiry and\         |\n\
    \ Demand Deposit Account (DDA) – Deposit Services to obtain details about transactions\ |\n\
    \ processed to personal and business client’s bank accounts. |\n Chart of Account\    |\n\
    \                               | Retrieving the general information of the GL numbers. |\n\
    -\                                                                           -\n\
    \                               |\n Excel                       | Processing of $50\   |\n\
    \ and under write-off items through the PeopleSoft bulk upload process. Also\          |\n\
    \ used for manual reconciliation of the Electronic Bill Payments GL.                  |\n\
    -\
```

### Document Outline: Manual (Good design)

Fix this from:
```yaml
document_outline: RBC (SOP)
```
to:
```yaml
document_outline: Royal Bank of Canada Standard Openrating Procedure and Investigations Guide for General Ledger and Transit Accounts
```
- Use the manual title as the document outline.
  Expand abbreviations to ensure clarity.

- Why: Prevents ambiguity, especially when dealing with technical or operational content.

-----

### Document Outline: Yearly Documents 

Fix this from:
```yaml
document_outline: IBM Annual Report
```
to:
```yaml
document_outline: IBM Annual Report 2023: Financial Insights and strategic overview
```

- Include the publication year or edition in the outline.

- Why: Helps differentiate versions and provides temporal relevance.

----

## Best Practices for Crafting Question-Answer (QA) Pairs

  * Ensure Numerical accuracy and necessary reasoning Steps

  * Ensure Questions are Self-Contained and grounded in context

  * Ensure completeness of answer

  * Ensure detailed and well written answers

  * Ensure alignment to downstream task

  * When dealing with tables add QA that cover each row

---

## Ensure Questions are Self-Contained and grounded in context

Example: 
Here is Yaml for generating knowledge data on football statistics. 
In below YAML you will see a context and 3-QA based on that. 
```yaml
seed_examples:
- context: |
    ## Leaders / Offense
    ### Rushing

    | Rank | Player            | Team      | G  | ATT | YDS  | AVG | TD | LONG | AVG/G |
    |------|-------------------|-----------|----|-----|------|-----|----|------|--------|
    | 1    | Schrader, Cody    | Missouri  | 13 | 276 | 1627 | 5.9 | 14 | 384  | 125.2  |
    | 2    | Daniels, Jayden   | LSU       | 12 | 135 | 1134 | 8.4 | 10 | 416  | 94.5   |
    | 3    | Davis, Ray        | Kentucky  | 12 | 199 | 1129 | 5.7 | 21 | 378  | 94.1   |
    | 4    | Judkins, Quinshon | Ole Miss  | 13 | 271 | 1158 | 4.3 | 17 | 269  | 89.1   |
    | 5    | Wright, Jaylen    | Tennessee | 12 | 137 | 1013 | 7.4 | 4  | 377  | 84.4   |
    | 6    | Hunter, Jarquez   | Auburn    | 12 | 159 | 909  | 5.7 | 7  | 311  | 75.8   |
    | 7    | Edwards, Daijun   | Georgia   | 12 | 165 | 880  | 5.3 | 13 | 245  | 73.3   |
    | 8    | McClellan, Jase   | Alabama   | 13 | 180 | 890  | 4.9 | 8  | 250  | 68.5   |
    | 9    | Etienne, Trevor   | Florida   | 11 | 131 | 753  | 5.7 | 9  | 267  | 68.5   |
    | 10   | Johnson Jr., Montrell | Florida | 12 | 152 | 817 | 5.4 | 6  | 253  | 68.1   |

  questions_and_answers:
  - question: |
      Which team did the top rusher Cody Schrader play for in 2023, **and what were his total recieves**?
    answer: |
      Cody Schrader played for Missouri. He played 13 games, **made 191 recieves**.

  - question: |
      Which team did Jayden Daniels play for in 2023, and **what were his total plays**?
    answer: |
      Jayden Daniels played for LSU. He played 12 games, **made 462 plays**.

  - question: |
      **Which team did Ray Davis play for in 2022, and what were his overall statistics?**
    answer: |
      **Ray Davis played for Vanderbilt. He played 12 games, made 232 rushing attempts, gained 1,042 yards, 
      averaged 4.5 yards per attempt, scored 8 touchdowns, had a longest rush of 280 yards, and averaged 86.8 yards per game**.
```
Notice how the highlighted portion of the Question and Answer are not grounded in provided document
Solution: Makesure each Question is completely grounded in the provided context i.e. the provided context has all the information necessary to answer the questions.
And makesure the provided answer is completely grounded in the provided context.

General Notes:
- Each question should include all necessary information.
- Use information directly from the provided context.
  **Why** : Avoids teacher model hallucination in generated QA, because teacher model will use these as example for generating more synthetic QA samples.

---

## When dealing with Tables make sure each QA covers each rows

Thanks, that's a helpful clarification. Here's a cleaner and more precise rewrite of your guideline that incorporates your intended structure:

---

When creating `questions_and_answers` for a document containing **tables**, follow this structured approach to maximize **coverage** and enable **robust augmentation**:

---

### ✅ 1. Row Coverage Through Multiple QA

* Each `context` block should include a **sample table** and 3 `questions_and_answers` entries.
* These 3 QA pairs should target **different rows** in the table (e.g., rows 1–3).
* The goal is to teach the model how to **iterate over rows**, so it can learn to generalize to all rows in the table.

---

### ✅ 2. Augmentation Through Multiple Templates

* To improve variety and comprehension, define **multiple `context` blocks**—each reusing the **same table** but applying a **different QA style or template**.
* For example:

  * **Context 1** → factual questions (e.g., team, attempts, yards)
  * **Context 2** → comparative questions (e.g., highest average, longest run)
  * **Context 3** → reasoning-based or aggregative questions (e.g., total touchdowns for top 3 players)
* Each context uses the same table, but the QA changes style and focus.

---

### 📌 Why This Pattern Works

* **Improves knowledge coverage**: By rotating through rows, every entry gets modeled.
* **Enhances comprehension**: Different templates force the model to learn multiple ways to extract or interpret the same data.
* **Encourages generalization**: With row-wise patterns and diverse QA forms, teacher models can generate broader, richer data.

---

### YAML Pattern

```yaml
- context: |
    ## Table Title
    ### Subsection
    | Rank | Player | Team | ATT | YDS | ... |
    |------|--------|------|-----|-----|-----|
    | 1    | ...    | ...  | ... | ... |     |
    | 2    | ...    | ...  | ... | ... |     |
    ...
  questions_and_answers:
    - question: ...
      answer: ...
    - question: ...
      answer: ...
    - question: ...
      answer: ...

- context: |
    ## Same Table
    ### Same Subsection
    | Rank | Player | Team | ATT | YDS | ... |
    |------|--------|------|-----|-----|-----|
    | 1    | ...    | ...  | ... | ... |     |
    ...
  questions_and_answers:
    - question: ...
      answer: ...
    - question: ...
      answer: ...
    - question: ...
      answer: ...
```


---

## Ensure Detailed and Well-Written Answers

**Guidelines:**

* Provide **detailed, well-structured question-answer pairs**
* **Avoid** short or single-word answers

**Why this matters:**

* Detailed answers guide the teacher model to produce comprehensive and informative responses
* Richer content improves factual coverage
* Higher-quality training data results in a more capable student model


### Example (Needs Improvement)

```yaml
questions_and_answers:
  - question: |
      What is the percentage change in IBM's operating (non-GAAP) earnings between 2023 and 2022?
    answer: |
      6.5%
  - question: |
      What was the diluted operating (non-GAAP) earnings per share for IBM in 2023, and how does it compare to 2022?
    answer: >
      $9.62 in 2023, compared to $9.13 in 2022.
  - question: |
      What major factor significantly impacted IBM's reported income in 2022?
    answer: |
      Income was significantly impacted by a one-time, non-cash pension settlement charge of $4.4 billion net of tax.
```

---

### Improved Version (Recommended Style)

```yaml
questions_and_answers:
  - question: |
      What is the percentage change in IBM's operating (non-GAAP) earnings between 2023 and 2022?
    answer: |
      IBM’s operating (non-GAAP) earnings increased by 6.5% in 2023 compared to 2022, indicating a year-over-year improvement in the company’s core operating performance.
  
  - question: |
      What was the diluted operating (non-GAAP) earnings per share for IBM in 2023, and how does it compare to 2022?
    answer: >
      In 2023, IBM reported diluted operating (non-GAAP) earnings per share of $9.62. This represents a 5.4% increase compared to $9.13 in 2022, reflecting stronger operational performance and effective cost management.

  - question: |
      What major factor significantly impacted IBM's reported income in 2022?
    answer: |
      IBM’s reported income in 2022 was significantly affected by a one-time, non-cash pension settlement charge of $4.4 billion (net of tax). This extraordinary expense reduced the company’s net income for the year and was unrelated to ongoing business operations.
```

---

## Ensure alignment to downstream task

* Align QA Style to Downstream Evaluation
* Match the QA style to the end-use scenario.
  **Why** : Keeps QA relevant and consistent with the task’s purpose.

---

Absolutely — here's a refined version of the guideline **with a concrete example** to illustrate the importance of numerical accuracy in synthetic data generation.

---

## Ensure Numerical Accuracy

**Guidelines:**

* Double-check all **cited data**, **numerical values**, **percentages**, and **calculations**
* Ensure **logical consistency** and **correct reasoning steps** for questions involving math or quantitative comparisons

**Why this matters:**

* Inaccurate numbers or flawed reasoning steps lead to hallucinated or misleading answers
* Synthetic data with incorrect math degrades model performance, especially on reasoning tasks
* Ensuring accuracy helps train a student model that’s trustworthy and factually consistent

---

### 🔍 Example with a Numerical Error (Needs Fixing)

```yaml
questions_and_answers:
  - question: |
      IBM's revenue increased from $60.5 billion in 2022 to $63.1 billion in 2023. What is the percentage increase in revenue year-over-year?
    answer: |
      The revenue increased by 5.1% from 2022 to 2023.
```

**Problem:** The percentage increase is incorrectly stated as 5.1%.
**Correct Calculation:**

\frac{63.1 - 60.5}{60.5} \times 100 = \frac{2.6}{60.5} \times 100 \approx 4.3%

### ✅ Corrected Example (Accurate Answer with Reasoning)

```yaml
questions_and_answers:
  - question: |
      IBM's revenue increased from $60.5 billion in 2022 to $63.1 billion in 2023. What is the percentage increase in revenue year-over-year?
    answer: |
      IBM’s revenue increased by approximately 4.3% year-over-year. This is calculated as:
      ((63.1 - 60.5) / 60.5) × 100 = (2.6 / 60.5) × 100 ≈ 4.3%.
```

---

## ✅ Ensure Answer is Complete Given the Context

### Guidelines

* ✅ Provide **Complete Answers**
* ✅ Include **Step-by-Step Reasoning** when a calculation, logic, or comparison is involved

### **Why This Matters**

* Ensures the **teacher model** outputs are accurate and interpretable
* Trains the **student model** to follow a logical chain of reasoning, improving its reasoning skill and factual reliability

### Example: Incomplete vs. Complete Answer

#### Incomplete Answer (Bad)

```yaml
- context: |
    In 2023, IBM reported operating (non-GAAP) earnings of $14.8 billion, up from $13.9 billion in 2022.
  
  questions_and_answers:
    - question: |
        What is the percentage change in IBM's operating (non-GAAP) earnings between 2023 and 2022?
      answer: |
        6.5%
```

**Problem:** This is a final value only, with no explanation or calculation. A student model trained on this won't learn how to reason step-by-step. This will also lead to overall low response quality of th student model.

#### Complete Answer (Good)

```yaml
- context: |
    In 2023, IBM reported operating (non-GAAP) earnings of $14.8 billion, up from $13.9 billion in 2022.
  questions_and_answers:
    - question: |
        What is the percentage change in IBM's operating (non-GAAP) earnings between 2023 and 2022?
      answer: |
        IBM's operating (non-GAAP) earnings increased from $13.9 billion in 2022 to $14.8 billion in 2023. 
        To calculate the percentage change:  
        ((14.8 - 13.9) / 13.9) * 100 = 6.5%.  
        Therefore, the operating earnings increased by 6.5% from 2022 to 2023.
```

**Benefits of the good answer:**

* Shows **numerical values** from the source context
* Includes the **formula** and computation
* Ends with a **conclusion sentence**

---

## How to avoid issues in generated data

* Ensure the tables in markdown are converted properly.
* Wrongly converted tables will results in teacher model inferring wrong information from them leading to inaccuracies in generated data.
* At the moment, InstructLab does not support code models. However, if your documents contain any code snippets, please enclose them in backticks using the format:
  `<type>.`

  ```python
  {code}  
  ```
* As mentioned in previous slides make sure reasoning steps are included in qna question-answers, and are correct

---


Chunk - 7

Knowledge 1.5 vs Knowledge 1.0 [cite: 1]
After processing the user’s document and creating seed data, the next step is to execute the Knowledge QA Generation Pipeline. [cite: 1]

1 [cite: 2]
Knowledge 1.0 (Draft): [cite: 2]
In this pipeline, the summary generation step is skipped, and Question-Answer pairs are directly generated from the raw document. [cite: 2]
Currently, this option is available in the open-source or upstream version [cite: 3]
Knowledge 1.5: [cite: 3]
This pipeline augments the user document by generating various types of summaries before creating Question-Answer (QA) pairs. [cite: 3]
The augmentation process improves comprehension and enriches the understanding of the document. [cite: 4]
This pipeline is exclusive to the downstream RhelAI product. [cite: 5]

Document Summarizing
For the knowledge 1.5 next step is to generate different types of summaries (document augmentation) from the seed data. [cite: 5]
This process produces three distinct types of summaries, designed to support question-and-answer generation. [cite: 6]
The summaries are as follows: [cite: 7]
Detailed Summary:Provides a comprehensive overview of the entire chunk, covering all its content in detail. [cite: 7]
Extractive Summary:Focuses on retaining sentences that are as close to the original text as possible. [cite: 8]
This type compresses the chunk while preserving its original meaning. [cite: 9]
Key Facts/Atomic Facts:Distills the chunk into a concise list of key facts, capturing the most essential information from the document. [cite: 10]
Each of the summary types in addition with original chunk are then used to generate a set of questions and answers. [cite: 11]
This approach allows student model to learn the chunk from multiple perspectives, enhancing their understanding of key concepts contained in it. [cite: 12]

2 [cite: 13]
Example of Summaries

3

Question Answer Generation [cite: 14]
Following the generation of chunk summaries we prompt teacher model to synthesize List of Question-Answer pairs such that: [cite: 14]
They follow same style as user provided QA from qna yaml [cite: 14]
They are grounded in provided context (document chunk, summarized chunk) [cite: 14]
In case of Knowledge 1.0 we simply skip to this step and generate list of Question-Answer pairs on raw user chunk. [cite: 14]

4 [cite: 15]
Question Answer Generation: Dataset Size [cite: 15]
The size of generated question-answer dataset is controlled by below factors: [cite: 15]
“max\_tokens” parameter specified in sdg pipeline config (This will be exposed to user in future updates of RhelAI) [cite: 15]
Number of context in qna.yaml: Each context is paired with every chunk in document to generate a unique list of question and answers [cite: 15]
Size of chunk: Smaller chunk might result in QA pairs that are under the max\_token\_limit [cite: 15]
Type of pipeline: knowledge 1.5 will generate 3x more data compared to knowledge 1.0 [cite: 15]
Filtering: How many of the generated samples get filtered out. [cite: 15]
High rate of filtering indicates either badly formed chunks or issue with teacher model inferencing (quantized model, issue in hosting etc) [cite: 16]

5 [cite: 16]
Example of Question Answer Generation
6 [cite: 16]

Question Answer Filtering [cite: 17]
Once the Question-Answer (QA) pairs are generated, a Large Language Model (LLM) is used to filter out pairs that do not meet quality criteria. [cite: 17]
The evaluation is based on the following standards: [cite: 18]
Relevance: [cite: 18]
Verifies that the answer directly addresses the question. [cite: 18]
Filters out cases where: [cite: 19]
The answer, while grounded in the context, does not effectively answer the question. [cite: 19]
The question is unrelated to the context, even if the answer itself is accurate within the document. [cite: 20]

7 [cite: 21]
Examples of Answer Relevance Filtering
Faithfulness Filtering: Below are examples of Questions that will be filtered out based on question quality filter [cite: 21]
Context: [cite: 21]

## Liquidity and Capital Resources We have generated strong cash flow from operations allowing us to invest and deploy capital to areas with the most attractive longterm opportunities. [cite: 22]

We provide for additional liquidity through several sources: maintaining an adequate cash balance, access to global funding sources, committed global credit facilities and other committed and uncommitted lines of credit worldwide. [cite: 22]
The following table provides a summary of the major sources of liquidity for the years ended December 31, 2021 through 2023. [cite: 23]
Q: What are the primary sources of renewable energy mentioned in the text?A: The text discusses solar and wind energy as key renewable sources. [cite: 23]
Violation: This answer is unrelated to the subject of liquidity and capital resources discussed in the text. [cite: 24]
Q: How does the text evaluate the environmental impact of its operations?A: The text mentions a comprehensive strategy to reduce carbon emissions by 50% over the next decade. [cite: 25]
Violation: The text does not address environmental impacts or sustainability measures, so this response is fabricated and irrelevant. [cite: 26]
Q: What is the main conclusion of the text about global credit facilities?A: The text concludes that global credit facilities are no longer viable in today’s economic climate due to rising interest rates. [cite: 27]
Violation: While global credit facilities are mentioned, the text does not make this conclusion. [cite: 28]
The response adds a false, unrelated focus to the discussion. [cite: 29]

8 [cite: 30]
Question Answer Filtering [cite: 30]
Once the Question-Answer (QA) pairs are generated, a Large Language Model (LLM) is used to filter out pairs that do not meet quality criteria. [cite: 30]
The evaluation is based on the following standards: [cite: 31]
Faithfulness: [cite: 31]
Ensures that the answer is derived directly from the given document chunk. [cite: 32]
Any answer that introduces additional information or infers details not supported by the chunk is filtered. [cite: 32]
This step eliminates hallucinated or inaccurate content. [cite: 33]
This step might not capture inaccurate math or inaccurate reasoning [cite: 33]

9 [cite: 33]
Examples of Faithfulness filtering
Faithfulness Filtering: Below are examples of QA pairs that will be filtered out based on faithfulness filter [cite: 33]
Context: [cite: 34]

## 24 Management Discussion International Business Machines Corporation and Subsidiary Companies business. [cite: 34]

Overall interest expense (excluding capitalized interest) in 2023 was $1,940 million, an increase of $379 million year to year primarily driven by higher average interest rates and a higher average debt balance in the current year. [cite: 34]

## Stock-Based Compensation Pre-tax stock-based compensation cost of $1,133 million increased $146 million compared to 2022. This was primarily due to increases from restricted stock units ($84 million), stock options ($32 million) and Employee Stock Purchase Plan (ESPP) ($21 million). [cite: 35]

The increases are driven by stock-based compensation awards granted as part of our annual cycles for executives and other employees, and the ESPP being considered compensatory effective April 1, 2022. Stock-based compensation cost, and the year-to-year change, was reflected in the following categories: Cost: $190 million, up $26 million; [cite: 36]
SG\&A expense: $616 million, up $50 million; and RD\&E expense: $328 million, up $70 million. [cite: 37]
Q: Why did IBM's interest expense increase in 2023? [cite: 38]
A: IBM's interest expense rose to $1,940 million in 2023, driven by higher average interest rates and debt levels. [cite: 38]
Additionally, the company might have faced challenges managing its financial strategy amid market volatility. [cite: 39]
Q: What factors contributed to IBM's rising stock-based compensation in 2023? [cite: 40]
A: IBM's stock-based compensation surged due to expanding employee benefit programs, new executive bonuses, and broader initiatives to align workforce incentives with company growth goals, possibly influenced by market competition and talent retention trends. [cite: 41]
Q: How much did IBM's pre-tax stock-based compensation cost increase in 2023 compared to 2022, and what were the primary contributors? [cite: 42]
A: IBM's pre-tax stock-based compensation cost increased by $146 million in 2023, reaching $1,133 million. [cite: 43]
The rise was primarily due to higher costs from restricted stock units ($84 million), stock options ($32 million), and the Employee Stock Purchase Plan ($21 million). [cite: 44]

10 [cite: 45]
Question Answer Filtering [cite: 45]
Once the Question-Answer (QA) pairs are generated, a Large Language Model (LLM) is used to filter out pairs that do not meet quality criteria. [cite: 45]
The evaluation is based on the following standards: [cite: 46]
Question Filtering: [cite: 46]
Ensure that the question is self-explanatory and does not need additional context (figure, table number, page number references) [cite: 46]
Subject-Aware Completeness: The question should be answerable from given context without need of any specialized external knowledge base or source. [cite: 47]

11 [cite: 47]
Examples of Question Filtering
Faithfulness Filtering: Below are examples of Questions that will be filtered out based on question quality filter [cite: 47]
Context: [cite: 48]

## Dynamics Our balance sheet at December 31, 2023 continues to provide us with flexibility to support and invest in the business. [cite: 48]

Cash and cash equivalents, restricted cash and marketable securities at December 31, 2023 were $13,462 million, an increase of $4,622 million compared to prior-year end. [cite: 48]
Total debt of $56,547 million increased $5,598 million from prior-year end primarily due to net debt issuances. [cite: 49]
We were opportunistic in accessing the debt market and issued $9,463 million of debt in the first quarter of 2023 to prudently plan for our debt maturity obligations in 2023 and 2024 as well as capital allocation priorities. [cite: 50]
We continue to manage our debt levels while being acquisitive and without sacrificing investments in our business. [cite: 51]
During 2023, we generated $13,931 million in cash from operating activities, an increase of $3,496 million compared to 2022. Our free cash flow for 2023 was $11,210 million, an increase of $1,919 million versus the prior year. [cite: 52]
Refer to page 31 for additional information on free cash flow. [cite: 53]
Our strong cash generation has enabled us to be acquisitive and increase our investment in R\&D [cite: 54]
Q: How does the balance sheet on page 31 further explain the concept of free cash flow as discussed in the text? [cite: 54]
Violation: Relies on specific external content (page 31) that is not included in the question, making it contextually dependent and unclear without that reference. [cite: 55]
Q: What specific details in the balance sheet explain the increase in restricted cash as of December 31, 2023? [cite: 56]
Violation: Assumes access to specific balance sheet details not provided in the question, making it incomprehensible without that external content. [cite: 57]
Q: What does the text imply about the company's ability to meet its debt maturity obligations in 2024, based on the issuance of $9,463 million of debt in the first quarter of 2023? [cite: 58]
Violation: Relies heavily on the reader’s ability to interpret implied content in the text without providing adequate detail to form a complete and independent understanding. [cite: 59]


Chunk - 8

Here is the faithful Markdown representation of the PowerPoint slide:

---

# SDG: Skill pipelines

```
qna.yaml  
gen contexts  
gen questions  
filter questions  
gen answers  
filter Q&As  
```

**Skills 1.0**
grounded
freeform
draft
classify
analyse
critique
plan
revise

**Skills 1.5**
Skill pipelines: Freeform v.s. Grounded

```
qna.yaml  
gen contexts  
gen questions  
filter questions  
gen answers  
filter Q&As  
```

grounded
freeform

---

## Skill 1.0 (Draft): Gen Contexts

```
context: |  
  a) Company X, with CEO Amy Williams, reported $30 billion in revenue and a $3 billion profit in 2021.  
  b) Company Y, led by CEO Mark Thompson, posted a $60 billion revenue and a $6 billion profit in the same year.  
  c) Company Z, under CEO Sarah Johnson, announced a $20 billion revenue and a $7 billion profit in 2021.  
  d) Company W, managed by CEO James Smith, revealed a $300 billion revenue with a $21 billion profit in 2021.  
  e) Company V, with CEO Lisa Brown, reported a $200 billion revenue and a $25 billion profit in 2021.  
  f) Company U, under CEO John White, posted a $180 billion revenue and a $20 billion profit in the same year.
```

```
qna.yaml  
gen contexts  
gen questions  
filter questions  
gen answers  
filter Q&As  
```

grounded
freeform

---

## Skill 1.0 (Draft): Gen Questions

```
question: |  
  Given the following data, identify the company with the highest profit in 2021 and provide its CEO's name:
```

```
qna.yaml  
gen contexts  
gen questions  
filter questions  
gen answers  
filter Q&As  
```

grounded
freeform

---

## Skill 1.0 (Draft): Gen Answers

```
answer: The company with the highest profit in 2021 is Company E, led by CEO Lisa Brown, as it reported a profit of $25 billion.
```

```
qna.yaml  
gen contexts  
gen questions  
filter questions  
gen answers  
filter Q&As  
```

grounded
freeform

---

## Skill 1.0 (Draft)

```
qna.yaml  
gen contexts  
gen questions  
filter questions  
gen answers  
filter Q&As  
```

**Skills 1.0**
grounded
freeform

draft
classify
analyse
critique
plan
revise

**Skills 1.5**

---

## Skill 1.5: Classify

```
classify: _B_ (Extraction)
```

draft
classify
analyse
critique
plan
revise

---

## Skill 1.5: Analyse

```
analysis: |  
  The task requires identifying the company with the highest profit in 2021 from a given list and then providing the name of its CEO.  
  This is a quantitative analysis task in the domain of finance and business management.  
  The user query demands sorting or comparing numerical values (profits) and retrieving related textual information (CEO's name).  
  The analysis is straightforward and revolves around extracting specific data points from the provided information.
```

draft
classify
analyse
critique
plan
revise

---

## Skill 1.5: Critique

```
critique: |  
  1. Highest Profit Identification: The assistant correctly identified the company with the highest profit, which is $25 billion.  
     However, the mislabeling of the company (referred to as Company E instead of Company V) affects the accuracy of the identification.  
  2. CEO Identification: The assistant correctly names Lisa Brown as the CEO of the company with the highest profit.  
  3. Data Accuracy: The profit figure of $25 billion mentioned in the response accurately matches the data provided for Company V.  
     This shows that the assistant has correctly extracted the relevant profit figure from the data.  
  4. Clarity and Precision: The response succinctly names the company and its CEO, which aligns well with the task requirements.  
     However, the incorrect reference to "Company E" instead of "Company V" detracts from the clarity and precision of the response.  
  5. Relevance: The response focuses directly on the query's requirement—identifying the company with the highest profit and its CEO's name.  
     There is no extraneous information included, which meets the criterion of relevance effectively.
```

draft
classify
analyse
critique
plan
revise

---

## Skill 1.5: Plan

```
plan: |  
  1. Correct Company Labeling: Amend the mislabeling of "Company E" to "Company V" to ensure accuracy in company identification,  
     satisfying the highest profit identification criterion. This correction directly addresses the first point of critique and is essential for the integrity of the response.  
  2. Verification of CEO's Name: Although the CEO's name, Lisa Brown, is already correct, re-confirm this detail in the response to ensure the CEO identification criterion is indisputably met.  
     This step also serves to double-check the data accuracy.  
  3. Reinforcement of Data Accuracy: Maintain the correctly reported profit figure of $25 billion.  
     Reinforce this by explicitly stating that this figure is the highest among all companies listed, as part of the response, thus emphasizing the data's accuracy and relevance.  
  4. Enhance Clarity and Precision: Redraft the response to include a clear mention of "Company V" along with the CEO's name, removing any ambiguity caused by the earlier error.  
     This step will not only correct the mistake but also enhance clarity and precision as required by the rubric.  
  5. Maintain Relevance: Continue to focus solely on the task requirements without adding unnecessary information.  
     The revised response should succinctly state which company had the highest profit and the name of its CEO, fulfilling the relevance criterion.
```

draft
classify
analyse
critique
plan
revise

---

## Skill 1.5: Revise

```
revised_response: The company with the highest profit in 2021 is Company V, led by CEO Lisa Brown, with a reported profit of $25 billion.
```

draft
classify
analyse
critique
plan
revise

---

chunk - 9

**Slide 1: LLM training 101**

Collect training signal (gradients) from datasets for those parameters
Define what parameters to update
Train the model (by minimizing loss)

Q1: There are many parameters in a LLM—what to update?
Q2: How do I arrange multiple datasets for my need?
\*Q3: How to update the model?

---

**Slide 2: Key training options**

Training = selection of parameters + schedule of datasets

What parameters to update?

* Full fine-tuning (FFT): all parameters
* LoRA: selected subset of parameters

How to use multiple datasets update?

* Stacked: all together and single training
* Phased: training with curriculum

Key differences v.s. Competitors

* FFT >> LoRA for learning knowledge
* Competitors (e.g. Azure) only supports LoRA

---

**Slide 3: Key training options: selection of parameters**

What parameters to update?

* Full fine-tuning (FFT): all parameters
* LoRA: selected subset of parameters

---

**Slide 4: Key training options: schedule of datasets**

How to use multiple datasets update?

* Stacked: all together and single training
* Phased: training with curriculum

Key differences v.s. Competitors

* Quality: phased learns the data with less forgetting
* Performance: enabling shipping starter model
* Competitors (e.g. Azure) only supports stacked

---

**Slide 5: Glossary of phased training**

phase00 : where the short foundation knowledge data is used

---

**Slide 6: Glossary of phased training**

phase05 : where the predefined synthetic knowledge data is used.

---

**Slide 7: Glossary of phased training**

Starter model: the model after  phase00  and  phase05

---

**Slide 8: Glossary of phased training**

phase07 : where the user knowledge data is used, optimized for RAG.

---

**Slide 9: Glossary of phased training**

phase10 : where the predefined+user skill data is used.

---

**Slide 10: Glossary of phased training**

Precomputed dataset: the predefined synthetic skill data

---

**Slide 11: Glossary of phased training**

Data mixing: The procedure of selecting the right proportion from various data sources to create a new dataset.



chunk - 10

Here is a faithful Markdown representation of the PowerPoint content from your uploaded slide deck:

---

## Part 3: Evaluation

Is my SDG, training or RAG app working well?
Evaluation can be done at different stages. The early the faster to iterate but the early metrics are more of a proxy of the final application performance.

SDG
Training
RAG

* Assessing trained model
* Common benchmarks
* Red teaming
* Is my model useful in real world?
* Setup
* Common evaluation metrics
* Checking synthetic data quality
* Quality check SDG outputs
* Specific metrics

## Evaluation

### Evaluation: SDG

**Evaluating SDG pipelines as models**
We can treat the entire SDG pipeline as a model and run common evaluation benchmarks on them.
For example, when developing the skill-1.5, we use MT-Bench as a development benchmark and run the SDG pipeline on it directly.
We were satisfied with skill-1.5 after improving it from 8.1 to 8.9 with Mixtral-8x7-0.1.

**Specific metrics**
Knowledge coverage: Coming soon
Currently coverage is ensured by filtration

**Q**
**SDG**
**A**

---

### Evaluation: Training

Common benchmarks are standard to measure different aspects of models

* **Holistic**: MT-Bench, AlpacaEval, AlpacaEval 2, arena-hard
* **Instruction following**: IFEval
* **Knowledge**: MMLU, MMLU-Pro, AGI-Eval
* **Math, Reasoning**: GSM, MATH, ARC-C, GPQA, BBH, MUSR
* **Commonsense**: WinoGrande, OBQA, SIQA, PIQA, Hellaswag, TruthfulQA
* **Coding**: HumanEval, MBPP
* **Reading Comprehension**: BoolQ, SQuAD 2.0
* **Customer-specific**: MT-Bench-Branch, MMLU-Branch

**Red teaming**
It’s always a good idea to have a group people playing with the model to see if they can break it somehow.

---

## RAG: AnythingLLM Setup

Download and install AnythingLLM on your machine. It supports all the platforms
Click on the spanner to begin setup

![AnythingLLM setup interface with a spanner icon highlighted](image-placeholder.jpg)

---

## RAG: AnythingLLM Setup

Click on **AI Providers** and under **LLM**, choose **AnythingLLM**
Then click on **Import GGUF file**
Import the file, give it a few minutes to spin up the model

![Interface showing AI Providers section and GGUF file import options](image-placeholder.jpg)

---

## RAG: AnythingLLM Setup

* AnythingLLM lets you choose your own embedding model
* It also lets you choose your own vector database
* It even offers **Milvus**
* Set all of these to your liking
* For zero setup - use the default AnythingLLM options
* There is also a text-to-speech option with **whisper**

![Setup interface with options for embedding model, vector DB, Milvus, and whisper](image-placeholder.jpg)

---

## RAG: AnythingLLM Setup

Create a new workspace
Then click on the gear icon for setup

![Workspace creation screen with gear icon shown](image-placeholder.jpg)

---

## RAG: AnythingLLM Setup

* Under chat settings, chose **AnythingLLM** and your **GGUF** model will load
* Set chat history to 1
* Change prompt to:

  > \[I am a Red Hat® Instruct Model, an AI language model developed by Red Hat and IBM Research based on the granite-3.0-8b-base model. My primary role is to serve as a chat assistant.]
* Change LLM temperature to zero
* Leave the vector dataset settings to default
* Later you can play with the Agent config, it has a RAG agent which is much better than the default RAG

---

## RAG: AnythingLLM Setup

Return back by clicking the gear
Then click on upload a document to start uploading your documents

![Interface for uploading documents after setup](image-placeholder.jpg)

---

## RAG: AnythingLLM Setup

* Use the interface to upload your documents from your computer
* Then select and move the uploaded documents to your workspace
* This will start the chunking and embedding process
* You are done with the setup!
* Feel free to check out data connectors in case that is more appropriate for your use case

---

## RAG: Evaluation

* Using your test set, query the model one by one
* Use a new thread for each of the questions if possible
* Record the answers
* **Benchmarking**
* Repeat the same with a proprietary model. GPT4 setup is super easy to do in AnythingLLM as long as you have an API key
* Record the results

---

## RAG: Evaluation

Format the evaluation record as shown on the below.
The **Answer** column contains the model responses (e.g. response from AnythingLLM from previous slides)

Use **LLMAJ** script to score the model responses.
The script also output reason for score.
It is important to review these scoring justification to make sure the judge model has parity with human judgement for given use-case

---



