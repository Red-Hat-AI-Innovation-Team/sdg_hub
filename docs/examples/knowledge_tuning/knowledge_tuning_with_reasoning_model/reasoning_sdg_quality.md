# Knowledge generation using a resoning model

- In this notebook, we will use the Nemotron Super 49B model to generate knowledge from a corpus of articles.
- Specifically, we will use the Quality Corpus for this purpose.
- The Nemotron Super 49B model will also be used not only for synthesizing training data but also for generation ICL examples for SDG.
- Since Nemotron Super is a thinking model, we will leverage it to generate synthetic "thinking data" for customizing a smaller thinking model - in this case, the Nemotron Nano model.

## Install sdg-hub

```bash 
pip install sdg-hub==0.1.0a2
```


## Installing Vllm

- Clone vllm repo
- Checkout PR #15008
- Build from source
    ```bash
    VLLM_USE_PRECOMPILED=1 pip install --editable .
    ```


```python
from datasets import load_dataset, concatenate_datasets
from openai import OpenAI

endpoint = f"http://localhost:8008/v1"
openai_api_key = "EMPTY"
openai_api_base = endpoint

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)
teacher_model = client.models.list().data[0].id
```

    /home/lab/.conda/envs/research_sdg/lib/python3.10/site-packages/tqdm/auto.py:21: TqdmWarning: IProgress not found. Please update jupyter and ipywidgets. See https://ipywidgets.readthedocs.io/en/stable/user_install.html
      from .autonotebook import tqdm as notebook_tqdm


### Construct ICL examples
- We will use nemotron super 49B model to generated some ICL examples
- We will use the generated ICLs to generate more examples from the quality corpus
- You can customize these prompts according to your downstream task
- Here we aim to create QA aimed at understanding not only specific details about articles but also to create a comprehensive understanding of the article


```python
def get_response_from_nemotron(msg):
    response = client.chat.completions.create(
        model=teacher_model,
        messages=msg,
        temperature=0.6,
        top_p=0.95,
        max_tokens=4096,
    )
    return response.choices[0].message.content
```


```python
ds = load_dataset("zitongyang/entigraph-quality-corpus", split='train').remove_columns(['entity', 'entigraph']).rename_columns({'raw': 'document', 'uid': 'document_outline'})
```


```python
prompt = f"Give me a short 3-4 line document and 3 detailed GRE-style thematic questions and answers based and grounded on the document."
prompt = f"Give me a short 8-9 line passage that has 2 paragraphs. Then, provide 3 detailed open-ended questions and their answers. Each question should require synthesizing or connecting ideas, entities, or concepts that appear in different parts of the passage"
prompt = f"Give me a short 8-9 line article. Then provide 3 Question and Answer based on the article such that the answers are long, comprehensive, detailed, and are grounded in the article."
msg = [{"role": "system", "content": f"detailed thinking on"}, {'role': 'user', 'content': prompt}]
generated_qa_pairs = get_response_from_nemotron(msg)
msg += [{"role": "assistant", "content": generated_qa_pairs}] + [{"role": "user", "content": "You refine the answers from previous response such that you distill your thinking into the answer"}]
edited_qa_pairs = get_response_from_nemotron(msg)
msg = msg + [{"role": "assistant", "content": edited_qa_pairs}] + [{"role": "user", "content": "Can you parse the generated question and answer pairs into a json format? If there are any steps include them in the json's answer's field"}]
json_verified_qa_pairs = get_response_from_nemotron(msg)
msg = msg + [{"role": "assistant", "content": json_verified_qa_pairs}] + [{"role": "user", "content": "Verify the generated question and answer pairs from previous response are correct, complete and grounded in the document. Fix all the issues in the answer if any."}]
verified_qa_pairs = get_response_from_nemotron(msg)
print(generated_qa_pairs)
print("-"*3)
print(edited_qa_pairs)
print("-"*3)
print(json_verified_qa_pairs)
print("-"*3)
print(verified_qa_pairs)
```

## Example of ICL samples generated chaining above prompts from Nemotron Super

### **Response 1** - Generate Initial Draft of Question and Answer Pairs
```text
<think>
Okay, the user wants a short 3-4 line document and three detailed GRE-style thematic questions with answers based on it. Let me start by crafting the document. It needs to be concise but rich enough for questions. Maybe a philosophical or historical topic since those lend themselves to thematic analysis.

I'll go with a quote from Nietzsche about morality. That's dense and can lead to questions on interpretation, implications, and comparisons. Now, the document should be 3-4 lines. Let me check the length. The Nietzsche passage I have in mind is four sentences. Perfect.

Next, the questions. GRE thematic questions often ask for interpretation, analysis of assumptions, and application to another context. First question: Identify the main thesis and explain. Second: Analyze the assumption underlying the claim. Third: Apply the idea to a different field, like science.

For the answers, they need to be detailed but not too long. Make sure each answer addresses all parts of the question. Check that the questions are indeed GRE-style—focusing on themes, structure, and application. Ensure the answers ground their points in the document. Avoid jargon, keep it clear. Let me review each part to ensure alignment with the user's request. Document length, question types, answer structure all seem to fit. Okay, that should work.
</think>

**Document (4 lines)**  
Nietzsche argued that traditional morality is a "slave morality" created by the weak to subjugate the strong, claiming, "The noble type of man experiences himself as the goal and measure of things; he does not require approval or justification from outsiders." In contrast, slave morality, rooted in resentment, posits equality and humility as virtues to constrain natural superiority. Nietzsche saw this as a perversion of life’s true will to power. Only the Übermensch could transcend such moral dichotomies.

---

**Thematic Questions & Answers**

### **1. Interpretation of Thesis**  
**Question:** Identify the central thesis of the passage and explain how the contrast between "noble" and "slave" morality supports Nietzsche’s broader philosophical claim.  
**Answer:**  
The thesis is that traditional morality is a subversive construct of the weak ("slave morality") to undermine natural hierarchy, contrasting with the self-affirming "noble morality" of the strong. The passage supports this by framing noble morality as intrinsic (e.g., "the goal and measure of things") and slave morality as reactive ("rooted in resentment"). This dichotomy underscores Nietzsche’s claim that conventional ethics are not objective but a political tool to dilute the "will to power" of exceptional individuals.

### **2. Assumption Analysis**  
**Question:** What unexamined assumption underlies Nietzsche’s critique of slave morality as a "perversion of life"? Analyze the implications if this assumption is challenged.  
**Answer:**  
The assumption is that "life’s true will to power" is a universal, instinctual drive that should not be morally constrained. If challenged (e.g., if cooperation, not domination, is life’s primary drive), Nietzsche’s critique collapses, as slave morality’s emphasis on equality could be seen as adaptive rather than perverse. This would invert his hierarchy, legitimizing collective ethics over individual supremacy.

### **3. Application to Another Context**  
**Question:** Apply Nietzsche’s framework to scientific progress. How might a "noble" vs. "slave" dynamic manifest in the pursuit of knowledge?  
**Answer:**  
A "noble" scientific ethos might prioritize unencumbered innovation (e.g., unfunded maverick researchers pursuing risky, visionary projects). Conversely, "slave morality" could manifest in bureaucratic, consensus-driven science (e.g., funding prioritizing collaborative, "socially responsible" projects over disruptive ideas). Nietzsche would argue the latter stifles genius, while defenders might see it as ensuring ethical accountability—a parallel to his moral critique.

```

---

### **Response 2** - Verify Question and Answer Pairs
```text
<think>
Okay, let me try to work through this. The user wants me to verify if the question and answer pairs generated earlier are correct, complete, and grounded in the document provided. First, I need to look at each question and answer carefully.

Starting with the first question: Interpretation of Thesis. The question asks to identify the central thesis and explain how the contrast between noble and slave morality supports Nietzsche’s claim. The answer given states the thesis correctly as traditional morality being a slave morality created by the weak. It also explains the contrast by saying noble morality is intrinsic and slave is reactive. The document does mention "noble type of man" experiencing himself as the goal and slave morality rooted in resentment. So the answer seems accurate here. The support from the document is there, so this pair is correct.

Next, the second question: Assumption Analysis. The question asks for an unexamined assumption in Nietzsche’s critique. The answer points out the assumption that the will to power is a universal drive that shouldn’t be constrained. The document mentions "life’s true will to power" as being perverted by slave morality. If this assumption is challenged, the critique would collapse. The implications analysis seems valid. The answer correctly identifies the assumption and discusses the consequences if it's challenged. This pair checks out too.

Third question: Application to Another Context. Applying Nietzsche’s framework to science. The answer differentiates between a noble ethos (unencumbered innovation) and slave (bureaucratic consensus). The document doesn’t mention science, but the framework of noble vs. slave morality is applied conceptually. Nietzsche’s ideas are being extended appropriately here. The answer stays true to his philosophy even though the context is new. So this is correct as well.

Now, checking completeness. Each question has a clear answer that addresses all parts. The answers reference the document specifically, like quoting "goal and measure of things" and "rooted in resentment." They don’t introduce information not in the document. The assumptions are correctly identified based on the text. 

Grounded in the document: All key points in the answers are backed by the document. For example, the Übermensch is mentioned in the document and used in the first answer to support transcending moral dichotomies. The will to power is directly from the document. 

Are there any inaccuracies? Let me re-read the document. The document says slave morality posits equality and humility to constrain natural superiority. The answers reflect that. The second answer’s assumption about will to power being universal is in the document as "life’s true will to power." The third question's application doesn't contradict the original ideas. 

I think all pairs are correct. The user might have been concerned if the answers added external info or misinterpreted the text, but they seem to align closely. Maybe the third question's answer could have a stronger link, but since it's an application, it's acceptable as long as the framework is applied correctly, which it is.
</think>

**Verification of Question & Answer Pairs**

All three question-answer pairs are **correct, complete, and grounded in the document**. Below is a detailed breakdown:

---

### **1. Interpretation of Thesis**  
- **Correctness**: ✅ The central thesis is accurately identified, and the contrast between "noble" and "slave" morality is explained using direct evidence from the document (e.g., "goal and measure of things," "rooted in resentment").  
- **Completeness**: ✅ Fully addresses both parts of the question (thesis + supporting contrast).  
- **Grounding**: ✅ Quotes and concepts (e.g., "intrinsic" vs. "reactive") align with the passage.  

---

### **2. Assumption Analysis**  
- **Correctness**: ✅ The assumption about "will to power" is correctly inferred from the phrase "perversion of life’s true will to power."  
- **Completeness**: ✅ Clearly states the assumption and analyzes implications if challenged.  
- **Grounding**: ✅ Directly tied to the document’s language about "life’s true will to power" and its contrast with slave morality.  

---

### **3. Application to Another Context**  
- **Correctness**: ✅ Faithfully applies Nietzsche’s framework to science without distorting his ideas.  
- **Completeness**: ✅ Distinguishes "noble" (unfettered innovation) and "slave" (bureaucratic) dynamics, engaging with potential counterarguments.  
- **Grounding**: ✅ While the *context* (science) is new, the moral dichotomy ("unencumbered pursuit" vs. "constrained collective ethics") is rooted in the document’s themes.  

---

### **Overall Strengths**  
1. **Document Alignment**: Answers cite specific phrases (e.g., "goal and measure of things," "rooted in resentment") to support claims.  
2. **Conceptual Accuracy**: No misinterpretation of Nietzsche’s terms (e.g., "slave morality" as reactive, "Übermensch" as transcendent).  
3. **Depth of Analysis**: Questions 2 and 3 require critical thinking (assumption-testing, application) beyond mere recall.  

### **Minor Suggestions for Enhancement (Optional)**  
- For **Question 3**, adding a brief phrase from the document to the answer (e.g., "mirroring how slave morality ‘constrains natural superiority’") could strengthen explicit grounding.  
- For **Question 2**, noting that the assumption also relies on Nietzsche’s value judgment that "domination" is more "natural" than cooperation could enrich the analysis.  

---

**Final Conclusion**: The pairs are well-designed, rigorous, and tightly anchored to the document. They effectively test interpretation, critical analysis, and thematic application—all hallmarks of GRE-style questions.  

Let me know if you’d like revisions or additional pairs!
```

---

### **Response 3** - Parse Question and Answer Pairs into JSON Format
```text
<think>
Okay, the user wants the question and answer pairs parsed into JSON format. Let me start by recalling the previous interaction. They had a document and three GRE-style questions with answers. Then they asked for verification, which I did, and now they want JSON.

First, I need to structure each question and answer properly. JSON requires key-value pairs. Each question should have a "Question" key, "Answer", and maybe "Type" to categorize them. The original questions were labeled as 1, 2, 3, so I can use those as IDs.

Looking at the first question: Interpretation of Thesis. The answer has bullet points for Correctness, Completeness, Grounding in the verification, but the actual answer is the paragraph before that. Wait, no. Wait, in the previous response, the answer for Question 1 was the paragraph starting with "The thesis is...", and then the verification had correctness etc. But the user now wants the original Q&A pairs in JSON, not the verification details.

So I need to go back to the initial Q&A provided. Let me check:

Original Q1:
**1. Interpretation of Thesis**  
**Question:** Identify the central thesis...  
**Answer:** The thesis is... 

Yes. So each entry in JSON should have the question text and the answer text. The verification part was separate, but the user didn't ask for that in JSON, just the pairs.

So for each of the three questions, structure as:

{
  "id": 1,
  "type": "Interpretation of Thesis",
  "question": "Identify the central thesis...",
  "answer": "The thesis is..."
},

And similarly for the others. Also, include the document in JSON if needed, but the user specified "question and answer pairs", so maybe just the Q&A.

Wait, the user said "parse the generated question and answer pairs into a json format". So the document is separate, but the pairs are the three questions with their answers.

So the JSON should be an array of objects, each containing id, type, question, answer.

Checking the original answer:

The three questions are:

1. Interpretation of Thesis
2. Assumption Analysis
3. Application to Another Context

Each has a Question and Answer.

So the JSON would look like:

{
  "questions": [
    {
      "id": 1,
      "type": "Interpretation of Thesis",
      "question": "Identify the central thesis of the passage and explain how the contrast between 'noble' and 'slave' morality supports Nietzsche’s broader philosophical claim.",
      "answer": "The thesis is that traditional morality is a subversive construct of the weak... This dichotomy underscores Nietzsche’s claim..."
    },
    {
      "id": 2,
      "type": "Assumption Analysis",
      "question": "What unexamined assumption underlies Nietzsche’s critique of slave morality as a 'perversion of life'? Analyze the implications if this assumption is challenged.",
      "answer": "The assumption is that 'life’s true will to power' is a universal, instinctual drive... This would invert his hierarchy..."
    },
    {
      "id": 3,
      "type": "Application to Another Context",
      "question": "Apply Nietzsche’s framework to scientific progress. How might a 'noble' vs. 'slave' dynamic manifest in the pursuit of knowledge?",
      "answer": "A 'noble' scientific ethos might prioritize unencumbered innovation... while defenders might see it as ensuring ethical accountability—a parallel to his moral critique."
    }
  ]
}

I need to make sure the quotes are properly escaped in JSON. Also, check if the user wants the document included. The instruction says "question and answer pairs", so probably not. But maybe confirm. The original document was separate, so likely just the Q&A.

Yes, the user's request was specifically about the generated pairs, so the above structure should be correct.

Now, formatting the JSON correctly with escaped quotes and proper syntax. Also, ensuring that each key is properly named. The types are the question types as labeled. The IDs are 1,2,3. Questions and answers are copied exactly from the original.

Double-checking for any markdown formatting in the answers. The original answers had line breaks but in JSON, line breaks can be preserved with \n or just as continuous text. Since the user didn't specify, keeping as continuous is safer.

So the final JSON should have the questions array with three entries as above.
</think>

Here’s the parsed JSON format for the question-answer pairs, including the document for context:

```json
{
  "document": [
    "Nietzsche argued that traditional morality is a \"slave morality\" created by the weak to subjugate the strong, claiming, \"The noble type of man experiences himself as the goal and measure of things; he does not require approval or justification from outsiders.\"",
    "In contrast, slave morality, rooted in resentment, posits equality and humility as virtues to constrain natural superiority.",
    "Nietzsche saw this as a perversion of life’s true will to power.",
    "Only the Übermensch could transcend such moral dichotomies."
  ],
  "questions": [
    {
      "id": 1,
      "type": "Interpretation of Thesis",
      "question": "Identify the central thesis of the passage and explain how the contrast between \"noble\" and \"slave\" morality supports Nietzsche’s broader philosophical claim.",
      "answer": "The thesis is that traditional morality is a subversive construct of the weak (\"slave morality\") to undermine natural hierarchy, contrasting with the self-affirming \"noble morality\" of the strong. The passage supports this by framing noble morality as intrinsic (e.g., \"the goal and measure of things\") and slave morality as reactive (\"rooted in resentment\"). This dichotomy underscores Nietzsche’s claim that conventional ethics are not objective but a political tool to dilute the \"will to power\" of exceptional individuals."
    },
    {
      "id": 2,
      "type": "Assumption Analysis",
      "question": "What unexamined assumption underlies Nietzsche’s critique of slave morality as a \"perversion of life\"? Analyze the implications if this assumption is challenged.",
      "answer": "The assumption is that \"life’s true will to power\" is a universal, instinctual drive that should not be morally constrained. If challenged (e.g., if cooperation, not domination, is life’s primary drive), Nietzsche’s critique collapses, as slave morality’s emphasis on equality could be seen as adaptive rather than perverse. This would invert his hierarchy, legitimizing collective ethics over individual supremacy."
    },
    {
      "id": 3,
      "type": "Application to Another Context",
      "question": "Apply Nietzsche’s framework to scientific progress. How might a \"noble\" vs. \"slave\" dynamic manifest in the pursuit of knowledge?",
      "answer": "A \"noble\" scientific ethos might prioritize unencumbered innovation (e.g., unfunded maverick researchers pursuing risky, visionary projects). Conversely, \"slave morality\" could manifest in bureaucratic, consensus-driven science (e.g., funding prioritizing collaborative, \"socially responsible\" projects over disruptive ideas). Nietzsche would argue the latter stifles genius, while defenders might see it as ensuring ethical accountability—a parallel to his moral critique."
    }
  ]
}

### **Key Features of the JSON Structure**
1. **Document**: Included as an array of sentences for clarity.
2. **Questions**: Array with `id`, `type`, `question`, and `answer` fields.
3. **Formatting**: Quotes are properly escaped (`\"`), and line breaks are omitted for compactness (add `\n` if line breaks are preferred).

### **Example Use Case (GRE Study Tool)**
```json
// Extracting a single question for display
{
  "question": questions[0].question,
  "answer":   questions[0].answer
}

Let me know if you’d like further refinements (e.g., adding metadata like "difficulty" or "keywords")!
```



```python
from datasets import load_dataset
icls = [{
  "icl_document": (
    "Nietzsche argued that traditional morality is a \"slave morality\" created by the weak to subjugate the strong, claiming, \"The noble type of man experiences himself as the goal and measure of things; he does not require approval or justification from outsiders.\"",
    "In contrast, slave morality, rooted in resentment, posits equality and humility as virtues to constrain natural superiority.",
    "Nietzsche saw this as a perversion of life’s true will to power.",
    "Only the Übermensch could transcend such moral dichotomies."
  ),
  "icl_query_1": "Identify the central thesis of the passage and explain how the contrast between \"noble\" and \"slave\" morality supports Nietzsche’s broader philosophical claim.",
  "icl_response_1": "The thesis is that traditional morality is a subversive construct of the weak (\"slave morality\") to undermine natural hierarchy, contrasting with the self-affirming \"noble morality\" of the strong. The passage supports this by framing noble morality as intrinsic (e.g., \"the goal and measure of things\") and slave morality as reactive (\"rooted in resentment\"). This dichotomy underscores Nietzsche’s claim that conventional ethics are not objective but a political tool to dilute the \"will to power\" of exceptional individuals.",
  
  "icl_query_2": "What unexamined assumption underlies Nietzsche’s critique of slave morality as a \"perversion of life\"? Analyze the implications if this assumption is challenged.",
  "icl_response_2": "The assumption is that \"life’s true will to power\" is a universal, instinctual drive that should not be morally constrained. If challenged (e.g., if cooperation, not domination, is life’s primary drive), Nietzsche’s critique collapses, as slave morality’s emphasis on equality could be seen as adaptive rather than perverse. This would invert his hierarchy, legitimizing collective ethics over individual supremacy.",

  "icl_query_3": "Apply Nietzsche’s framework to scientific progress. How might a \"noble\" vs. \"slave\" dynamic manifest in the pursuit of knowledge?",
  "icl_response_3": "A \"noble\" scientific ethos might prioritize unencumbered innovation (e.g., unfunded maverick researchers pursuing risky, visionary projects). Conversely, \"slave morality\" could manifest in bureaucratic, consensus-driven science (e.g., funding prioritizing collaborative, \"socially responsible\" projects over disruptive ideas). Nietzsche would argue the latter stifles genius, while defenders might see it as ensuring ethical accountability—a parallel to his moral critique."
},
{
    "icl_document": (
      "The coastal town of Willow Creek, once renowned for its pristine beaches, now struggles with rampant pollution. Plastic debris and oil spills have devastated marine life, prompting a decline in tourism and fishing industries. Residents have organized weekly clean-up initiatives, but the scale of the problem overwhelms their efforts.",
      "Technologists at the local university have developed an AI-powered buoy system to combat this. The buoys, equipped with solar panels and filtration technology, can identify and absorb oil spills while collecting microplastics. Data from the buoys is shared publicly, raising awareness and pressuring corporations to adopt sustainable practices. Though costly, the project has sparked hope for revitalizing the ecosystem and economy."
    ),
    "icl_query_1": "How does the technological solution address the economic *and* environmental challenges highlighted in the document?",
    "icl_response_1": "The buoys directly mitigate environmental harm by absorbing spills and collecting plastics, which could restore marine life. Economically, this restoration might revive tourism and fishing, reversing their decline. Public data-sharing also pressures corporations, potentially reducing pollution and creating a sustainable economic cycle aligned with environmental health.",
    
    "icl_query_2": "What implicit values or priorities do the community’s actions (clean-up initiatives) and the technologists’ project reflect, and how do these align or contrast?",
    "icl_response_2": "Both reflect a prioritization of environmental stewardship and community engagement. The clean-ups emphasize collective responsibility, while the buoy project highlights innovation and systemic accountability (targeting corporations). They align in seeking solutions but contrast in scale and approach—manual vs. technological, local vs. systemic.",
    
    "icl_query_3": "Imagine the buoy project succeeds. What unintended consequences might arise from its impact, considering document's themes?",
    "icl_response_3": "Success could lead to reduced community participation in clean-up efforts if residents perceive the technology as a sufficient solution. Economically, revitalization might exacerbate existing resource strains (e.g., overwhelming local infrastructure) or create dependency on the buoy system, undermining long-term sustainability if the technology falters."

}
]

# Load the quality corpus
ds = load_dataset("zitongyang/entigraph-quality-corpus", split='train').remove_columns(['entity', 'entigraph']).rename_columns({'raw': 'document', 'uid': 'document_outline'})

# Add ICLs to the dataset
ds = [ds.map(lambda x: icl) for icl in icls]
ds = concatenate_datasets(ds)

# Add additional columns needed for the knowledge generation pipeline
ds = ds.add_column('domain', ['Article']*len(ds))

# We now have the seed data ready. Let's save it
ds.to_json("seed_data.jsonl", orient='records', lines=True)
```

### Once we have the right set of ICLs, we now start generating the data
- For running nemotron with reasoning we need to add few new custom blocks to the pipeline
    - Block for post-processing the thinking output
    
    Note: Since merging of [#87](https://github.com/Red-Hat-AI-Innovation-Team/sdg_hub/pull/87) you can also simply use the `start` and `end` tag in prompt
    - Block for parsing based on regular expression
- We we also register nemotron thinking prompt with `detailed thinking on`



```python
from sdg_hub.flow import Flow
from sdg_hub.pipeline import Pipeline
from sdg_hub.sdg import SDG
from sdg_hub.prompts import PromptRegistry
from sdg_hub.blocks import BlockRegistry, Block
from transformers import AutoTokenizer
import re
from typing import List
from datasets import load_dataset, Dataset

tokenizer = AutoTokenizer.from_pretrained("nvidia/Llama-3_3-Nemotron-Super-49B-v1")

@PromptRegistry.register("nvidia/Llama-3_3-Nemotron-Super-49B-v1")
def nemotron_chat_template():
    return """{{- bos_token }}
{{- "<|start_header_id|>system<|end_header_id|>\n\n" }}detailed thinking on{{- "<|eot_id|>" }}
{%- for message in messages %}
  {%- if message['role'] == 'assistant' and '</think>' in message['content'] %}
    {%- set content = message['content'].split('</think>')[-1].lstrip() %}
  {%- else %}
    {%- set content = message['content'] %}
  {%- endif %}
  {{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' + content | trim + '<|eot_id|>' }}
{%- endfor %}
{%- if add_generation_prompt %}
  {{- '<|start_header_id|>assistant<|end_header_id|>\n\n' }}
{%- endif %}"""

ds = load_dataset('json', data_files="seed_data.jsonl", split="train")
@BlockRegistry.register("PostProcessThinkingBlock")
class PostProcessThinkingBlock(Block):
    def __init__(self, block_name: str, column_name: str) -> None:
        super().__init__(block_name=block_name)  
        self.column_name = column_name
    
    
    def generate(self, samples: Dataset):
        def post_process_thinking(x):
            if '</think>' in x[self.column_name]:
                x[self.column_name] = x[self.column_name].split('</think>')[-1].lstrip()
            return x
        samples = samples.map(post_process_thinking)
        return samples

@BlockRegistry.register("RegexParserBlock")
class RegexParserBlock(Block):
    def __init__(self, block_name: str, column_name: str, parsing_pattern: str="", parser_cleanup_tags: List[str]=[], output_cols: List[str]=[]) -> None:
        super().__init__(block_name=block_name)
        self.column_name = column_name
        self.parsing_pattern = parsing_pattern
        self.parser_cleanup_tags = parser_cleanup_tags
        self.output_cols = output_cols

    def generate(self, samples: Dataset):
        
        if self.parsing_pattern:
            new_data = []
            for sample in samples:
                parsed_outputs = self._parse(sample[self.column_name])
                
                max_length = max(len(value) for value in parsed_outputs.values())
                for values in zip(*(lst[:max_length] for lst in parsed_outputs.values())):
                    new_data.append({**sample, **dict(zip(parsed_outputs.keys(), values))})
            samples = Dataset.from_list(new_data)
        if self.parser_cleanup_tags:
            for clean_tag in self.parser_cleanup_tags:
               samples = samples.map(lambda x: {column_name: x[column_name].replace(clean_tag, "") for column_name in self.output_cols})
        return samples

    def _parse(self, generated_string):      
        pattern = re.compile(self.parsing_pattern, re.DOTALL)
        all_matches = pattern.findall(generated_string)
        matches = {column_name: [] for column_name in self.output_cols}
        if all_matches and isinstance(all_matches[0], tuple):
            for match in all_matches:
                for column_name, value in zip(self.output_cols, match):
                    value = value.strip()
                    # for clean_tag in self.parser_cleanup_tags:
                    #     value = value.replace(clean_tag, "")
                    matches[column_name].append(value)
        else:
            matches[self.output_cols[0]] = (
                [match.strip() for match in all_matches] if all_matches else []
            )
        return matches

knowledge_agentic_pipeline = "flows/synth_knowledge_reasoning_nemotron_super_49b_summary_diversity.yaml"
flow_cfg = Flow(client).get_flow_from_file(knowledge_agentic_pipeline)
sdg = SDG(
    [Pipeline(flow_cfg)],
    num_workers=1,
    batch_size=1,
    save_freq=1000,
)
ds_subset = ds.select(range(1))

# Generate data
generated_data = sdg.generate(ds_subset, checkpoint_dir="Tmp")
```


```python
print("### Question")
print(generated_data['question'][2])
print("### Response")
print(generated_data['response'][2])
```

## Output from reasoning SDGpipeline

### Question
Apply the "necessity vs. cosmetic" dynamic from the article to the field of plastic surgery. How might the strategies dentists use to market discretionary treatments parallel or diverge from those in cosmetic surgery, and what does this reveal about consumer behavior in health industries?  

### Response
```text
<think>
Okay, let's tackle this question. The user wants me to apply the "necessity vs. cosmetic" dynamic from the dentistry article to plastic surgery. I need to compare the marketing strategies between dentists and cosmetic surgeons, see where they're similar or different, and what that says about consumer behavior.

First, I'll recall the article. Dentists shifted from necessity (fillings) to cosmetics (whitening, etc.) using tools like intra-oral cameras, scare tactics, and upselling. They exploited the lack of insurance coverage, making patients pay out-of-pocket for discretionary treatments.

Now, plastic surgery. Cosmetic procedures are already discretionary, so maybe the strategies are similar. Both use technology (like simulations) to show results. Dentists use intra-oral cameras; surgeons might use 3D imaging to show potential outcomes. Both might use marketing based on societal beauty standards.

But there might be differences. Plastic surgery is more invasive, so maybe more emphasis on safety and long-term results. Dentists use scare tactics like mercury in fillings; surgeons might focus on health benefits (e.g., breast reduction for back pain) even if the primary motive is cosmetic.

Consumer behavior: Both industries thrive on discretionary spending driven by vanity and anxiety induced by marketing. However, plastic surgery might have higher barriers due to cost and risk, so marketing might need to balance desire with reassurance about safety. The article mentions that dentists blur necessity and indulgence; surgeons might frame cosmetic procedures as investments in well-being or confidence.

Wait, the question also asks what this reveals about consumer behavior. Probably that consumers prioritize appearance over health needs when given the option, and industries exploit this by creating perceived needs through technology and marketing. The lack of insurance coverage forces consumers to weigh discretionary spending, leading them to prioritize wants over needs if marketed effectively.

Need to structure the answer: parallel strategies (technology, societal ideals, upselling), divergent strategies (invasiveness, health angles), and consumer insights (vanity-driven, influenced by marketing, willing to spend on desires when enabled by financial models).
</think>


Applying the "necessity vs. cosmetic" dynamic to plastic surgery reveals both parallels and divergences in marketing strategies. **Parallels** include the use of technology (e.g., 3D simulation in surgery, akin to intra-oral cameras in dentistry) to amplify perceived flaws and sell discretionary treatments. Both fields leverage societal beauty standards (e.g., "Great American Smile" in dentistry, ideal body types in surgery) and employ upselling tactics ("treatment acceptance" scripts in dentistry, bundling procedures in surgery). **Divergences** arise from surgery’s higher stakes: while dentists focus on low-risk, repeatable cosmetics, surgeons must balance vanity-driven marketing with reassurances about safety and long-term outcomes (e.g., framing breast augmentation as a "confidence investment").  

This dynamic reveals that consumer behavior in health industries is **vulnerability-driven** and **market-shaped**. Patients prioritize discretionary aesthetics over necessary care when industries successfully manufacture anxiety (e.g., "halimeters" for halitosis, or surgical "imperfection" analyses) and align treatments with cultural ideals. The absence of insurance coverage for cosmetics forces a calculus where consumers weigh desires against costs, often favoring indulgences when framed as transformative. Ultimately, both fields exploit a **decoupling of health needs from consumer demands**, proving that profitability in discretionary health markets hinges on blurring clinical necessity with emotional and social aspiration.  
```

## Output from non-reasoning SDG pipeline
### Question
Analyze the economic paradox underlying the dental industry’s shift: Why did patients prioritize discretionary cosmetic treatments over medically necessary care?
### Response
The paradox stems from dentistry’s reliance on out-of-pocket spending (60% of costs). Cosmetic procedures (e.g., bleaching) were marketed as immediate, desirable investments in appearance, while necessary care (e.g., gum surgery) was perceived as inconvenient or costly with delayed benefits. Dentists capitalized on this by aggressively selling "wants" to offset the resistance to "needs," ensuring profitability in a market where insurance coverage was limited.


```python
print("### Faithfulness Evaluation")
print(generated_data['explanation'][2])
print("### Judgment")
print(generated_data['judgment'][2])
```

### Faithfulness Evaluation
The question references a specific "article" without providing its content, violating Non-Referential Clarity as it relies on unprovided external context. While it requires specialized knowledge (Subject-Aware Completeness), the core framework ("necessity vs. cosmetic") is not defined within the question, making it dependent on external information.
### Judgment
YES

### Data Mixing for training

### Prepare Nemotron Replay Buffer


```python
from datasets import load_dataset, concatenate_datasets

nemotron_ds = load_dataset("nvidia/Llama-Nemotron-Post-Training-Dataset-v1", "SFT")
nemotron_ds = nemotron_ds.filter(lambda x: x['used_in_training'] == 'yes')
nemotron_ds = nemotron_ds.map(lambda x: {'question': x['input'][x['input'].find("user<|end_header_id|>") + len("user<|end_header_id|>") : x['input'].find("<|eot_id|><|start_header_id|>assistant")].strip()})
nemotron_ds = concatenate_datasets(nemotron_ds.values())
nemotron_ds = nemotron_ds.shuffle(seed=894375).select(range(200000))
nemotron_ds = nemotron_ds.add_column('unmask', [False]*nemotron_ds.num_rows)
nemotron_ds = nemotron_ds.map(lambda x: {'messages': [{'role': 'system', 'content': 'detailed thinking on'}, {'role': 'user', 'content': x['question']}, {'role': 'assistant', 'content': x['output']}]})
nemotron_ds = nemotron_ds.remove_columns(['input', 'output', 'category', 'license', 'reasoning', 'generator', 'used_in_training', 'question'])
nemotron_ds.to_json("nemotron_replay_buffer_data.jsonl", orient='records', lines=True)
```

### Create functions for data mixng with conversation templates


```python
from utils import create_training_mix
```

### Create quality training mix of: reasoning dataset, with non-reasoning dataset, nemotron replay buffer


```python
### For this tutorial, we will use the following document uids from the quality dataset:
DOC_UIDS = [
    ' Defining Decay Down by David Plotz',
    ' Fight Clubbed by David Plotz',
    ' I, Antichrist? by Jeffrey Goldberg',
    " It's Time To Keelhaul U-Haul! by Jeffrey Goldberg",
    " My Father's Estate by Ben Stein",
    '"Phone Me in Central Park" by McConnell, James V.',
    '...After a Few Words... by Garrett, Randall', 
    '...And It Comes Out Here by Del Rey, Lester',
    'A Coffin for Jacob by Ludwig, Edward W.',
    'A Fall of Glass by Lee, Stanley R.',
    'A Filbert Is a Nut by Raphael, Rick',
    'A Gift from Earth by Banister, Manly',
    'A Gleeb for Earth by Schafhauser, Charles',
    'A Good Year for the Roses? by David Edelstein',
    'A Pail of Air by Leiber, Fritz',
    'A Planet Named Joe by Hunter, Evan',
    "AI: what's the worst that could happen? by Harry Armstrong",
    'Accidental Death by Baily, Peter',
    'All Day September by Kuykendall, Roger',
    'Ambition by Bade, William L.',
    'And Then the Town Took Off by Wilson, Richard',
    'Atom Mystery [Young Atom Detective] by Coombs, Charles Ira',
    'Beach Scene by King, Marshall',
    'Big Ancestor by Wallace, F. L. (Floyd L.)',
    'Birds of a Feather by Silverberg, Robert',
    'Bodyguard by Gold, H. L. (Horace Leonard)'
]
```


```python
from datasets import load_dataset
 
# Load tokenizer for pre-training formatting
tokenizer = AutoTokenizer.from_pretrained("nvidia/Llama-3.1-Nemotron-Nano-8B-v1")

# Load non-reasoning dataset from nemotron super 49b
nemotron_non_reasoning_ds = load_dataset("json", data_dir="/new_data/knowledge_rh/quality/knowledge1.25_nemotron_super_49b/", split="train")
nemotron_non_reasoning_ds = nemotron_non_reasoning_ds.filter(lambda x: x['score'] == '2' and x['judgment'] == 'YES')
nemotron_non_reasoning_ds = nemotron_non_reasoning_ds.filter(lambda x: x['document_outline'] in DOC_UIDS)
print(nemotron_non_reasoning_ds)

# Load reasoning dataset from nemotron super 49b
nemotron_reasoning_ds = load_dataset("json", data_dir="/new_data/knowledge_rh/quality/synth_knowledge_reasoning_nemotron_super_49b/gen", split="train")
nemotron_reasoning_ds = nemotron_reasoning_ds.filter(lambda x: x['score'] == '2' and x['judgment'] == 'YES')
nemotron_reasoning_ds = nemotron_reasoning_ds.filter(lambda x: x['document_outline'] in DOC_UIDS)
print(nemotron_reasoning_ds)
# Load nemotron replay buffer
nemotron_ds_replay_buffer = load_dataset("json", data_files="/new_data/knowledge_rh/quality/training_mix/nemotron_replay_buffer_data.jsonl", split="train")


# Create non-reasoning training mix
nemotron_ds_training_mix = create_training_mix(nemotron_non_reasoning_ds, tokenizer, 'off').shuffle(seed=894375)

# Create reasoning training mix
nemotron_reasoning_ds = create_training_mix(nemotron_reasoning_ds, tokenizer, 'on')

# Concatenate reasoning and non-reasoning training mixes
quality_reasoning_ds = concatenate_datasets([nemotron_reasoning_ds, nemotron_ds_training_mix]).remove_columns(['metadata', 'id']) # .select(range(40000))
print(quality_reasoning_ds)

# Concatenate training mix with replay buffer
training_mix = concatenate_datasets([quality_reasoning_ds, nemotron_ds_replay_buffer.shuffle(seed=894375).select(range(len(quality_reasoning_ds)))])

print(training_mix)
training_mix.to_json("/new_data/knowledge_rh/quality/training_mix/quality_knowledge_1.25_nemotron_49b_first_24.jsonl", orient='records', lines=True)
```

### Train student model
- Setup the training by cloning `https://github.com/instructlab/training` and following the instructions in the README
- The create `train.py` using below code
    ```python
    import argparse
    from instructlab.training.config import TorchrunArgs,TrainingArgs,DistributedBackend,FSDPOptions
    from instructlab.training.main_ds import run_training
    import os
    def parse_args():
        parser = argparse.ArgumentParser(description='Training script with configurable paths')
        parser.add_argument('--data_path', type=str, required=True,
                        help='Path to the training data file')
        parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the model or model identifier')
        parser.add_argument('--chat_tmpl_path', type=str, required=True,
                        help='Path to the chat template file')
        parser.add_argument('--exp_dir', type=str, required=True,
                        help='Path to the experiment directory')
        parser.add_argument('--parent_exp_dir', type=str, required=True,
                        help='Path to the parent experiment directory')
        return parser.parse_args()

    def main():
        args = parse_args()
        
        torch_args = TorchrunArgs(
            nproc_per_node=8,
            nnodes=1,
            node_rank=0,
            rdzv_id=123,
            rdzv_endpoint="0.0.0.0:8888",
        )
        output_dir = os.path.join(args.parent_exp_dir, args.exp_dir)
        train_args = TrainingArgs(
            model_path=args.model_path,
            data_path=args.data_path,
            ckpt_output_dir=output_dir,
            data_output_dir="data/processed-data",
            max_seq_len=20000,
            max_batch_len=25000,
            num_epochs=5,
            effective_batch_size=256,
            learning_rate=5e-6,
            warmup_steps=25,
            save_samples=0,
            use_dolomite=False,
            checkpoint_at_epoch = True,
            accelerate_full_state_at_epoch = False,
            process_data=True,
            chat_tmpl_path=args.chat_tmpl_path,
            distributed_backend=DistributedBackend.FSDP,
            fsdp_options=FSDPOptions(cpu_offload_params=False),
        )

        run_training(torch_args=torch_args,train_args=train_args)

    if __name__ == "__main__":
        main()
    ```

- Now create bash script with run command

    ```shell
    python train.py \
    --data_path "quality_knowledge_1.25_nemotron_49b_first_24.jsonl" \
    --model_path "nvidia/Llama-3.1-Nemotron-Nano-8B-v1" \
    --chat_tmpl_path "<chat_template_path>" \
    --exp_dir "nano_customized_thinking_quality_model" \
    --parent_exp_dir "<parent_exp_dir>"
    ```

## For reference: scripts for running SDG, starting student model and training

### Start Teacher Model
```shell
export HUGGINGFACE_HUB_CACHE="/new_data/hf_cache"   
export HF_DATASETS_CACHE="/dev/shm/hf"
export HF_HOME="/new_data/hf_cache"
export HF_MODEL_CACHE="/new_data/hf_cache"

for i in $(seq 0 2 6); do
    port=$((8000 + i/2))
    CUDA_VISIBLE_DEVICES=$i,$((i+1)) python -m vllm.entrypoints.openai.api_server \
        --model nvidia/Llama-3_3-Nemotron-Super-49B-v1 \
        --dtype float16 \
        --tensor-parallel-size 2 \
        --port $port \
        --trust-remote-code > log_$((i/2+1)).log 2>&1 &
done
```

### Run SDG in parallel
```shell
# Get dataset size and save into variable
dataset_size=$(wc -l seed_data.jsonl | awk '{print $1}')
number_of_processes=4
port=8000
for i in {0..4}; do
    # Continue until i=4
    if [ $i -eq 4 ]; then
        break
    fi
    dataset_start_index=$((i * dataset_size / number_of_processes))
    dataset_end_index=$((dataset_start_index + dataset_size / number_of_processes))
    python generate.py --ds_path  seed_data.jsonl \
        --bs 2 --num_workers 10 \
        --save_path /new_data/knowledge_rh/quality/knowledge1.25_nemotron_super_49b/gen.jsonl \
        --flow flows/synth_knowledge1.5_nemotron_super_49b.yaml \
        --endpoint http://localhost:$port/v1 \
        --checkpoint_dir /new_data/knowledge_rh/quality/knowledge1.25_nemotron_super_49b/data_checkpoints \
        --save_freq 1000 \
        --dataset_start_index $dataset_start_index \
        --dataset_end_index $dataset_end_index > run_sdg_$i.log 2>&1 &
    echo "Starting process $i with dataset from $dataset_start_index to $dataset_end_index on port $port"
    port=$((port + 1))
done
```

### Run SDG on new reasoning pipeline
```shell
# Get dataset size and save into variable
dataset_size=$(wc -l seed_data.jsonl | awk '{print $1}')
number_of_processes=4
port=8000
for i in {0..4}; do
    # Continue until i=4
    if [ $i -eq 4 ]; then
        break
    fi
    dataset_start_index=$((i * dataset_size / number_of_processes))
    dataset_end_index=$((dataset_start_index + dataset_size / number_of_processes))
    python generate.py --ds_path  seed_data.jsonl \
        --bs 2 --num_workers 10 \
        --save_path /new_data/knowledge_rh/quality/synth_knowledge_reasoning_nemotron_super_49b/gen.jsonl \
        --flow synth_knowledge_reasoning_nemotron_super_49b.yaml \
        --endpoint http://localhost:$port/v1 \
        --checkpoint_dir /new_data/knowledge_rh/quality/synth_knowledge_reasoning_nemotron_super_49b/data_checkpoints \
        --save_freq 1000 \
        --dataset_start_index $dataset_start_index \
        --dataset_end_index $dataset_end_index > run_sdg_$i.log 2>&1 &
    echo "Starting process $i with dataset from $dataset_start_index to $dataset_end_index on port $port"
    port=$((port + 1))
done
```
