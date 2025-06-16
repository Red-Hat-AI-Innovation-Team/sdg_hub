# Knowledge Tuning / Customization of Reasoning Models Slides

## Slide 1 Recaping our sdg_hub **knowledge pipeline**:
Uses `LLMBlocks` and a `FilterBlock`

### Here’s a breakdown of the pipeline flow:

1. **LLMBlock: 3 Types of Summary**

   * This stage has 3 blocks that create three types of summaries from the input content (likely *extractive*, *detailed*, and *atomic facts*).
   * It serves as the starting point of the pipeline.

2. **LLMBlock: Generate Question-Answer based on qna yaml**

   * Based on the summaries and a `qna.yaml` schema, this block generates structured Q\&A pairs.
   * The qna.yaml has seed_examples that act has "template" for teacher model to generate more similar synthetic QA pairs.
   * The qna.yaml also contains the document outline needed as extra context for teacher model when looking at document chunks (like a title of the document)

3. **LLMBlock: Evaluate Faithfulness**

   * This evaluates the faithfulness (i.e., groundedness of the answer to the document chunk) of the generated Q\&A pairs against the original document or summary.
   * It ensures quality control before filtering.

4. **FilterBlock: Filter on Score**

   * This block filters out low-quality or unfaithful Q\&A pairs based on evaluation scores.
   * The result is a refined set of high-confidence Q\&A pairs.

### Summary:

This slide outlines a **4-stage pipeline** to convert documents into high-quality Q\&A data:

> *Summarization → QA Generation → Faithfulness Evaluation → Score-based Filtering.*

It’s a sdg_hub's knowledge pipeline (also known as knowledge 1.5) synthetic data generation flow for tuning or evaluating LLMs. Let me know if you'd like a cleaner version for your presentation or documentation.

## Slide 2-3 What we want to achieve

### When you do regular knowledge tuning:

- When the trained student model is asked below question:

```text
According to 'Defining Decay Down' by
David Plotz, how do dentists convince
patients to get expensive treatments?
```

It responds with:

```text
The article suggests that dentists use "fear, fraud and
finesse" to convince patients to get expensive
treatments. They use fear by showing patients the
potential harm of not getting the treatments and by
using technical jargon to make the treatments sound
more severe than they actually are. The article also
suggests that dentists use fraud by inflating the
severity of the condition and by using unnecessary
x-rays and exams to justify expensive treatments.
Finally, the article suggests that dentists use finesse
by building a relationship with patients and by using
marketing techniques to make the treatments seem
appealing.
```

- What we want to achieve is a reasoning student model trained on your document such that when I ask same question as above:

```text
According to 'Defining Decay Down' by
David Plotz, how do dentists convince
patients to get expensive treatments?
```

It responds with:

```text
<think>
Okay, let's tackle this question. The user is asking how dentists convince patients to
get expensive treatments according to "Defining Decay Down" by David Plotz. First,
I need to recall the document provided.
The document mentions that the article discusses how dentists use a technique
called "treatment acceptance" strategies to convince patients. The main methods
highlighted are: loss of insurance benefits, fear of bad teeth affecting appearance,
fear of health risks, and aggressive sales tactics.
So, the answer should outline these strategies. Let me check each point again to
ensure accuracy. The article states that when patients lose their insurance, dentists
offer packages with lower upfront costs, which might make the treatment more
affordable. Then, the appearance of bad teeth can lead people to feel pressured to
fix it, even if they don't have bad teeth. Health risks are another trigger, such as gum
disease leading to heart issues, which might make patients feel they need
immediate treatment. Lastly, aggressive tactics like seminars on bad teeth,
pamphlets, and scripted sales pitches are used.
I should structure the answer by listing each strategy clearly, making sure to
reference the document's examples. Also, note that the article criticizes these
methods as manipulative, so it's important to mention that. Avoid adding external
information, only stick to the provided text. Keep the answer concise but cover all
four points. Let me piece it together now.
</think>
```

```text
According to the article, dentists use several tactics to
convince patients of expensive treatments: (1) loss of
insurance benefits by offering discounted packages, (2)
fear of appearance (e.g., seminars highlighting how bad
teeth undermine self-esteem), (3) perceived health risks
(e.g., linking gum disease to heart disease), and (4)
aggressive sales tactics like scripted pitches and
pamphlets. These strategies exploit psychological
vulnerabilities rather than transparent medical necessity.
```


## Slide 4 What do we want to achieve?

- We would like to convert a set of documents into training data that is in messages format
- We want to use sdg_hub and a large reasoning teacher model like: phi-4-reasoning-plus, Nvidia Nemotron Super etc.

## Lets take step-by-step approach

- We will first try by changing our teacher model. The sdg_hub's typical teacher model for knowledge generation is mixtral.
- This model, when we ask question, directly responds with the answer. It does not give any thinking trace
- We change it to Nvidia Nemotron Super such that when I ask Question it first does thinking between <think>...</think> tag followed by the answer

(Transcipt: boston-abacus-3s336: This was the session few months ago where question in answer comes out because the model knows your content. what will happen after today's session question comes in model starts thinking and this will be actually quite fascinating if you see how the model is trying to get to the answer. it's actually kind of fun reading the thought process because lets you figure out how the model arrived at the answer. and sometimes it does look like that it is thinking and maybe it is who knows even after thinking it's thinking about how to structure the output. Exactly.
boston-abacus-3s336: so it kind of is setting the output to verbose and looking at models's process of arriving it.  So if there's any fault or any mistakes because the model is verbalizing its thought process typically it is also able to correct itself right why the reasoning model performance in practice is always better than the vanilla model is simply because of this additional block before answering the model tries to figure out how to get to the answer and then often you will see maybe not in this example
boston-abacus-3s336: But often you'll see things but wait let me think about this. It will write a graph and then say things like that. Especially if I use a mathematical example it tends to stop and rethink whatever it has done before criticize itself and then improve upon polish it or start from a scratch or go back a few steps. Right.  So if you were interested in that how reasoning model works why they are better in accuracy when it comes to tasks that require common sense reasoning reading the trace is very much advisable like you you'll see that why things are and maybe that's the right way of thinking about the think block it's basically a trace okay but that is the objective of today how to generate this data
boston-abacus-3s336: And then we'll probably be able to show you post training how the model changes in terms of performance. Exactly. As Akash was mentioning this is our objective. I have some documents. I'll use SDG hub and then generate the training data you see on the right where I have a question a sum thinking process and answer and I would like to train my model on this to start thinking on my data. I like this slide for one more reason.
boston-abacus-3s336: every model there you can use it if you like for this process you can u and I'm going to walk through how you will use one of these models as teacher and then he also show you his script that he uses for training u the u maybe one point to just say right out of the bat to generate data here which requires thinking block you need a model that can think generically so that's a piece to keep an eye for but otherwise all models are supported in Given the abundance of small models I think nowadays. Yeah. There's a lot of choices you can make. we are a big fan of the last one Microsoft 54. definitely worth a try. It's an amazing model. It's small but it just works so well.
boston-abacus-3s336: it's very much recommended. Yeah it fits on a single GPU versus most of the other models and it's a actually properly open source MIDI license. Yeah. Okay. so I want to start by showing you the change we will do step by step. So we'll make one change at a time. the first thing I want to do is I'll pull up my knowledge pipeline which is knowledge pipeline generate data and training. The input to this pipeline was mixed all teacher model. We use that to generate the data. I'm going to switch from mix to neotron and I'll show you the steps I did just to switch the teacher model. Hey maybe a quick question.
boston-abacus-3s336: Are people familiar with what the Neotron reasoning models So Neotron is a word that Nvidia has overused for basically everything they created. so just real quick when we refer to Neotron, we're referring to their reasoning models that they released during the GTC event a week before that.  And these are basically llama 3 series models that have been fine-tuned or a data set that they released reason. additionally and you will look at their sizes and their sizes don't match to llama sizes, right?

Wes Jackson: So when we have a reasoning model, is it simply a model that's been given a skill of reasoning or is it a fundamentally fresh architecture like from the ground up?
boston-abacus-3s336: Right. …
boston-abacus-3s336: it's just a model that has been given the skill of reasoning. it's just training process.
Wes Jackson: All right.
boston-abacus-3s336: No architectural change. And then I mean if you're interested I think that's the link that we sent. There's also a training hub. So to SDG is a training In training hub there is a library called async grpo…
Wes Jackson: Okay.
boston-abacus-3s336: if you're interested in how to make your own reasoning model from a scratch worth checking out. So that's our work that we are hopefully will release soon but that those are the set of post training techniques that you can apply. So you can take data from neotron choose your favorite model and use that those techniques to then create a reasoning model. you may not succeed.
boston-abacus-3s336: because it's L. But if you're interested, that's the way to go and create your own reasoning models.)


Slide 13
Summary. Instructions to register the NEMOTRON model via prompt templates. Extract and reuse the chat template from its tokenizer.

To add any new chat template you can use PromptRegistry:
```python
@PromptRegistry.register("nvidia/Llama-3_3-Nemotron-Super-49B-v1")
def nemotron_chat_template():
   return """<chat template goes here>""""
```

You can access any model's chat template throught the tokenizer by doing:
```python
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("nvidia/Llama-3-3-Nemotron-Super-49B-v1")
tokenizer.chat_template
```

Transcript. If I want to add nemotron as my teacher model, these are the steps I follow. The first step is to add the chat template. I do this by simply doing prompt registry and then register the nemotron model. Don't worry about the big chat template string I simply pulled it out of the tokenizer and I included the code I used to pull it out. Just two lines. once you do that, you can just paste it or you can just simply call it. Once you do this, you're good to go. This chat template can be used anytime you need it with the nemotron model. 

Example yaml flow:
```yaml
- block_type: DuplicateColumns
  block_config:
    block_name: duplicate_document_col
    columns_map:
      document: base_document

- block_type: LLMBlock
  block_config:
    block_name: gen_detailed_summary
    config_path: configs/knowledge/detailed_summary.yaml
    model_id: mistralai/Mixtral-8x7B-Instruct-v0.1
    output_cols:
      - summary_detailed
  gen_kwargs:
    max_tokens: 4096
...
```

Update the model-id to:
```yaml
- block_type: DuplicateColumns
  block_config:
    block_name: duplicate_document_col
    columns_map:
      document: base_document

- block_type: LLMBlock
  block_config:
    block_name: gen_detailed_summary
    config_path: configs/knowledge/detailed_summary.yaml
    model_id: nvidia/Llama-3_3-Nemotron-Super-49B-v1
    output_cols:
      - summary_detailed
  gen_kwargs:
    max_tokens: 4096
...
```

Slide 14
Summary. Update pipeline to use NEMOTRON by replacing model ID references. Ensures the pipeline now runs with the reasoning teacher.

Transcript. Once this is done, the next thing you want to do is take your flow and simply replace the model IDs with the new model that you created. So this model ID is the same one I used here.  You see in the register the model ID I have I need to make sure that the name matches. So simply add that model ID we had mistral  before and I simply go and replace that. And just for this, the pipeline should work with the nemotron model. I did that and I tested it.



Slide 15
Summary. When summarizing, the reasoning model adds unnecessary thinking blocks. A post-processing step is needed to remove them for clean summaries.

Output of Summary Block:
```text
<think>Okay, let's tackle this summary. The user wants an economic analysis focusing on ...</think>
Economic Analysis: Dentists’ Shift to Cosmetic...
```

Transcript. With that change when I run that this is what I get. If you remember the first part of our knowledge pipeline was to summarize a document but when I summarize it I have this thinking process and the summary.
This is great, but I would like for it to create question answers only on summary, not on the entire thing so I need to make sure I remove that thinking process when I'm summarizing the document because the super model will basically think on everything and it will always give you that think process. So I can make a small change. 




Slide 16
Summary. Recap of the required change: remove the <think> blocks immediately after summarization to ensure cleaner downstream QA generation.

Existing sdg_hub knowledge flow (also known as knowledge 1.5) illustration of the yaml flow:
LLMBlocks: 3 Types of Summary -> LLMBlock: Generate Question-Answer base on qna yaml -> LLMBlock: Evaluate Faithfulness -> FilterBlock: Filter on Score

Transcript. Just to recap this was our knowledge pipeline.  So the change I need to make is after the first block where I summarize it just so that I can get rid of this thinking process. 


Slide 17
Summary. Insert a new reusable block in the YAML that strips out <think> tags. Option to do this inside or outside the LLM block.

Python code for implementing new Block in sdg_hub
```python
from sdg_hub.blocks import BlockRegistry, Block
import re
from typing import List
from datasets import Dataset

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
```
Once implemented add the block can be used by flow yaml
```yaml
- block_type: PostProcessThinkingBlock
  block_config:
    block_name: post_process_thinking
    column_name: summary_detailed
```

Transcript. To do that I'm going to add a simple block called postprocess thinking out. It's a very simple block that simply looks inside a string for the occurrence of that tag and then removes it.
And then since I think two or three days ago, Shiv made a fix. So you can also achieve this inside llm block. There are two ways of achieving this. You can have your own separate block or you can simply use tags inside the LLM block. although I will say the preferred mode would be that you separate all atomic transports into separate blocks.  The good thing is then you can reuse it however you like it and you can decide where to put it or remove it. But once you do that and register that block then on the left is how I would use the postprocessing block in my YAML.When we go to the notebook I'll point you to the files where I added these.
But for now I've pasted the snippets that the changes you will be or you have to make to get from the previous to the next. 


Slide 18
Summary. Use the postprocessing block not just after summary but also before evaluation to ensure QA generation is based only on relevant content.

New sdg_hub knowledge flow for reasoning teacher models, illustration of the yaml flow:
LLMBlocks: 3 Types of Summary -> Post Process Thinking -> LLMBlock: Generate Question-Answer base on qna yaml -> LLMBlock: Evaluate Faithfulness -> Post Process Thinking  -> FilterBlock: Filter on Score

Transcript. So once I add this, this is kind of the edit I'll be making right. So I've injected the post-processing after summary. If you remember we also evaluate and I also don't want the thinking blocks left in the evaluation part.  So I just added the thinking post-processing block there too. because it's reusable, I can just put the exact block there. 


Slide 19

Summary. Validation: postprocessed summaries are clean, allowing better QA generation without interference from previous reasoning traces.

Generated Summary:
```text
## Economic Analysis: Dentists’ Shift to Cosmetic...
```
Transcript. And just with that, this is our output. So now we've gotten rid of the thinking tag and the summary looks good. It's clean to the point and we can use this to generate question and answers. 


Quick recap.

What is the changes? Number 1: Switch the teacher model. Number 2: introduce <think> post-processing blocks. We started with the knowledge pipeline you're very much aware of. You might not have seen that pipeline before just because it was behind the service but what we're really doing here is just modifying that. Again, step one: switch the teacher model and that is a very cosmetic change and it almost already works if you didn't even do the filter block right you can see the pattern emerging and then the very next thing you have to do is just filter out the think block while you're evaluating and while you're generating question and answers because you don't want to generate your question and answers on the thinking part you want to generate that on the summary itself.


Slide 20

Summary. Problem: given our current prompt and flow the reasoning model is asked to generates series of QA. The output of model is thinking block followed by list of QA. What we want is for each generated Question there should be thinking followed by the answer. Solution: separate question and answer generation into distinct blocks to control thinking injection.

Current output of reasoning teacher model:
```text
<think>
Thought process here ....
</think>
[Question]
Question 1 text here
[Answer]
Answer 1 text here
[Question]
Question 2 text here
[Answer]
Answer 2 text here
```

This will get parsed as
```yaml
[Question]
Question 1 text here
[Answer]
Answer 1 text here
[Question]
Question 2 text here
[Answer]
Answer 2 text here
```
What we want:
```text
[Question]
Question 1 text here
[Answer]
<think>
...
</think>
Answer 1 text here
[Question]
Question 2 text here
[Answer]
<think>
...
</think>
Answer 2 text here
```

Transcript. Once summarization is done, I need to generate question and answers. The way it happens is I give the teacher model a summary and then ask it to create a series of QA. So the output is something on what you see the block on the left. With the nemotron model, it will think and then give you a series of QA. Now the problem is that's not what we want to do. So when I postprocess the thinking out I end up with what is on the right which is QA 1, QA 2. This is not what we want. What we want is something like this where for each answer there is explicit thinking process. To do that what I need to do is separate QA block into two blocks. Generate questions and you can remove the thinking and then for each question you generate the answer. That's the process I'll follow which is break this apart into two blocks instead of one do question then answer. 
Just maybe again zooming out a little bit. One challenge that we really have when we're using a thinking teacher model is that any question you ask it, it will think. My knowledge flow requires me to take your context or document, generate some short summaries, then use those summaries to generate questions and then use those question to answer the questions using the context. That's the flow. What I want to make sure is I don't continuously generate these thinking blocks and append them because my target is a context, a question, a thinking block, and an answer. 


Slide 21
Summary. Clarification: Only retain the thinking that precedes the answer, discard all other reasoning traces during QA generation.

```text
[Question]
Question 1 text here
[Answer]
<think>
...
</think>
Answer 1 text here
[Question]
Question 2 text here
[Answer]
<think>
...
</think>
Answer 2 text here
```

Transcript. The only thinking block I want to keep is the one that the nemotron teacher model used before answering the set question. So it may sound complicated. All he's saying is guys this model will generate thinking block.  We will keep removing those until we reach to the thinking block pertaining to the answer. That one we will keep. 



Slide 22
Summary. Strategy: Use one prompt for generating thoughtful questions (without answers), and another for thoughtful answers (with retained thinking).

Prompt for generating Question:
```yaml
system: You are a very knowledgeable AI Assistant that will faithfully assist the user with their task.

introduction: Develop a series of educational questions from a chapter in a {{domain}} textbook. 

principles: |
  The questions should:
  * Self-contained – understandable without needing to reference tables, figures, or specific text sections.
  * Focus on the provided example and follow the format and style of the provided examples.
  * Relevant to the subject – based on the textbook’s domain (e.g., legal, scientific, etc.).
  * Independently answerable – avoid direct references to theorems, figures, or text numbers.
  * Varied in difficulty - Make difficult same as the provided examples.
  * Use same format as the provided examples.

  Strictly follow this format for each question your generate while responding

  [QUESTION]
  <Insert question here>
  [END]


examples: |
  Here are some examples of questions:

  [Document]
  {{icl_document}}

  [QUESTION]
  {{icl_query_1}}
  [END]

  [QUESTION]
  {{icl_query_2}}
  [END]

  [QUESTION]
  {{icl_query_3}}
  [END]

generation: |
  Here is the document:
  
  [DOCUMENT]
  {{document_outline}}
  {{document}}

start_tags: [""]
end_tags: [""]
```
Prompt for generating Answer:
```yaml
system: You are a very knowledgeable AI Assistant that will faithfully assist the user with their task.

introduction: Answer the question based on the provided document.  

principles: |
  The answers should:
  * The answer is grounded in the provided document.
  * Follows the format and style of the provided examples.
  * Directly answers the question.
  Strictly follow this format for each question your generate while responding

  [ANSWER]
  <Insert answer here>
  [END]


examples: |
  Here are some examples of answers for given questions for a document:

  [Document]
  {{icl_document}}

  [QUESTION]
  {{icl_query_1}}

  [ANSWER]
  {{icl_response_1}}
  [END]

  [QUESTION]
  {{icl_query_2}}

  [ANSWER]
  {{icl_response_2}}
  [END]


generation: |
  Here is the document:
  
  [DOCUMENT]
  {{document_outline}}
  {{document}}

  [QUESTION]
  {{question}}

  [ANSWER]

start_tags: [""]
end_tags: [""]
```

Transcript. To do that and I just put the prompts here. You can always go back and read this. but the high level overview of this slide is I just need to create two separate prompts for question and answer. So the output would be of the left prompt list of questions, right prompt for each question the answer of it. And this is a neat solution because what this is saying is hey model think as much as you think but only generate the questions first. Think hard about what kind of questions you want to ask. 


Slide 23
Summary. Final QA pipeline: clean summary → questions (no thinking) → answers (with thinking) → evaluation (thinking removed). Evaluate using faithfulness, relevance, accuracy.

LLMBlocks: 3 Types of Summary -> Post Process Thinking -> LLMBlock: Generate Question base on qna yaml -> Post Process Thinking -> LLMBlock: Generate Answer base on qna yaml -> LLMBlock: Evaluate Faithfulness -> Post Process Thinking  -> FilterBlock: Filter on Score

Transcript. And then once you have it separately generate the answers by thinking about them again. And then when it generates answers. I'm not going to remove the think this time. I will keep it. For questions, I will remove all the thinkings. This is the new flow, an edited version of what we had before. I had three types of summaries. I removed the thinking out of it. Now I have nice clean summaries. Now I'm going to generate a series of questions. Remove the thinking. I have a list of questions. Then I'm going to go generate answers for each one of them. I'll give the summary, the question, and let it answer. And once I have the answer, I'll keep the thinking. Then I'm going to evaluate each answer to see if the answer is based on the context you provided. Again, I'm going to remove all the thinking. I let it think and then I'll remove it and then I'll just filter it based on how it got scored. If it got poor score, I don't want to keep that sample. High score means that the answer is actually based on your context. So, I'll keep it. That is the new flow. 



QA
Where are the evaluation metrics coming from?  They are from the original knowledge pipeline where we designed bunch of evaluations like faithfulness, relevancy, accuracy and in this case we're pulling that here. Just to give you a little bit more context, they actually come from rag evaluations. So in rag evaluation, faithfulness, relevance and couple other metrics are very common. Rag version is made for a particular model. When the model changes, you have to change the prompt and other things so that the metric is calibrated. Think about it as the calibrated version for the teacher model. It also goes to say that do not use rag without modifying it if you're changing the target teacher model. 

If you're trying to add something to where the teacher model can't figure out the question, can't figure out the way of answering it, the thinking process, then you can't add that information into the student model. Is that correct? What I'm saying is basically so at the moment this be considered distillation process from the nemotron model? On knowledge it is not distillation technically, because the knowledge is coming from the document. But here two things are happening. The knowledge is coming from your documents but the skill of thinking is being distilled from the teacher model. 

If the model cannot figure out the question, how do we know that it doesn't hallucinate one? That's where the faithfulness eval is. What it does is it checks every claim the answer makes against the document. so it's going to compare the claims from answer to the documents. If there is even one claim that is not supported by document, you basically throw it out. And of course, you can make it even better by saying, okay, let me apply this on the question. So I think one thing to maybe keep in mind is the reason they are more accurate is because they do think about the answer they are generating.  So their instruction following ability is remarkably high because they have the ability to verify themselves in the thinking block. So the same verification that we applying at the very end. Think of reasoning models as amortization of a verification process. So it has learned to search in its output space, verify, decide whether you want to go forward or you want to backtrack and then start again. The reason they're good at math or reasoning is because before they reach the answer they try the answer multiple times every single time they check whether it is correct they're following the instruction correctly and if they find it's not they correct so 

Can the filter block does it include the approach to either remove whatever didn't pass the filter or to mark it with some kind of tombstone flag is the only output that you would discard whatever failed the filter. 
At essence it's just annotating each sample and then you can score those samples. You can go and inspect it and see why the model thought it was in fact then if you already have something annotated you can run this pipeline and see how accurate the model is in filtering out. So if a customer is saying hey I want to check that this thing really works against hallucination.  You can create a data set where there are things that are hallucinated and things that you know are not hallucinated. Run the filter and see if you are above 90% accuracy or not. That would be a good way of proving to the customer that this filteration process does remove hallucinations. In fact, one of the PC's actually worked because we just applied filteration. So this filter blow actually comes as from one of the PCs we did where hallucinations are very high and when we brought this back in they went drastically low but I think this is a question that the moment people hear the word synthetic data they ask right even though as we were saying earlier in the old question that the knowledge is still coming from the document. It's not being distilled. Technically, it's not a hallucination or the chances are already very low. But if we want to go the extra mile and prove that hey this is 90% accurate or hallucinations are only 5%, which in the training will not matter right if it's that low and when you train the model it won't impact the final model as much we can go and prove that. I'm going to say one more thing if the customer already has a hallucination detection or some kind of verification you can literally wrap that function in the filter block and apply it.

I was wondering are there certain use cases that you would recommend using a reasoning model or given that they have superior ability to sort of filter out questions and answers like this, would we just want to use these uniformly across the board? If you look across any agentic tasks, the reasoning models are becoming the default for many reasons. One of them is when you start doing tool calling, the chances of the model generating the right signature is like a model like llama is pathetically low. If you're trying to get a model working with MCP. A real world scenario not where you have two MCP servers and you are giving it the question where it is going to call those things but you're in actual production environment. If you benchmark these non-reasoning models against claude, they are pathetically low. So that's the number one use case at least if you look at major AI so agentic application development processes where reasoning models are making their way but more specifically in finance domain which has almost always struggled with even in something as simple as rag, to perform queries on SCC filings and so on and so forth where the idea is to use these things as a research or a deep research agents. That's where you're trying to as part of your answer manipulate something. Do a mathematical query. As we have said many times, language models absolutely suck at any kind of mathematical manipulation. This was very pertinent for the finance domain. Even when they're doing rag, they probably want some kind of manipulation of a table or entries. That's where reasoning models are amazing.  In fact, one of the other demos is for the inference time scaling is on showing that if you apply the same technique to the finance bench and if you're not familiar with finance bench it's a rag benchmark for financial domain. The improvements are drastic like we can close the gap up to claude and GPT4 with reasoning models. 
The final one is I recently had a chat with the CTO of Fidelity. one of their biggest requirements right now is they're super impressed by the deep research agent for their traders that Gemini offers. problem is they can't put their proprietary data in Gemini.  their number one ask is can we get that on prem right and that's basically where reasoning models are the only possible way of doing it. 





Slide 24
Summary. Demonstrates end result: QA pairs with a clear reasoning trace. This thinking-aware synthetic data is now ready to train a student model.

Transcript. Once I make these changes where I have replaced the QA block with the separate question and answer block and inject the postprocessing block. This is the answer I would expect to get out of this or this is what I get out of this which is it generated a question for me. Then it gave me an answer. But before that, it gave me a really nice thinking block where it kind of goes through the document, goes over every different fact, reasons about it, and then finally it gives me a very focused targeted answer. and This is what we would want the student model to learn, which is the thinking process on your document. With this what we have is synthetic data that can think and we can now start training a model. 

New slide (I couldn’t find this slide from the slides shared with me)
So I have some reference slides on how do I actually generate synthetic data. it only takes four lines. You simply point it to the flow that we created that is in yaml.  You simply point it to that and then you create the object and start generating the data


New slide (I couldn’t find this slide from the slides shared with me)
This is my training script that uses the training hub. You can go back try this out if you like and I think they also have a notebook as an example that you could try and then I'll leave the slide in here for you to look at later.




Slide 25
Summary. Introduction to the next topic: improving the pipeline by going beyond just three basic summaries.
Transcript. We're moving to the next section which is how can I make it better. So I'll take the knowledge pipeline and make some changes so that it starts doing more than three simple summaries. 













