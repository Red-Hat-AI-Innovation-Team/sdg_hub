## Slide 25 We can make this better:
* We can go beyond the 3 summaries by asking the thinking model to brainstorm different ways of summarizing
* We can do this with existing blocks

## Slide 26 Prompt 1:

### Add prompt for creating summarization instruction

```yaml
system: You are an AI assistant that is expert at summarizing text.

introduction: |
    Given below document, analyze it, and generate a list of 10 diverse instructions for summarizing it.
    Each instruction should vary in perspective, tone, or purpose and should be relevant to the document. Keep them short and distinct.

examples: |
    Example:
    1. Summarize the article in simple terms for a 10-year-old.
    2. Highlight the major arguments and counterarguments.
    3. Provide a summary focusing on implications for future research.

generation: |
    Document:
    {{document_outline}}

    {{document}}

    Now given above document, generate 10 more diverse instructions for summarizing it.
```


## Slide 27 Prompt 2:
### Add prompt for summarizing based on instruction
```yaml
system: |
  You are an AI assistant that is expert at summarizing text.

introduction: |
  Given below document, summarize it using the following instructions:
  {{summary_instruction}}

principles: |
  - Include as much of the document as possible to create a comprehensive summary.
  - If there are tables, include all the data of the table in the summary.

examples: ""

generation: |
  Document:
  {{document_outline}}

  {{document}}
```
(Transcript: boston-abacus-3s336: I'll start sharing my screen. So, quickly recap, this is kind of where we were which is I had the knowledge pipeline.  I injected some new blocks in it highlighted by green and we were able to get the final output. There is one more block called a regular expression block. I have added the details for it at the end of the slides if you would like to go and look at it. It's similar to postprocessing. it just uses regular expression. you need it to convert say the response that is list into actual Python list.
boston-abacus-3s336: so you can find the details at the end and then the diagram on how it fits. It's just a utility. So I guess we can start so we can change how the pipeline works. the way we will do it is if you remember it used to summarize the document into three types. one can say I can do more than three types and I can come up with the types based on the document. So a finance document is different than a document or an article on some topic. So the summary types might also be very different.
boston-abacus-3s336: And given that we are now using a reasoning model, it can actually brainstorm different ways of summarization. So the more summarization we do, the more synthetic data we different types of synthetic data we can generate. So let's make that change. So we're going to move away from this three types of summaries into a brainstorming block and then the summarization block based on the results of the brainstormings. 
boston-abacus-3s336: And then in generation just put the variable of your document and then it will runtime fill up your document in there. And the second prompt is the actual summarization. So I just take the summarization prompt. I modify it to now have summarization instruction as a variable. So this is the output of previous block. It gets populated and then everything else remains same.  So it is just saying that hey follow this instruction and summarize based on that and just with these two prompts we can simply reuse our LLM block and point them to these two new prompts 
)

## Slide 28 Now let us edit the flow


### **Flow Pipeline**

#### 1. **LLMBlock: Generate Summary Instruction**

* **Goal:** Create diverse summarization instructions (e.g., “Summarize for a policymaker” or “Simplify for a 10-year-old”).
* **Purpose:** Promotes varied outputs for better training or robustness.

#### 2. **Post Process Thinking**

* Applies transformation:
  * Remove the thinking block (<think>...</think>) from the teacher model's response

#### 3. **LLMBlock: Generate Summary Based on Instruction**

* **Input:** Document + Instruction
* **Output:** A targeted summary aligned with the style or intent of the instruction.

#### 4. **Post Process Thinking**

* Applies transformation:
  * Remove the thinking block (<think>...</think>) from the teacher model's response

#### 5. **LLMBlock: Generate Question**

* **Goal:** Create a question based on the generated summary.
* Follows the question style of QnA YAML's seed examples.

#### 6. **Post Process Thinking**

* Applies transformation:
  * Remove the thinking block (<think>...</think>) from the teacher model's response

#### 7. **LLMBlock: Generate Answer**

* **Input:** Document + Question
* **Output:** Faithful answer using source document, not just the summary.

#### 8. **LLMBlock: Evaluate Faithfulness**

* **Goal:** Judge whether the answer is faithful to the document (not hallucinated).
* Returns a **score** or **label** (faithful/unfaithful).

#### 9. **Post Process Thinking**

* Applies transformation:
  * Remove the thinking block (<think>...</think>) from the teacher model's response

#### 10. **FilterBlock: Filter on Score**

* **Final gate:** Removes Q\&A pairs below a faithfulness threshold or quality bar.
* Ensures only high-quality, faithful pairs are retained.


LLMBlocks: Generate Summary Instruction -> Post Process Thinking -> LLMBlocks: Generate Summary Based on Instruction -> Post Process Thinking -> LLMBlock: Generate Question base on qna yaml -> Post Process Thinking -> LLMBlock: Generate Answer base on qna yaml -> LLMBlock: Evaluate Faithfulness -> Post Process Thinking  -> FilterBlock: Filter on Score


(boston-abacus-3s336: and our final flow looks something like this. So I've replaced the three summary block with these new blocks.
boston-abacus-3s336: First one instruction generation I'm going to postprocess then I'm going to do a summary based on the instruction post-process and then I'll follow my usual flow of creating question and then answer and then evaluation let me show you how it looks like so that you have an idea of what's happening behind the scen so I have a super short document on electric vehicles the output of the very first block of brainstorming is something  like this which is it looked at a document that talks about EVs and these are the different summary types it came up with you can write an academic essay you can write a comparative essay maybe you can write an essay from an environmental point of view and etc and then let's pick one and you can see on the summary it has generated for each point it will do a very distinct summary with focusing on distinct parts of the document)

## Slide 29 How does the output of First two stage look like?

Output of the Summarization Instructions Block after Thinking Post-Process:
```text
1. Academic Essay: Summarize for a college
research paper...
2. Comparative: Contrast EV benefits with those of
hybrid...
3. Environmental Focus: Highlight...
4. ...
5. ...
```

Generated Summary for no 3. (Environmental Focus):
```text
# Electric vehicles (EVs) offer two primary ecological advantages:
(1) **zero tailpipe emissions**, reducing air pollution in urban areas and lowering
greenhouse gas emissions, and
(2) **reduced dependence on fossil fuel**...
```

## Slide 30 Putting Everything Together

* We now have a flow that can summarize a document in different ways.
* It can now generate answers with thinking

(boston-abacus-3s336: so maybe it might help to explain why are we approaching it this way. I'll explain why the reasoning the knowledge pipeline is a structured the way it is a structure.  So there's a lot of early work in psychology research that happened in u early 1900s on understanding human reading comprehension right so when humans read a new piece of text really dense what actually happens so there's a couple of models there's a popular one called construction integration
boston-abacus-3s336: But the general gist is humans don't directly go to think about question and answers. So what they actually do is they go through an intermediate representation, And this intermediate representation could be of three four different types. when we say summary, that's kind of what we are talking about here.
boston-abacus-3s336: So what we're saying we're going to model the knowledge pipeline after how humans do reading comprehension right so initially in the first version of the knowledge pipeline we would take a piece of context and ask a generate question and answer directly what we found is that as we migrate towards this intermediate representation where I don't think has the time to go into the details of what are those intermediate representation but I'll give you a very  high level summary. One of them is think of it as a global summary, So if you have to summarize Game of Thrones and a paragraph, what would that look like? that's a global summary.
boston-abacus-3s336: there's something called atomic facts or this is a summary where we point out in almost like a bullet point list of things right very fine details right that I'm forgetting my game of thrones completely but Jonas Snow was dead in episode this and then whatever I don't know and that he knows nothing and all that and he knows everything  Those will be your atomic summaries. And then there's a third type of summary which is think of it as a relational summary where you look at all the entities and then you try to create a relationship between them and you summarize that.
boston-abacus-3s336: So now you have these three form of summaries right and then you go or what this old literature would call intermediate representations and then for each one of those you go and create these and as you can see in this case you would need to create different kind of because they are capturing different aspects right so the new knowledge pip pipeline.
boston-abacus-3s336: The reason it is fact in this particular way is because a very similar process goes through your mind when you're reading a piece of text for the first time. It's specifically like a technical piece of text. if that helps with how we are structuring this and why this is grounded into wellound science and not sort of opinionated version of what we think is the best option.
)

## Slide 32
(boston-abacus-3s336: this is the section you'll find at the end of the notebook which is data mixing and training. So the data mixing if I draw it out as block diagram is something like this. So we first generated thinking data. I'm going to clean it up and then convert it in something messages format is what I had shown earlier at the start of the slide. Let me see if I can call a messages format. This is a universal format everyone use uses for training.
boston-abacus-3s336: So it's always good to convert it into that and then you can plug it into any training pipeline you would like including the instruing package that completion is for inference…
Noel O'Connor: Is that the chat completion format that's used in OpenAI?
boston-abacus-3s336: but I believe it receives the requests in this message.
Noel O'Connor: Yeah, I know. But it seems very similar.
boston-abacus-3s336: Yeah, it's always good to have a global format that everyone understands. I think messages does that. So I would like to convert that into messages.
boston-abacus-3s336: Then I'm also going to go and generate data with my original knowledge pipeline just so that I can show the model what thinking and not thinking looks like. This is optional. You can do it or you can just skip it. But if you have that just convert it to messages and then I'm going to take some neotron instruction data.  So this is something that Nvidia combine that with our data and then now the model can learn your thinking while retaining its skills of thinking on other domains. Maybe just to double click on this point because I think no brought it up earlier that hey there are two things happening here and you're learning the skills of reasoning and knowledge. So that's the part that captures the skill this buffer if you will.
boston-abacus-3s336: Yeah. that we bring back from general reasoning technically optional depending on how much is your overall reasoning workload but good idea to include it as a small payload. And this data is huge so you will have no problem subsetting it and matching it to your data.  With this I think this kind of goes we're at the end of the slides where we show the process with this you're good to train a model and then at the end I've included some reference slides on how to train the regular expression block that I mentioned before and how I'm using it on the right is the code for using it for executing it.
)



Absolutely! Here's the cleaned-up and anonymized version of the transcript with speaker names changed to neutral roles like `Audience 1`, `Audience 2`, etc., and the presenter referred to consistently. It preserves structure, clarity, and technical depth:

---

## **Replay Buffer & Data Mixing Clarification**

**Audience 1**: You mentioned a “replay buffer” — could you clarify what that is? I may have missed that part.

**Presenter**: Of course. This step is slightly beyond the core SDG pipeline — it’s part of training preparation. After generating data, you often want to mix it with other datasets or skill samples. On Day 1, we discussed that you can generate data for multiple skills, either individually or combined, and then merge them before training. That’s more efficient, especially for smaller datasets.

Here, we’re customizing a NeMoTron model. NVIDIA released a massive dataset of reasoning samples used during pretraining. We typically take a very small subset — maybe a few thousand examples — and mix it into our generated data. This subset is referred to as the **replay buffer**. It serves as a small “reminder” to the model, since it has already been trained on it.

**Audience 1**: So it's kind of like a sampling technique for the final training set, right?

**Presenter**: Exactly. It’s not part of SDG itself, just something you can add during training prep. There’s no direct dependency between SDG Hub and this module. In fact, in version 1.5, this is handled for you. But the idea here is to show you how to configure this manually — especially useful for advanced users or customer engagements.

**Audience 1**: That makes sense — especially helpful if a customer wants to inject their own dataset into the process.

**Presenter**: Right. They’d just need to modify the data mixing API accordingly.

---

## **Notebook Walkthrough: Prompt Blocks & Seed Data**

**Presenter**: Let’s switch to the notebook now. The only change you need is replacing `localhost` with the remote model endpoint you've been using in your demos.

Once connected, we begin with constructing ICL (in-context learning) examples. Early notebook cells show different prompts overwriting each other — you can skip those. They’re just my scratchpad for testing prompts.

The idea is to create Q\&A examples using your documents. These are structured as Python dictionaries (instead of YAML for simplicity). The goal is to teach the model to read a document, reason over it, and answer related questions. This produces your `seed_data.json`.

---

## **Block Registration and Flow Configuration**

**Presenter**: Now we move to registering prompt templates and blocks. In this demo, I’m writing blocks inline in the notebook — just for demonstration. Normally, you should write blocks in the `blocks/` directory as separate files.

The flow we’re using is called `summary_diversity`. It includes blocks for:

* Summarization
* Question generation
* Regex-based postprocessing
* Evaluation

All these blocks are modular. You can reuse them by just switching the prompts. Once the blocks are set up, you point to a YAML file and create the SG object, selecting one row to test.

Generated outputs include summaries, questions, answers, evaluations, etc. Each step’s output can be inspected individually.

---

## **Generated Data Example**

**Presenter**: One document was about dental cosmetic surgery. The model generated summary instructions like:

* "Critique with irony"
* "Patient warning"
* "Economic case study"

Each instruction yielded a unique summary. Then, we generated questions and answers for each. The final dataset includes:

* Document
* Instruction
* Summary
* Q\&A
* Evaluation (faithfulness score, reasoning trace)

This full set is ready for training.

---

## **Training Setup**

**Presenter**: You’ll find training scripts at the bottom of the notebook. You can substitute your own models — e.g., DeepSeek — and run training with this same data. Just change the teacher model if needed.

train.py using training_hub
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
    # parser.add_argument('--chat_tmpl_path', type=str, required=True,
    #                   help='Path to the chat template file')
    parser.add_argument('--exp_dir', type=str, required=True,
                      help='Path to the experiment directory')
    parser.add_argument('--parent_exp_dir', type=str, required=True,
                      help='Path to the parent experiment directory')
    parser.add_argument('--num_epochs', type=int, default=5,
                      help='Number of epochs to train')
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
        max_seq_len=9000,
        max_batch_len=25000,
        num_epochs=args.num_epochs,
        effective_batch_size=128,
        learning_rate=5e-6,
        warmup_steps=25,
        save_samples=0,
        use_dolomite=False,
        checkpoint_at_epoch = True,
        accelerate_full_state_at_epoch = False,
        process_data=True,
        distributed_backend=DistributedBackend.DEEPSPEED,
        fsdp_options=FSDPOptions(cpu_offload_params=False),
    )

    run_training(torch_args=torch_args,train_args=train_args)

if __name__ == "__main__":
    main()
```
---

## **Live Q\&A**

### **Audience 2**: How many samples do we need to meaningfully improve model performance?

**Presenter**: Great question. For general tasks like table manipulation or writing in a certain style, 50–100 examples can be sufficient. You're not trying to get the model to memorize — you're helping it interpolate.

When it comes to reasoning tasks, it depends more on *tokens*, not *samples*. One document might produce 300–500 Q\&A pairs. What matters more is the total number of tokens the model sees.

There’s a research paper showing recall performance as a function of token count — I’ll share that. We avoid raw document injection for this reason — documents are short, but reasoning-generated data yields more tokens per concept.

During generation, you can go large. But for training, especially data mixing, you might want to subsample smartly.

---

### **Audience 3**: What are some practical use cases where reasoning models outperform regular models?

**Presenter**:

1. **Agentic workflows**: Tool use with MCP servers or API calls. General models hallucinate tool names or get signatures wrong. Reasoning models are more precise — reducing deployment bugs drastically.
2. **Finance domain**: Tasks like computing total cash flow across filings, or summarizing complex reports. Even 5B models struggle here. We’ve seen 20–30% RAG improvements in these use cases.
3. **Deep research agents**: Used in institutions like Fidelity for internal analysis. These agents answer deep, technical queries. OpenAI models performed well here, but for proprietary data, custom reasoning models are preferred.

We also cover this in our “Inference Time Scaling” podcast. I’ll share the link.

---

### **Audience 4**: How well do these skills transfer to user input in the wild?

**Presenter**: Excellent question. A year ago, small models couldn’t generalize well to user phrasing. Now, with proper post-training (e.g., RLHF), even small models handle linguistic variation quite well.

If you don't break the model during post-training (e.g., by overwriting its skills), generalization is strong. As long as the model *knows the content*, it can handle variations in how questions are phrased.

For reasoning-specific skills, you need very little additional training — just enough to establish the pattern.

---

## **Wrap-up**

**Facilitator**: Thanks, everyone. We're slightly over time, but that was a great session. I’ll post notes and prep for tomorrow in Slack.

**Presenter**: Thanks all! Looking forward to continuing tomorrow.
