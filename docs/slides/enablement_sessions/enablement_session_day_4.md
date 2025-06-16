Transcript of Day 4 workshop on sdg_hub

We recap previous sessions: Adapting synthetic data pipelines for reasoning models which includes introducing thinking blocks to postprocess reasoning model output, and demonstrating skills use cases like document-to-table conversion, structured summarization, and multi-step task customization — all within an extendable, modular data generation framework (sdg_hub).

Q&A Transcript (Cleaned Up)
audience 1:
 With all this tuning for reasoning models, we're not teaching the model the actual tokens or specific steps, right? We're just showing it how to think — patterns of thinking, not exact procedures?
presenter:
 Exactly. The student model here is also a reasoning model, so it already knows how to think. What we’re doing is guiding it on how to apply that reasoning to our data using its pre-trained capabilities.
If you want to influence how it reasons, you need to be prescriptive when generating your "thinking data." For instance, if you want it to break a topic into steps — like listing sources, selecting the right one, then critiquing — you must reflect that in the prompt used for generating that block.
audience 1:
 Let’s say we go into a customer scenario. A few years back, we worked with Large European Bank on workflows — they had thousands. Could we train a model to understand or assist with those workflows?
presenter:
 Potentially, yes — it depends on the task. If the goal is to assess whether workflows are successful or complete, then a reasoning model is well-suited. Suppose you want the model to verify each workflow step — you’d prescribe those steps and have the model check them, reporting any missing parts.
You could even build this without customization, just through a structured pipeline. A strong reasoning model should follow the logic as long as it's broken into discrete, modular tasks (e.g., Step 2: check condition, Step 3: convert to JSON). That structure often works better than throwing the whole workflow at it and asking for an assessment.
audience 2:
 I have a question about processing. If you have multiple Q&A YAML files with various context sections, are they processed sequentially or in parallel? And within each YAML, does it handle one context and Q&A at a time?
presenter:
 With the new SDG Hub, processing is handled independently. It depends on your backend and batching configuration. If batching is set up for parallel processing, it will process in parallel. Otherwise, it follows the logic you’ve implemented. SDG pipelines are more flexible than just data generation — as Noel pointed out recently.
audience 3:
 Can this setup be used for code generation use cases?
presenter:
 Absolutely. You can switch the teacher model to a code-capable one and adjust the prompt accordingly. We’ve used this framework for everything from natural language tasks to code.
audience 3:
 I was unsure because code inputs can be large and include tricky characters. Have we run into any issues?
presenter:
 We've used this approach with code without major problems. The flow is flexible — blocks can be configured for any use case. Just switch the teacher model to one that supports code (e.g., LLaMA 7TV), and you’re good to go.
audience 2:
 We’ve created synthetic data. How do we know if it’s ready for the next step, like fine-tuning?
presenter:
 As long as your data is in the correct messages format, it’s ready for training. That’s the only requirement.

presenter: use case three was that where we didn't want the model to hallucinate the context. we wanted to provide our own documents and ask a fixed question which was a structured summary. So that's bucket two.  And then in use case two, we wanted to take our unstructured document (doesn't really matter where it comes from) to then convert it into a table.

presenter: So those were pure skills. But then additionally we also showed you yesterday how you can customize a reasoning model which is technically a knowledge task. Though it is subtly a skill as well (doesn't really matter).  The takeaway there was how do you now take now that the world has moved on to reasoning models how does the SDG (synthetic data generation) process change to accommodate for that and the answer was very simple. we created a pipeline in which we were not only generating the question and the answer based on our given context but also a thinking block.

presenter: that's the full summary for the last three days. so what we thought today to do was maybe break ourselves into little themes and using the notebooks that we have used for the first two sessions, Where the backend was a llama model. essentially make little changes to help us understand how to modify and adapt these pipelines for more complicated use cases. So here are three examples and of course we don't have to limit ourselves to these three examples. 

presenter: and then in the other session what we'll try to do is add a block. so if you remember in the structured summary use case we went from a document to creating this sort of structured summary that you see on the left hand side.  And then what we're going to do is we're going to change (there's a little bit of mistake here). we're not replacing the summary block. We're creating one more block. that additionally does a TLDR. So you have your normal summary, but then you add a block that then gives you a single line TLDR type thing.

presenter: if you're interested in the slides, we have shown you how to run the annotation pipeline. definitely try it out. It's a use case that I think comes out has been asked for very frequently. but as I said, it'd be hard to cover this in this one session. 


## Slides Summary 

### Recap of SDG Hub
SDG Hub organizes work into pipelines made up of linear sequences of blocks.
Blocks of the LLM type use prompts to control behavior.
Every SDG (skill or knowledge task) is a variant of this workflow.
You can wrap custom Python functions into blocks.
You’re not limited to predefined blocks — they’re extendable to code models, vision-language models, etc.
Three skill types exist (for now, based on version 1.5):
1. Free-form: Self-contained question and answer, generated using just a task description (e.g., manipulating a row in a table).
2. Context-grounded: Provided document + question, task data generated on top of user's document (e.g., summarizing a transcript).
3. Generic skill: Teach the model to generalize across diverse domains by generating varied contexts, synthetic document generation.

### Customizing Reasoning Models
Reasoning is technically a knowledge task, but closely related to skills.
We introduced a thinking block in the pipeline to support post-processing of a reasoning model's output.
This block generates reasoning steps alongside the usual Q&A output.
Adapting sdg_hub for reasoning models is simple: you just add support for reasoning model and block necessary to handle reasoning model's output.


