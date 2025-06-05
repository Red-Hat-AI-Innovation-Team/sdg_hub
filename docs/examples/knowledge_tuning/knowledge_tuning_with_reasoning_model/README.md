# Synthetic QA Generation for reasoning on your data

This notebook demonstrates how to use the `sdg` package to generate synthetic question-answer data with NEMOTRON Super as the teacher model. The generated data is designed to improve both the reading comprehension and reasoning abilities of a smaller student model, NEMOTRON NAMO, on a benchmark dataset of high-quality articles.

## Table of Contents
- [Overview](#overview)
- [Reasoning Pipeline Overview](#reasoning-pipeline-overview)
- [Notebook Structure](#notebook-structure)
  - [Creating ICLs](#creating-icls)
  - [Generating Data and training](#generating-data-and-training)
  - [Results](#results)
  - [Model Response Examples](#how-does-the-models-response-look)
- [Key Takeaways](#key-takeaways)
- [Changing backends](#changing-backends)

## Overview

The workflow includes:

- Defining an SDG pipeline using a YAML flow file
- Creating custom SDG blocks for handling output of thinking model
- Custom block for parsing the response into individual question-answer pairs
- Mixing synthetic QA data with instruction-tuned examples to form a training dataset
- Training a Nvidia Nemotron nano model on the the training dataset using our [training library](https://github.com/instructlab/training).

## Reasoning Pipeline Overview

```mermaid
graph LR
    A[Input Document] --> B[Summary Instruction Generation]
    B --> C[Summary Generation]
    C --> D[Question Generation]
    D --> E[Answer Generation]
    E --> F[Answer Evaluation]
```

### Summary Generation

* Our new SDG pipeline leverages the reasoning capabilities of language models to generate a diverse set of summarization instructions through structured brainstorming.
* These instructions are tailored to each document, enabling the creation of varied summaries from a single source.
* We found that generating multiple summaries—each with fewer than five QA pairs—produced the best performance on quality benchmarks.

### QA Generation

* Once summaries are generated, we use them to create question–answer pairs.
* We first generate a list of questions, extract the `<think>` tag, convert it into a list, and then generate an answer for each item.
* While generating answers, we preserve the `<think>` tag to enable downstream training that encourages model reasoning.

### Evaluation

* Since the answers are derived from the source document, it’s important to ensure they are properly grounded.
* To validate this, we use the faithfulness evaluation component from our knowledge pipeline.
* This evaluation checks whether each claim in a generated answer is supported by the document, and marks the answer as `No` if even one claim is not supported.

## Notebook Structure
The notebook is logically divided in below sections:
### Creating ICLs
- SDG works by creating a set Question-Answer pairs from the source document.
- To do this we first need to create an example document and set of Question-Answer pairs. The SDG will use these to generate more synthetic QA pairs on top of all your document.
- For example, I have a news article. I want to train a model to be able to answer any question about that article. 
- I can craft example Question-Answer that show that:
  - Deal with specific facts in the article
  - Deal with broader theme of the aritcle
  - Reason on multiple parts of the article
- To acheive these my example QA also in this context known as ICLs will be of these types.
- The first section of the notebook will show you how I used Nemotron Super to generate these with chaining prompts.

### Generating Data and training
- In next few sections you will:
    - Learn how we added prompts for new teacher models
    - Generated data and did data mixing for training
    - And used our training library to train the nemotron nano model

### Results

Once the model is trained we evaluated it in a **closed-book setting** on [QuALITY Benchmark](https://github.com/nyu-mll/quality/tree/main), where the model answers questions based only on what it has learned during training, without access to the original articles.

We also evaluate the base model and customized nano model in a **RAG setting** where we use OpenAI embeddings and cohere re-ranker to retrieve relevant context from the article.

- Evaluation measures how well the model has **memorized and internalized** information from the articles
- Accuracy is compared between the base nano model and the Quality Customized nano model

![Plot of % improvement in Quality Benchmark](assets/customized_nano_quality_results.png)

*Plot showing percentage improvement in performance on the QuALITY Benchmark in Closed Book and RAG (Retrieval-Augmented Generation) settings after training*


## How does the generated data look like?

#### Input Raw Document
```text
" It's Time To Keelhaul U-Haul!", Jeffrey Goldberg, 1999.
 It's Time To Keelhaul U-Haul! 

         Like all superheroes worthy of the title, the Shopping Avenger has an Achilles' heel. In the case of the Shopping Avenger, his Achilles' heel is not animal, vegetable, or mineral but something less tangible. 

         An explanation: Last week, the magazine you are currently reading forced the Shopping Avenger at gunpoint to read a series of treacle-filled self-help books, and then to . The Shopping Avenger, who can withstand radiation, extreme heat and cold, hail, bear attacks, and Eyes Wide Shut , almost succumbed to terminal jejuneness after reading these books. Except for one thing: One of the books, The Art of Happiness , which collects and simplifies the Dalai Lama's philosophy, got the Shopping Avenger to thinking. This, in a way, is the Shopping Avenger's Achilles' heel: thinking. Perhaps it is wrong, the Shopping Avenger thought, to complain about the petty insults and inconveniences of life in the materialistic '90s. The Shopping Avenger felt that perhaps he should counsel those who write seeking help to meditate, to accept bad service the way one accepts the change of seasons, and to extend a compassionate hand of forgiveness to those who provide poor customer care. 

         But then the Shopping Avenger sat down, and the feeling passed. 

         The Shopping Avenger does not make light of the Dalai Lama or of the notion that there is more to life than the impatient acquisition of material goods. If the Shopping Avenger were not, for a superhero, extremely nonjudgmental--as opposed to his alter ego, who is considered insufferably judgmental by his alter ego's wife--the Shopping Avenger would tell the occasional correspondent to let go of his petty grievance and get a life. 

         But the Shopping Avenger also believes that the Dalai Lama has never tried to rent a truck from U-Haul. If he had tried to rent from U-Haul, he never would have escaped from Tibet. (For the complete back story, see "Shopping Avenger" column and one.) 

         The complaints about U-Haul's nonreservation reservation policy continue to pour in through the electronic mail. One correspondent, B.R., wrote in with this cautionary tale: "Last weekend, I went to San Francisco to help my brother and his family move into their first house. My brother had reserved a moving truck with U-Haul for the big day. I warned my brother about U-Haul's 'not really a reservation per se' policy that I learned from the Shopping Avenger. He didn't believe such a thing would happen to him, so he didn't act on my warning." 

         B.R. continues--as if you don't know what happened already--"I went to U-Haul with my brother to get our 'reserved' truck. The store had many customers standing around looking frustrated. When we got to the front of the line, the clerk informed us that our 'reserved' truck had not yet been returned. We asked if we could rent one of the many trucks sitting idle in the parking lot. The clerk laughed and said the keys to those trucks were lost." 

         B.R. and his chastened brother--the Shopping Avenger is resisting the urge to gloat--went to Ryder. "Ryder had a truck available for us. The gentleman who helped us at Ryder said Ryder prides itself on being everything U-Haul is not." 

         The Shopping Avenger has still not received a call from U-Haul spokeswoman Johna Burke explaining why U-Haul refuses to provide trucks to people who reserve trucks, but the Shopping Avenger is pleased to note that several correspondents have written in over the past month saying that, based on what they have read in this column, they will be taking their business to Ryder or Budget or elsewhere. 

         The Shopping Avenger will undoubtedly return to the sorry state of affairs at U-Haul in the next episode, but now on to this month's airline debacle. 

         Before we begin, though, the Shopping Avenger nearly forgot to announce the winner of last month's contest, in which readers were asked to answer the question, "What's the difference between pests and airlines?" 

         The winner is one Tom Morgan, who wrote, "You can hire someone to kill pests." Tom is the winner of a year's supply of Turtle Wax, and he will receive his prize just as soon as the Shopping Avenger figures out how much Turtle Wax actually constitutes a year's supply. The new contest question: How much Turtle Wax comprises a year's supply of Turtle Wax? 

         This month's airline in the spotlight is Southwest. Loyal readers will recall that last month the Shopping Avenger praised Southwest Airlines for its "sterling" customer service. This brought forth a small number of articulate dissensions. The most articulate, and the most troubling, came from M., who wrote, "Last year, flying from Baltimore to Chicago with my entire family (two really little kids included), we set down at Midway in a rainstorm. And waited for our bags. And waited for bags. And waited for bags." 

         An hour later, M. says, the bags showed up, "soaked through. We took them to baggage services at SW and were faced with the most complicated, unclear, and confusing mechanism for filing a claim we experienced flyers have ever seen." 

         When they arrived at their destination, M. and her family made a terrible discovery, "We discovered that our clothes were soaked through--the top clothes were so wet that the dye had bled through down to the lower levels, destroying lots of other clothes. Obviously, our bags had just been sitting out on the runway in the rain. To this day, I've never heard a thing from SW, despite calls and letters." 

         This, of course, is where Shopping Avenger steps in. Shopping Avenger knows that Southwest is different from the average airline, in that it doesn't go out of its way to infuriate its paying customers (see: ), so I expected a quick and generous resolution to M.'s problem. 

         What I got at first, though, was a load of corporate hoo-ha. 

         "The airline's policy, which is consistent with all contracts of carriage at all airlines, requires that passengers file a report in person for lost or damaged luggage within four hours of arrival at their destination," a Southwest spokeswoman, Linda Rutherford, e-mailed me. "[M.] indicates she called for a few days, but did not file a report in person until April 12--three days later. Southwest, as a courtesy, took her report anyway and asked for follow up information and written inventory of the damage." Rutherford said that M. should have submitted detailed receipts and photographs of the damage in order to make a claim. 

         Harrumph, the Shopping Avenger says. It is a bad hair day at Southwest when its officials defend themselves by comparing their airline to other airlines. I forwarded this message to M., who replied: 

         "Wow. Well, of course I didn't file it at the airport on the 9 th because I didn't know the clothes were ruined at the airport. I didn't know until I opened the baggage at my hotel and saw the ruined stuff. (And it's worth noting that we had already waited for about an hour for our luggage with two little kids and impatient in-laws nipping at our heels.)" 

         She goes on, "I did call that evening ... and was told that that sufficed. This is the first time I've been told that I had to file a complaint in person within four hours. ... When I filed on the 12 th , I was never told that I needed any receipts or photos or other type of documentation. The baggage folks seemed pretty uninterested in all of this. ... They know that the type of 'evidence' they want is impossible to obtain. They also know that on April 9 they screwed up the luggage retrieval and left bags out in the rain a long time." 

         Southwest's response actually served to anger M. more than the original problem. "Before, they had a mildly annoyed but loyal customer (who would have been placated by an apology and thrilled with some modest token of their regret). Now they have a pissed-off customer." 

         Things do look bad for Southwest, don't they? The Shopping Avenger sent M.'s response to Rutherford, who e-mailed back saying she thought the Shopping Avenger was asking for "policy information." The Shopping Avenger e-mailed back again, stressing to Rutherford that the Great Court of Consumer Justice would, if this case were brought to trial, undoubtedly find for the plaintiff (the Shopping Avenger serves as prosecutor, judge, and jury in the Great Court of Consumer Justice--defendants are represented by the president of U-Haul), and that Southwest was precipitously close to feeling the sword of retribution at its neck. 

         But then she came through, provisionally, "Yep, you can be sure if [M.] will call me we will get everything squared away. I'm sorry it's taken this long for her to get someone who can help, but we will take care of it from here." 

         Stay tuned, shoppers, to hear whether Southwest makes good it promise to compensate M. and apologize to her for her troubles. 

         The story of M. reminds the Shopping Avenger of a central truth of consumer service: It's not the crime, it's the cover-up. 

         Take the case of K., who found himself waiting in vain for Circuit City to repair his television. Televisions break, even 1-year-old televisions, as is the case with K's. But Circuit City, where he bought the television, gave him a terrible runaround. The Shopping Avenger dispatched his sidekick, Tad the Deputy Avenger, to get to the bottom of K.'s story. This is what he found: K. grew concerned, Tad the Deputy Avenger reports, after his television had been in the Circuit City shop for a week. When he called, he was told to "check back next week." When he asked if someone from the store could call him with more information, he was refused. Weeks went by. When K. told one Circuit City employee that he really would like to get his television back, the employee, K. says, asked him, "Don't you have another television in your house?" 

         More than a month later--after hours and hours and hours of telephone calls and days missed at work--K. received his television back. 

         Mistakes happen, but not, Tad the Deputy Avenger found out, at Circuit City. The case, K. was told by a Circuit City official, was "handled perfectly." Another official, Morgan Stewart in public relations, assured Deputy Avenger Tad that "We got to be a big and successful company by treating customers better than the other guy." The Shopping Avenger and his loyal sidekick would like to hear from other Circuit City customers: Does Circuit City, in fact, treat its customers better than the other guy? 

         Stay tuned for answers. And next month, a Shopping Avenger clergy special: TWA screws with a Hasidic rabbi's travel plans, leaving the rabbi's wife crying at the airport. Find out if the Shopping Avenger can save TWA from certain heavenly punishment, in the next episode. 

                                            Got a consumer score you want settled? Send e-mail to shoppingavenger@slate.com.
```
#### Generated Summary Instruction
```text
**Summarize for a college audience** focusing on the intersection of consumer rights and corporate accountability.
```

#### Generated Document Summary
```text
**Summary: "It's Time To Keelhaul U-Haul!" and the Intersection of Consumer Rights and Corporate Accountability**  

Jeffrey Goldberg’s 1999 column, "It's Time To Keelhaul U-Haul!", features "The Shopping Avenger," a satirical consumer advocate, exposing corporate failures in honoring commitments and addressing grievances. The piece highlights three cases illustrating the tension between consumer rights and corporate accountability:  

1. **U-Haul’s Deceptive Reservation Policy**  
   U-Haul’s practice of accepting "reservations" without guaranteeing vehicle availability is critiqued through a reader’s experience. Despite a confirmed reservation, the company denied access to a truck, citing unreturned vehicles and "lost" keys for idle trucks. The Shopping Avenger contrasts this with Ryder’s reliability, noting that readers began boycotting U-Haul in response to such practices. The incident underscores how corporations may prioritize operational convenience over contractual obligations, eroding trust.  

2. **Southwest Airlines’ Baggage Fiasco and Bureaucratic Deflection**  
   A family’s luggage was left in the rain during a Southwest flight, destroying clothing. Southwest initially blamed the passenger for not filing a claim in person within four hours—a policy the family was never informed of. The Shopping Avenger intervened, prompting a spokesperson to eventually promise resolution, but not before the airline’s defensive response exacerbated the issue. This case demonstrates how rigid adherence to policies can override customer fairness and how advocacy can force accountability.  

3. **Circuit City’s Neglect and Denial**  
   A customer’s television repair was delayed for over a month, with Circuit City staff showing indifference (e.g., asking if the customer had another TV). Despite the company’s claim of "perfect" handling, the Shopping Avenger challenged this narrative, inviting public testimony to counter corporate self-exoneration. The example illustrates how companies may dismiss consumer complaints while touting superior service.  

**Analysis for a College Audience**  
These cases illustrate systemic gaps in corporate accountability:  
- **Deceptive Practices**: U-Haul’s reservation policy exemplifies how companies may exploit legal loopholes to prioritize profits over transparency.  
- **Deflective Accountability**: Southwest and Circuit City’s initial responses—blaming policy or the customer—reflect a common corporate strategy to shift responsibility away from operational failures.  
- **Power of Advocacy**: The Shopping Avenger’s interventions highlight how public pressure and media scrutiny can compel corporations to address grievances they might otherwise ignore.  

The column argues that consumer rights are often compromised not by mistakes alone but by corporations’ refusal to acknowledge and rectify them transparently. It emphasizes the importance of vigilant advocacy in holding companies accountable, particularly when legal or policy technicalities are used to evade responsibility. For colleges, this raises questions about ethical business practices, the role of consumer activism, and the need for regulatory oversight to protect rights in a market-driven economy.  

**Key Takeaway**  
The intersection of consumer rights and corporate accountability hinges on transparency, empathy, and willingness to resolve grievances fairly. When companies prioritize bureaucratic self-protection over customer justice, consumer advocacy becomes critical in restoring balance—a dynamic relevant to ongoing debates about regulation, corporate ethics, and the leverage of public scrutiny.
```


### Generated data

#### Question
```text
Apply the framework of "deflective accountability" and "advocacy-driven resolution" to a hypothetical scenario involving a pharmaceutical company’s handling of drug side-effect complaints. How might regulatory oversight influence the dynamic between corporate responsibility and consumer protection in this context?  
```
#### Generated Answer
```text
<think>
Okay, let's tackle this question. First, I need to understand what "deflective accountability" and "advocacy-driven resolution" mean based on the document. From the examples, deflective accountability is when companies blame policies, customers, or other factors to avoid taking responsibility. Advocacy-driven resolution happens when external pressure, like media or consumer groups, forces the company to fix the issue.

The question is asking to apply these concepts to a hypothetical pharmaceutical scenario. Let me think of a plausible case. Maybe a drug has side effects, and the company initially denies responsibility or blames the patient. Then, advocacy groups step in, leading to a resolution.

Now, how does regulatory oversight play into this? The document mentions that advocacy can force accountability, but regulations could either support or hinder this. If there are strong regulations, maybe the company can't deflect as easily. But if regulations are weak, the company might rely more on deflective strategies until advocacy pushes them.

I should structure the answer with the hypothetical scenario first, then apply the two frameworks, and finally discuss regulatory influence. Make sure to tie it back to the document's analysis, like how deflective accountability works and how advocacy drives change. Also, mention the Key Takeaway about transparency and empathy.

Wait, the answer needs to be concise. Let me outline:

1. Hypothetical: Pharma company's drug causes side effects. They blame pre-existing conditions or say it's rare, ignoring complaints.
2. Deflective Accountability: Company uses medical jargon, cites approval processes, shifts blame.
3. Advocacy-Driven Resolution: Patient groups, media expose the issue, leading to investigations and recalls.
4. Regulatory Oversight: Strong regulations (like mandatory reporting) would reduce deflective tactics. Weak regulations allow more deflection until advocacy steps in. Regulatory bodies can enforce transparency, balance corporate and consumer interests.

Check the document's examples: U-Haul's policy, Southwest's blame-shifting. The analysis part talks about systemic gaps and the role of advocacy. The Key Takeaway emphasizes transparency and advocacy's role when companies prioritize self-protection.

Make sure the answer references these elements. Also, the implications of challenging the assumption in the previous example might not be relevant here, but the structure of applying the frameworks and discussing regulation is key.

Avoid going off-track. Keep each part clear. Use terms from the document like "deflective accountability," "advocacy-driven resolution," and mention how regulations affect the dynamic, possibly by either enabling or mitigating the company's deflective strategies.
</think>


**Hypothetical Scenario**: A pharmaceutical company, "MediCorp," releases a drug with undisclosed severe side effects. When patients report issues, MediCorp attributes them to "pre-existing conditions" or claims the complaints are statistically rare, citing FDA approval as justification. Initially, they refuse compensation or recalls, blaming "isolated incidents" and emphasizing the drug’s "proven efficacy."  

- **Deflective Accountability**: MediCorp employs deflective strategies by (1) blaming patients’ health histories, (2) invoking regulatory approval to legitimize inaction, and (3) downplaying severity through statistical minimization. This mirrors Southwest’s policy-based deflection and Circuit City’s denial, prioritizing reputational protection over accountability.  

- **Advocacy-Driven Resolution**: A patient advocacy group, aided by media exposure, compiles evidence of widespread harm and organizes a boycott. Public pressure forces MediCorp to establish a compensation fund and recall the drug, illustrating how external scrutiny overrides corporate resistance, as seen in U-Haul’s boycott response.  

**Regulatory Oversight Influence**:  
- **Strong Oversight**: Mandatory adverse-event reporting laws and independent review boards could prevent deflective tactics by forcing transparency (e.g., public side-effect databases). Regulations might also impose fines for non-disclosure, aligning corporate incentives with consumer safety.  
- **Weak Oversight**: Without such measures, MediCorp could delay action until advocacy intervenes, replicating the "systemic gaps" noted in the column. Effective regulation would reduce reliance on advocacy by preemptively ensuring accountability, balancing corporate autonomy with consumer protection.  

This dynamic reflects the document’s thesis that corporate accountability often requires either robust regulation or public advocacy to counter self-serving practices, particularly when transparency and empathy are absent.
```




### How does the model's response look?

#### Example 1
**Prompt:**
```text
#### Question
 Defining Decay Down by David Plotz how do dentists sell unnecessary cosmetic procedures to their patients?  There is only one correct choice.
#### Choices
A. All of the options are correct
B. By relying on psychological tricks and societal pressure
C. By using medical jargon to confuse the patient 
D. By implying that the procedures are more necessary than they really are
#### Thought Process and Answer
Thought process:
```
#### Model Response
```text
<think>
Okay, let's tackle this question. The question is about how dentists in the context of "Defining Decay Down" by David Plotz sell unnecessary cosmetic procedures. The options are A to D.
First, I need to recall what "Defining Decay Down" is about. From what I remember, the article discusses how dentists often use certain techniques to make patients feel pressured into getting treatments they don't strictly need. The title itself, "Defining Decay," suggests that dentists create a narrative of inevitable decay in the mouth, making patients think they need various procedures.
Looking at the options:
A. All of the options are correct. That seems unlikely because the question states there's only one correct choice. So probably not A.
B. Psychological tricks and societal pressure. This aligns with the idea of creating a sense of necessity. Dentists might use psychological manipulation, like making patients feel they need a certain procedure because of societal expectations or their own perceptions.
C. Using medical jargon to confuse. Hmm, medical jargon is standard in healthcare, but does it confuse patients? Maybe, but I think the article focuses more on psychological pressure than confusing through jargon. Not sure yet.
D. Implying procedures are more necessary than they are. This seems similar to B but more specific. The article might argue that dentists frame procedures as necessary, even if they're cosmetic. For example, claiming that a small crack is inevitable and only they can fix it properly.
Now, comparing B and D. The article's premise is that dentists create a sense of decay (necessity) through psychological means. So B says they use psychological tricks and societal pressure. D says they imply procedures are more necessary. Which one is more accurate?
In the article, Plotz describes how dentists use psychological tactics. For instance, they might say that a small imperfection is not just cosmetic but also a sign of aging or wear and tear that's inevitable. They might also use societal pressure, like the idea that everyone is getting their teeth done, so you should too. So B seems to cover both psychological and societal aspects.
But D is about implying necessity. If the procedures are cosmetic but presented as necessary, that's part of psychological manipulation. So maybe B and D are related. Wait, the options are separate. Let me check the exact answer from the article.
From what I recall, the key point is that dentists create a narrative of decay, making patients feel they need various treatments. They use psychological manipulation (B) and societal pressure (part of B). So B would be the correct answer. D is a subset of B perhaps. But the options are distinct. The answer is likely B. Alternatively, maybe D is a trick. Let me think again.
The question is about selling unnecessary procedures. Dentists might use psychological tricks (like making patients feel they have to fix things) and societal pressure (like peer pressure to have a certain appearance). So B combines both psychological and societal factors. D is more about the necessity aspect. The article's main argument is that dentists create a sense of decay (necessity) through psychological means and societal expectations. So B is the correct answer.
</think>

Thought process: The article "Defining Decay Down" highlights how dentists use psychological manipulation and societal pressure to frame cosmetic procedures as necessary, even when they are unnecessary. Dentists create a narrative of inevitable "decay" in the mouth, making patients feel pressured to undergo treatments they don't strictly require. While option D ("implying necessity") is partially true, the broader and more accurate answer encompasses both psychological tricks and societal pressure, as described in the article. 

Answer: B.
```

---

#### Example 2
**Prompt:**
```text
#### Question
 Defining Decay Down by David Plotz how would Stephen compare humans and machines? There is only one correct choice.
### Choices
A. He thinks they are similar enough that a conflict will arise
B. They are complementary in their abilities and can benefit from one another
C. They operate with similar systems of intelligence but to entirely different ends
D. Humans are at risk of losing access to knowledge if they let machines take over most tasks
```

#### Model Response
```text
<think>
Okay, let's tackle this question. The question is about how Stephen compares humans and machines in the context of the article "AI: what's the worst that could happen?" by Harry Armstrong in 2017. The options are A to D.
First, I need to recall the article. From what I remember, the article discusses the potential risks and benefits of AI. Stephen, who I think is a character or a person mentioned in the article, might have a specific viewpoint. The key here is to figure out his comparison between humans and machines.
Option A says they are similar enough that a conflict will arise. That sounds like a negative outcome where there's a conflict between the two. But Stephen's view might be more about their relationship than a conflict.
Option B suggests they are complementary and can benefit each other. This is a positive view, where they work together. But the article might be discussing risks, so maybe Stephen is more cautious.
Option C states they operate with similar intelligence systems but for different ends. This implies a structural similarity but different goals. The article might mention that AI systems are becoming too intelligent, leading to a loss of human agency.
Option D claims humans are at risk of losing knowledge if machines take over tasks. This is a common concern about AI replacing human work, leading to job losses and knowledge erosion.
Now, considering the article's context, Stephen's quote is crucial. If I recall correctly, Stephen (maybe a character from the article) says something like, "We're creating machines that are so intelligent that they're almost like us, but they don't have the same sense of morality or self-awareness." So he's suggesting that AI and humans operate with similar intelligence systems but for different purposes. That aligns with option C. 
Alternatively, option D is about losing knowledge, which is a consequence, but the question is about how Stephen compares them, not the consequence. The answer should be the direct comparison, not the outcome. So C seems right. Let me double-check. The article might have Stephen saying that AI is becoming too intelligent, leading to a loss of human control. So their intelligence systems are similar but with different ends. Yes, that's C.
</think>

Thought process: The article highlights Stephen's view that AI and humans operate with similar intelligence systems but serve different purposes. He argues that AI's rapid advancement could lead to a loss of human agency and control. This aligns with option C, which emphasizes structural similarity in intelligence but divergent goals. While option D addresses a consequence (knowledge loss), the question focuses on direct comparison, making C the best choice.

Answer: C.
```

## Key Takeaways

- Reasoning-capable LLMs can be used to create high-quality QA datasets to teach the model to for improving a model's reading comprehension and its ability to recall and reason on specific knowledge from source documents.
- How to use the `sdg` package to generate high-quality QA pairs from a source document, add new prompts, blocks and easy create pipeline variations.
- How to use our training library to train the model on the generated data.