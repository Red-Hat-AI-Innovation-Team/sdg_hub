### Construct ICL examples
Here is script for generating seed examples/ICLs using the nemotron super model.
The reason we would like to use the super model is because in our seed examples we want to show similar reasoning as the super model
Seed examples/ICL: These are the examples the teacher model uses when generating synthetic data as reference of simiilarity i.e. it will generate more QA like this. 

Steps:
- We will use nemotron super 49B model to generated some ICL examples
- We will use the generated ICLs to generate more examples from the quality corpus
- You can customize these prompts according to your downstream task
- Here we aim to create QA aimed at understanding not only specific details about articles but also to create a comprehensive understanding of the article
Note: This is not part of the standard flow and is only meant as one way of creating the seed examples. Users can also hand-craft these, use a prorietary model or use combination of models.


Step 1: We write a simple generic function to get response from a vllm deployed endpoint using a OpenAI endpoint.
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

Step 2: Load the Quality Corpus. 
```python
ds = load_dataset("zitongyang/entigraph-quality-corpus", split='train').remove_columns(['entity', 'entigraph']).rename_columns({'raw': 'document', 'uid': 'document_outline'})
```

Step 3: We will use series of prompts that describe an example document and types of Question-Answer we want. The Nemotron Super model will generate answer with thinking trace/block. We can also use the same model to check the output of its own response, put the output in specific format (like JSON) etc. Below prompts show different examples of describing the examples passage we would like (short passages of varying length), types of Question-Answers (GRE-Style, Open-ended etc.)
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