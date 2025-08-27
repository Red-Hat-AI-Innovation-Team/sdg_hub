### 1\. Document Summarization

To kickstart the process, we generate three unique summaries of your source documents. This multi-faceted approach helps the model to thoroughly memorize and recall the key information. The summaries include:

  * **Detailed Summaries:** Comprehensive overviews of the content.
  * **Extractive Summaries:** Key sentences and passages pulled directly from the text.
  * **Atomic Facts:** A list of the most critical, standalone pieces of information.


### 2\. Synthetic Q\&A Generation

Next, our pipeline leverages user-provided "seed examples"—sample questions and answers—to generate a wealth of synthetic Q\&A pairs. These new pairs are contextually grounded in the summarized documents, effectively scaling up your initial examples into a diverse training dataset.


### 3\. Quality Control

To ensure the integrity of our generated data, we employ a quality-checking phase. Using a "teacher" model, we perform a faithfulness evaluation by:

1.  Providing the model with a generated answer and the original source document.
2.  Tasking the model to extract every claim made in the answer.
3.  Verifying that each claim is factually supported by the provided document.

This process filters out inaccuracies and ensures that only high-quality, faithful Q\&A pairs make it into the final dataset.


### Data Generation Statistics

#### Quality
|   Cut/n=3 |   Token Count |
|-------|---------------|
|     1 |       2,193,502 |
|     2 |       4,383,655 |
|     5 |      10,870,396 |
|    10 |      21,815,170 |
|    20 |      43,601,976 |
|    30 |      65,395,710 |
|    40 |      87,118,308 |
|    50 |     108,779,213 |


#### Finance Bench
| Cut/n=1   | Token Count   |
|-------|---------------|
|    50 |     213,333,192 |
