# Continued Pre-training (CPT) Results

We performed continued pre-training (CPT) using next-token prediction on augmented documents, without applying any chat template for the model input. To improve generalization and mitigate overfitting, we incorporated **RedPajama v2** data as a replay buffer, constituting 10% of the total input tokens.

<table>
  <caption><b>Table 2: CPT data scaling and resulting model accuracy. Higher augmentation "cuts" correspond to increased training data and performance.</b></caption>

| Cut (NUMBER\_OF\_SUMMARIES) | Token Count  | Accuracy (%) | Method     |
|-----------------------------|--------------|--------------|------------|
| Input Corpus                | 1,517,465    | 43.67        | Baseline   |
| 50                          | 373,183,414  | 51.64        | SFT        |
| 25                          | 42,904,412   | 56.77        | CPT        |
| 50                          | 83,750,884   | 57.49        | CPT        |
</table>

Notes:
- CPT shows signs of overfitting at higher token count (number of summaries) on knowledge data.
- We use red pajama mix to prevent some of this overfitting.
