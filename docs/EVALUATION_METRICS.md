# Model Evaluation Metrics

EngagePulse uses multiple metrics to evaluate student-risk predictions.

| Metric | Purpose |
|--------|---------|
| ROC-AUC | Measures the ability to distinguish at-risk students |
| Average Precision | Evaluates prediction ranking for positive outcomes |
| Brier Score | Measures probability prediction error |
| Precision at Top 10% | Measures accuracy within the highest-risk group |
| Recall at Top 10% | Measures how many at-risk cases that group captures |

## Business Relevance

Universities have limited resources for student interventions.
Precision and recall help determine how effectively these
resources could be allocated.

Model selection should balance predictive performance,
probability reliability and practical intervention capacity.
