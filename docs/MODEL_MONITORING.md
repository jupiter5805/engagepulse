# Model Performance Monitoring

EngagePulse models should be monitored after deployment
to ensure predictions remain reliable.

## Key Monitoring Areas

- Data drift: changes in student engagement patterns
- Model performance: ROC-AUC and average precision
- Calibration: predicted risk versus observed outcomes
- Data quality: missing values and unexpected records
- Fairness: performance differences across student groups

## Monitoring Strategy

Compare new course data against the training distribution.
Investigate significant changes before retraining.

Model retraining should require validation against a
held-out dataset before deployment.

## Objective

Maintain accurate, transparent and responsible student
engagement risk predictions over time.
