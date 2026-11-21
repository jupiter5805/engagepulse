# Preventing Data Leakage

EngagePulse predicts student academic risk using information
available during the first 28 days of a course.

## Validation Rules

- Exclude learning activity recorded after day 27.
- Exclude outcomes unavailable at prediction time.
- Keep training and test observations separate.
- Avoid using future assessment results as model features.
- Evaluate performance on unseen course presentations.

## Why It Matters

Data leakage can produce misleadingly high model scores.

Reliable student-risk predictions require realistic evaluation
using only information available when predictions are made.
