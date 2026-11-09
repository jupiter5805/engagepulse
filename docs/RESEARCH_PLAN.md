# Research-to-model plan

## Research question

Can changes in *educationally purposeful* digital engagement during the first four weeks improve early identification of students who may later withdraw or fail, compared with a simple, transparent baseline?

## Operational hypotheses (not established causal claims)

| Behavioural construct | Observable proxy | Confounders / limitations |
|---|---|---|
| Consistency of engagement | Number of days active | Course design; offline study |
| Sustained self-regulated learning | Week 4 vs week 1 clicks | Different workload patterns; holidays |
| Withdrawal/disengagement signal | Days since last VLE activity | Mobile/offline resource usage |
| Breadth of study activity | Distinct VLE resources used | Larger courses have more resources |
| Assessment engagement (Phase 2) | On-time submission and scored work available by cutoff | Assessment schedules vary by module |

## Core validation protocol

1. Establish a prevalence, majority-class and simple Logistic Regression baseline.
2. Add theory-led behavioural features in documented ablation experiments.
3. Compare Logistic Regression, Random Forest and gradient boosting.
4. Move from grouped random split to **future course presentation** holdouts and cross-module transfer.
5. Reject all features unavailable at the actual decision cutoff; check early withdrawals.
6. Evaluate average precision, recall and precision under a fixed tutor-support capacity, ROC-AUC, calibration/Brier and false positive burden.
7. Audit group differences *separately* from model inputs; provide clear, non-stigmatising explanations.
8. Validate usefulness by a hypothetical tutor workflow and interviews with volunteer educators if ethically feasible. Do not claim causal benefits without an intervention study.

## Ethical constraints

No live student data. No automatic punitive decisions. Human-in-the-loop support, transparency, access control and a documented appeals/correction process if ever deployed. Guidance: https://www.jisc.ac.uk/guides/code-of-practice-for-learning-analytics

## Competitor review (public sources only)

- Kortext stream: https://kortext.com/stream/features/
- Jisc Learning Analytics: https://www.jisc.ac.uk/learning-analytics

Compare public descriptions of data sources, intervention workflow, student transparency, explanation, fairness and validation; do not imply access to proprietary product internals.
