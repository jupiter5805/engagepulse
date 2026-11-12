# Explainability: Phase 3, first milestone

This experiment measures **global permutation importance**. For each
feature, it shuffles values in a held-out dataset and measures the change
in **average precision** (AP). A larger positive AP drop indicates greater
model reliance on that feature *within the evaluation population*.

Run the pretrained baseline explanation:

```bash
python -m engagepulse.explain --data-dir data/raw/oulad
```

If you have already trained Phase 2's temporally selected model, use its
validation presentation for explanations (not the final test presentation):

```bash
python -m engagepulse.explain --data-dir data/raw/oulad --mode temporal
```

Report: `outputs/feature_importance.json` (gitignored along with raw data).

**Interpretation cautions:** Feature importance is *not causal*. Correlated
activity features can share or mask importance. The target is a later
"Fail or Withdrawn" course outcome, not a diagnosis of student motivation.
Results should not be used to make automatic decisions about students.
The baseline explanation uses the already-evaluated random student split;
only the temporal mode uses a later presentation for explanations.
