# part-3-
# Capstone Part 3 — Advanced Modeling: Ensembles, Tuning, and Full ML Pipeline

## Dataset
`cleaned_data.csv` (included — output of Part 1). This script rebuilds
Part 2's exact preprocessing and `train_test_split(..., random_state=42)`,
so `X_train_scaled`, `X_test_scaled`, `y_clf_train`, `y_clf_test` here are
identical to Part 2's.

## How to Run
```bash
pip install pandas numpy scikit-learn joblib
python3 part3_ml.py
```
Runs the full pipeline, prints every metric to console (also saved to
`part3_analysis_log.txt`), and saves the tuned model to `best_model.pkl`.

---

## Task 1 — Decision Tree Baseline (Unconstrained)
| Metric | Value |
|---|---|
| Train accuracy | 1.0000 |
| Test accuracy | 0.7333 |
| Gap | 0.2667 |

**This tree clearly overfits** — 100% training accuracy but only 73.3% on
unseen data is the textbook overfitting signature. Decision trees are
**high-variance models** because they grow greedily: at each split, the
algorithm picks the locally-best feature/threshold to separate the current
node's data and never revisits that decision later, even if a different
early split would have generalized better. Left unconstrained, the tree
keeps splitting until every leaf is pure (or has one sample), which means
it ends up memorizing noise and outliers specific to the training set
rather than learning the general pattern — small changes in the training
data would produce a very different tree.

## Task 2 — Controlled Decision Tree
`max_depth=5, min_samples_split=20`:

| Metric | Value |
|---|---|
| Train accuracy | 0.8750 |
| Test accuracy | 0.8167 |
| Gap | 0.0583 |

**`max_depth`** caps how many splits deep the tree can grow — it directly
limits model complexity, trading some bias (the tree can't capture very
fine-grained patterns) for a large reduction in variance. **`min_samples_split`**
blocks a node from splitting further unless it has at least 20 samples,
which prevents the tree from carving out splits based on tiny, noisy
subsets of just a handful of rows.

**Comparison:** the train-test gap shrank dramatically, from **0.2667**
(unconstrained) to **0.0583** (controlled) — a clear demonstration that
constraining tree growth trades a bit of training accuracy (100% → 87.5%)
for much better generalization (73.3% → 81.7% test accuracy).

## Task 3 — Gini vs Entropy
Both at `max_depth=5`:

| Criterion | Test Accuracy |
|---|---|
| Gini | 0.8167 |
| Entropy | 0.8208 |

**Gini impurity:** `Gini = 1 - Σ pᵢ²`, where pᵢ is the proportion of class
*i* in the node.
**Entropy:** `Entropy = -Σ pᵢ log₂(pᵢ)`.

Both measure how "mixed" a node's classes are, and both are minimized (=0)
when a node is perfectly pure. **A node with Gini = 0** means every sample
in that node belongs to the same class — there is no impurity left to
reduce, so a split there would provide no additional information. In this
case the two criteria produced nearly identical results (0.8167 vs
0.8208), which is typical — Gini and Entropy usually agree on which splits
are best; Entropy is slightly more computationally expensive due to the
log term but rarely changes the resulting tree much in practice.

## Task 4 — Random Forest
`n_estimators=100, max_depth=10, random_state=42`:

| Metric | Value |
|---|---|
| Train accuracy | 0.9615 |
| Test accuracy | 0.8208 |
| ROC-AUC | 0.9000 |

**Top 5 features by importance:**
| Feature | Importance |
|---|---|
| YearsExperience | 0.4992 |
| Age | 0.2225 |
| PerformanceScore | 0.0957 |
| TrainingHours | 0.0927 |
| Department_Sales | 0.0115 |

**How Random Forest computes feature importance:** for each feature, the
model averages the **reduction in Gini impurity** achieved by that feature
across every split where it was used, across all 100 trees. A feature that
consistently produces large, pure splits gets a high importance score.
This is fundamentally different from a **linear regression coefficient**:
a coefficient reflects a feature's *linear, additive* relationship with
the target holding other features constant (and its sign tells you the
direction of that relationship), while Random Forest importance only says
*how useful* a feature was for splitting decisions — it captures
non-linear and interaction effects, but tells you nothing about direction
or magnitude of impact on the prediction, and it's biased toward
high-cardinality numeric features (which is part of why YearsExperience
and Age dominate here, over the one-hot categorical dummies).

**Bagging explanation:** Random Forest builds each of its 100 trees on a
**bootstrap sample** — a random sample of the training rows drawn *with
replacement*, so each tree sees a slightly different (and overlapping)
subset of the data, some rows repeated, others left out entirely. On top
of that, at **each split**, only a random subset of √(number of features)
candidate features is considered, rather than all of them — this
decorrelates the trees further, since even similar bootstrap samples
would otherwise tend to pick the same dominant feature (YearsExperience)
at the top of every tree. Averaging predictions across many
weakly-correlated, individually high-variance trees causes their
individual errors to cancel out, producing an ensemble whose variance is
far lower than any single deep tree — this is exactly why the Random
Forest generalizes much better than the unconstrained tree from Task 1
(test accuracy 0.8208 vs 0.7333) while still reaching high training
accuracy.

## Task 4a — Gradient Boosting
`n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42`:

| Metric | Value |
|---|---|
| Train accuracy | 0.9177 |
| Test accuracy | 0.8042 |
| ROC-AUC | 0.8926 |

## Task 4b — Feature Ablation Study
5 lowest-importance features (from Task 4): `Department_IT`, `City_Delhi`,
`City_Mumbai`, `City_Chennai`, `Department_HR`.

| Model | AUC |
|---|---|
| Full model (13 features) | 0.9000 |
| Reduced model (8 features) | 0.8960 |
| Change | −0.0040 |

The AUC drop is tiny (−0.004), so **these 5 features were genuinely close
to uninformative** — removing them barely moved performance, meaning most
of what they contributed was likely noise rather than real signal (which
matches Task 4's importance ranking, where all 5 department/city dummies
scored far below YearsExperience and Age). **Implication for production:**
a simpler 8-feature model would be cheaper to serve (less data to collect
and validate at inference time, smaller model, easier long-term
maintenance) for a AUC cost of only 0.004 — well within a typical
tolerable threshold (e.g., 0.01–0.02). In this case, dropping the 5
lowest-importance features looks like a reasonable production trade-off,
though the decision should also weigh whether those City/Department
categories might become more predictive as more data comes in.

## Task 5 — Cross-Validated Comparison
5-fold `StratifiedKFold(shuffle=True, random_state=42)`, `scoring='roc_auc'`:

| Model | CV Mean AUC | CV Std AUC |
|---|---|---|
| LogisticRegression | 0.9097 | 0.0148 |
| DecisionTree (max_depth=5) | 0.8864 | 0.0256 |
| RandomForest | 0.8963 | 0.0199 |
| GradientBoosting | 0.8990 | 0.0184 |

**Why cross-validation is more reliable than a single train-test split:**
a single 80/20 split gives one estimate of generalization performance that
depends heavily on which particular rows happened to land in the test set
— an unlucky split could make a good model look mediocre, or vice versa.
5-fold cross-validation instead rotates through 5 different train/test
partitions and averages the results, so the final estimate reflects
performance across multiple different "unseen" subsets rather than just
one. The **standard deviation** across folds is itself valuable
information the single-split approach can't give you — e.g., Decision
Tree's std (0.0256) is noticeably higher than Logistic Regression's
(0.0148), confirming the tree's higher variance / sensitivity to which
rows it's trained on, exactly as discussed in Task 1.

## Task 6 — GridSearchCV Hyperparameter Tuning
Pipeline: `SimpleImputer(median) → StandardScaler → RandomForestClassifier`.

**Grid:**
```python
{
    'randomforestclassifier__n_estimators': [50, 100, 200],
    'randomforestclassifier__max_depth': [5, 10, None],
    'randomforestclassifier__min_samples_leaf': [1, 5],
}
```

**Best params:** `max_depth=None, min_samples_leaf=5, n_estimators=200`
**Best CV score (mean AUC):** 0.9029

**Total configurations evaluated:** 3 × 3 × 2 = **18 unique configurations**,
each fit **5 times** (one per CV fold) = **90 total model fits**.

**Grid Search vs Randomized Search trade-off:** Grid Search is
**exhaustive** — it tries every combination in the grid, guaranteeing the
best combination *within that grid* is found, but its cost grows
multiplicatively with the number of hyperparameters and values (adding one
more parameter with 3 options would triple the search here). **Randomized
Search** instead samples a fixed number of random combinations from the
specified distributions — it won't guarantee finding the single best
combination, but in practice it explores a much larger hyperparameter
space per unit of compute, and research has shown it often finds
near-optimal configurations faster than grid search, especially when only
a few hyperparameters actually matter much (the ones that don't matter get
"wasted" grid cells in an exhaustive search, whereas random search doesn't
waste budget on them).

## Task 7 — Manual Learning Curve
Using the best pipeline from Task 6, trained on growing subsets of
`X_train`:

| Training Fraction | Training AUC | Test AUC |
|---|---|---|
| 0.2 | 0.9770 | 0.9078 |
| 0.4 | 0.9760 | 0.9090 |
| 0.6 | 0.9723 | 0.9104 |
| 0.8 | 0.9739 | 0.9044 |
| 1.0 | 0.9684 | 0.9064 |

**(i) Does training AUC decrease as the training set grows?** Slightly —
from 0.9770 at 20% down to 0.9684 at 100%. This is the expected pattern
for a high-variance model like Random Forest: with very little data it can
nearly memorize the training rows (near-perfect training AUC), and as more
data is added it becomes progressively harder to fit every point exactly,
so training AUC edges down.

**(ii) Does test AUC increase with more data?** Not meaningfully — it
moves in a narrow band (0.904–0.910) with no clear upward trend, actually
peaking at 60% and dipping slightly by 100%.

**(iii) Conclusion — data-limited or capacity-limited?** The test AUC has
**plateaued**, not continued rising, even as training data increased
5×. This suggests the model is currently **capacity/feature-limited
rather than data-limited** — collecting more rows of the same kind of data
probably wouldn't meaningfully improve this model further. The more
promising path to improvement would be better or additional *features*
(the dataset only has a handful of predictive signals, chiefly
YearsExperience and Age), not simply more rows.

## Task 8 — Model Serialization
Best pipeline saved to **`best_model.pkl`** via `joblib.dump()` (~2.1 MB —
well under the 100MB GitHub limit, committed directly to the repo).

Reload-and-predict demo (included in `part3_ml.py`):
```python
import joblib

loaded_model = joblib.load("best_model.pkl")
sample_rows = X_test.iloc[:2]          # two hand-picked test rows
sample_preds = loaded_model.predict(sample_rows)
print("Predictions:", sample_preds.tolist())
```
Output confirmed correct: predictions `[1, 0]` matched the actual labels
`[1, 0]` for those two rows — the reload-and-predict cycle runs without
errors.

## Task 9 — Summary Comparison Table (Parts 2 + 3)

| Model | CV Mean AUC | CV Std AUC | Test AUC |
|---|---|---|---|
| LogisticRegression (Part 2) | 0.9097 | 0.0148 | 0.9034 |
| DecisionTree (max_depth=5) | 0.8864 | 0.0256 | 0.8733 |
| RandomForest | 0.8963 | 0.0199 | 0.9000 |
| GradientBoosting | 0.8990 | 0.0184 | 0.8926 |
| **RandomForest (GridSearchCV tuned)** | **0.9029** | — | **0.9064** |

**Recommendation:** I would recommend the **GridSearchCV-tuned Random
Forest** to the client. It posts the highest test-set AUC (0.9064) of any
model tested, its cross-validated mean AUC (0.9029) is competitive with
Logistic Regression's, and — critically — it comes packaged as a single
reproducible `sklearn` `Pipeline` (imputation → scaling → model) rather
than a manually-scaled array, which makes it far safer to deploy since
preprocessing can't be accidentally skipped or mismatched at inference
time. Logistic Regression is a very close second and has the advantage of
being simpler and directly interpretable via coefficients — if the client
values interpretability over the last percentage point of AUC, it would
be a reasonable alternative recommendation. The plain Decision Tree is not
recommended given its high variance and comparatively poor CV performance.

---

### Files in this deliverable
- `part3_ml.py` — full, commented Python script (all ensembles, tuning, CV, pipeline, serialization)
- `cleaned_data.csv` — input dataset (from Part 1)
- `best_model.pkl` — serialized, tuned Random Forest pipeline (GridSearchCV winner)
- `part3_analysis_log.txt` — full console output of every printed statistic
- `README.md` — this file
