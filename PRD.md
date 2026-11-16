# ShopperSense — Implementation PRD

Version 2.0 | 16 November 2026
Project: Predicting E-Commerce Conversions Using Customer Browsing Behaviour
Deliverables: reproducible ML repository and a complete research website deployable on GitHub Pages.

## 1. Goal and scope

Build a modest undergraduate ML project whose strength is clear reasoning and correct evaluation. The website must explain the investigation from question to conclusion. The repository must let a grader reproduce and inspect every reported result.

This revision supersedes version 1.0's mandatory seven-model comparison, advanced boosting, live predictor, and calibration workflow. The current instruction to use basic models and avoid unnecessary complexity governs this revision. No grading weights, course deadline, or proposal details absent from the supplied material should be invented.

Use the already supplied dataset. Inspect existing project files before making changes and preserve working code where possible. Never replace real data with synthetic data or invent performance values. This is a specification; no models have been trained as part of preparing this document.

### Core scope

- One binary classification question.
- A majority-class baseline and three standard classifiers.
- A justified preprocessing pipeline and a consistent evaluation protocol.
- A small set of informative figures, error analysis, and one feature-availability experiment.
- A readable static website with a meaningful threshold explorer.
- Code, notebooks, saved splits, complete experiments, and a clear README.

Do not add login, databases, chatbots, neural networks, extensive hyperparameter searches, product recommendation systems, or commercial dashboards. XGBoost, AdaBoost, and Gradient Boosting are outside this simplified core. Do not claim customer price/quality preferences: those labels and product attributes are absent from the dataset.

## 2. Research question and claims

Question: How well can recorded browsing-session characteristics identify sessions that end in a purchase, compared with always predicting no purchase?

Secondary questions:
1. Does a tree or forest improve meaningfully over Logistic Regression?
2. Which mistakes remain, especially missed purchases?
3. How does the classification threshold change precision and recall?
4. How sensitive are findings to excluding PageValues?

One row represents one recorded browsing session. Revenue is the binary target: 1 means purchase, 0 means no purchase. The analysis uses session summaries; it does not establish early-session predictive performance. Some measurements may only be available after browsing progresses or ends.

Findings are associations within this historical dataset. They do not prove what causes purchases, demonstrate conversion uplift, or establish performance on a different shopping website. Use the website conclusion to answer the actual research question with measured evidence.

## 3. Instructor requirements and acceptance evidence

These are requirements supplied by the instructor, not an invented points rubric.

| Requirement | Website evidence | Repository evidence |
| --- | --- | --- |
| Question and row meaning | Opening question, target, unit, prediction timing | README and notebook introduction |
| Source, size, variables, quality | Data card, variable table, measured quality summary | Data manifest, loader, quality report |
| Preprocessing and why | Table pairing each transformation with rationale | Pipeline code and fitted configuration |
| Baseline and compared models | Methods section explaining each model | Estimator definitions and parameter grids |
| Split and metrics | Explicit split diagram/table and metric rationale | Exact row IDs, seed, search results |
| Results and error analysis | Comparison table, PR curve, confusion matrices, mistakes | Full metrics and row-level predictions |
| Limitations | Dedicated visible discussion | README and analysis notebook |
| Useful interaction | Validation threshold explorer | Exported validation predictions and JS logic |
| Reproduction and integrity | Repository link, sources, AI-use note | Runnable code/notebooks, README, citations, disclosure |

The site must contain the explanation itself. Linking to a notebook does not substitute for explaining a required point on the website.

## 4. Data contract and inspection

Source: UCI Online Shoppers Purchasing Intention Dataset, dataset 468.
Source URL: https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset
DOI: https://doi.org/10.24432/C5F88Q

UCI describes 12,330 sessions, 17 predictors and Revenue as target, with 1,908 purchase and 10,422 non-purchase sessions and no missing values. These are source expectations, not results of inspecting the supplied file. Verify the actual file before publishing its counts. Record dataset attribution and the license from the source.

### Feature groups

| Type | Fields | Treatment |
| --- | --- | --- |
| Counts and durations | Administrative, Administrative_Duration, Informational, Informational_Duration, ProductRelated, ProductRelated_Duration | Numeric |
| Analytics and event measures | BounceRates, ExitRates, PageValues, SpecialDay | Numeric; explain definitions and availability |
| Categories | Month, OperatingSystems, Browser, Region, TrafficType, VisitorType | Categorical, including numeric-coded categories |
| Boolean | Weekend | Consistent binary encoding |
| Target | Revenue | Convert explicitly to 0/1; exclude from inputs |

Required inspection:
- Record SHA-256 checksum, source filename, row count, column names, data types, class counts, and source access information.
- Count missing, nonfinite, invalid, and unusual values; investigate unexpected negatives and categorical values.
- Count complete duplicate rows and identical predictor vectors. Repeated sessions are not automatically erroneous. Preserve rows by default and disclose the decision.
- Inspect skew and extreme values in counts/durations. Do not drop outliers merely because they are large.
- Preserve raw bytes; assign stable source_row_id from original row order before transformations.
- Show actual quality findings, including when no issues are found. If supplied data differ from the expected source, document the difference rather than silently modifying them.

EDA should use training data for exploratory choices; report whole-file structural counts separately. Required EDA: class distribution; training-set purchase rates by VisitorType with denominators; one browsing-intensity distribution by target. A month comparison is optional. Label which partition each chart uses. Explain each figure in one or two sentences.

## 5. Split and preprocessing

Create one stratified 70% training / 15% validation / 15% test partition using random_state=42. First hold out 15% test; split the remaining data using validation fraction 15/85. Record actual integer counts produced by the library. Persist source_row_id arrays and verify complete, disjoint coverage. Reuse these IDs across all models and feature experiments.

All data-dependent preprocessing must be fitted within training folds using sklearn Pipeline and ColumnTransformer. Never fit imputation, category vocabulary, or scaling on the complete dataset.

| Choice | Implementation | Reason to explain on website |
| --- | --- | --- |
| Numeric missing values | Training median if needed | Retains rows without using holdout information |
| Categorical missing values | Fixed missing category if needed | Makes missingness explicit |
| Categorical encoding | OneHotEncoder(handle_unknown='ignore') | Codes do not imply meaningful ordered distances |
| Numeric scaling | StandardScaler for Logistic Regression | Features have different units; scaling supports regularization and fitting |
| Tree numeric features | Keep original scale | Threshold-based trees do not need standardization |
| Imbalance | Stratification and purchase-focused metrics | Majority-class accuracy can obscure missed buyers |

Do not add SMOTE or automatic outlier removal. The core has no resampling or class weighting; threshold exploration addresses the precision/recall tradeoff. If a later optional experiment uses weighting, record it separately.

Audit identical predictor vectors across partitions. Quantify overlap. If overlap could materially inflate the result, add a secondary group-aware split that keeps identical vectors together; report it separately and explain the tradeoff. Never label this sensitivity analysis as the original partition.

PageValues is an analytics measure whose availability at prediction time needs scrutiny. Do not assert that it definitely leaks the target without evidence. Compare full-feature and no-PageValues results using the same rows to show dependence on this feature.

## 6. Models and bounded tuning

| Model | Role | Small training-only search |
| --- | --- | --- |
| DummyClassifier(strategy='most_frequent') | Establish majority baseline | None |
| LogisticRegression(max_iter=2000) | Simple linear comparison | C in [0.1, 1, 10] |
| DecisionTreeClassifier | Interpretable nonlinear rules; overfit comparison | max_depth in [3, 5, 8], min_samples_leaf in [5, 20] |
| RandomForestClassifier(n_estimators=150) | Standard bagging comparison | max_depth in [8, None], min_samples_leaf in [2, 10] |

Set random_state=42 wherever supported. Use 3-fold StratifiedKFold(shuffle=True, random_state=42) on training only, and GridSearchCV(scoring='average_precision', refit=True). Save every candidate and fold score. Record convergence warnings and resolve them before final reporting.

Explain each model's mechanism, why it is included, and one weakness. Explain that a forest averages multiple randomized trees to reduce variance; it is included as a standard comparator. Do not present all models as equally interpretable.

Do not add algorithms merely to lengthen the comparison. The selected model need not be the most complex one.

## 7. Evaluation and decision protocol

Primary ranking metric: average precision (AP) for the purchase class. Label AP precisely; do not silently equate it with trapezoidal PR-AUC. AP assesses ranking across thresholds and focuses attention on the minority class.

Report purchase-class precision, recall, F1, AP, ROC-AUC and accuracy. Display class support and confusion matrices. Define all metrics briefly. Explain that accuracy alone rewards the dominant non-purchase class. Handle zero predicted positives with zero_division=0 and disclose undefined metrics where appropriate.

Procedure:
1. Tune parameters on training-only cross-validation.
2. Evaluate each tuned model on validation at threshold 0.5. Save probabilities and metrics.
3. Select the highest validation AP. If AP differs by at most 0.005, prefer the simpler model in order Logistic Regression, Decision Tree, Random Forest. This is a declared practical selection rule, not a significance test. Keep Dummy as reference.
4. For the selected model choose the validation threshold maximizing purchase-class F1 on the fixed grid 0.05, 0.06, ..., 0.95. Break ties by choosing the larger threshold. This is an illustrative balance rule; no business cost ratio is known.
5. Freeze parameters, feature set, selected model and threshold in a decision record before test evaluation. Keep the fitted training-only models; do not silently refit on validation after choosing thresholds.
6. Evaluate all frozen models on the held-out test for the final comparison at 0.5. Additionally report the selected model at its validation-selected threshold. No subsequent model selection or tuning from test results.

The website must explicitly distinguish validation selection results from final test results. If the test ranking differs, report that honestly without changing the selected model. Re-running a deterministic pipeline for reproduction is allowed; revising decisions in response to test outcomes invalidates the untouched-holdout claim.

Required figures/tables: all-model metric table; PR curves for learned models; selected-model confusion matrix at 0.5 and selected threshold; train versus validation F1 for a small Decision Tree depth comparison. Use fixed thresholds when interpreting the overfitting comparison.

Do not claim probabilities are calibrated. The required interaction can label predict_proba outputs as model scores. No calibration project is required in this core.

## 8. Error analysis and one extra experiment

For the selected model at the frozen threshold:
- Explain false positives as predicted buyers who did not purchase and false negatives as missed actual buyers.
- Show test counts and relevant rates with denominators, not just a percentage accuracy.
- Show a small deterministic selection of FP/FN examples, source row IDs, key observed features, true label, score and predicted label. If an error category is empty, say so.
- Compare errors by VisitorType and low/medium/high ProductRelated bins. Define numeric bin boundaries using training quantiles and persist them. Show subgroup support; avoid strong claims about small groups.
- Distinguish measured patterns from hypotheses about why the model failed. Data do not reveal the visitor's motives.

Required sensitivity experiment: retrain Logistic Regression and Random Forest without PageValues. Use the same partitions and bounded training CV protocol. Assess validation dependence before freezing decisions; full features remain the primary research condition, not a feature set chosen using test results. Freeze both variants and report their final test metrics as a prespecified secondary comparison.

Show the with/without PageValues change on the website. Explain that removing one questionable feature does not prove all remaining features were available in real time.

Optional interpretation: validation permutation importance for selected original columns, with repeat variation, or scaled Logistic Regression coefficients with careful category-reference explanations. Do not require SHAP, causal explanations, or extra models.

## 9. Website requirements

Build one responsive research report with anchor navigation, readable typography, restrained colors, clear chart captions, and visible repository access. Prefer HTML/CSS/vanilla JS, with Python-generated chart assets and JSON. GitHub Pages serves the static website; Python runs offline to generate evidence.

Required reading order:
1. Question and brief answer: scope, row meaning, target, measured takeaway once results exist.
2. Data: source link, verified size, feature groups, prevalence and quality findings.
3. Preprocessing: choices and rationale, including categorical codes, scaling, split isolation and PageValues.
4. Methods: baseline and three classifiers with concise explanations.
5. Evaluation: split counts, seed, CV versus validation versus test, metrics and selection policy.
6. Results: actual comparison table and key figures with interpretation and model choice.
7. Mistakes and sensitivity: errors, subgroup supports, PageValues experiment.
8. Explore the threshold: required interactive lab.
9. Conclusions and limitations: direct answer supported by evidence and boundaries of inference.
10. Reproduce and credits: repository, data, references, accurate AI-use note and actual contributors.

Every result claim must identify the relevant model, feature set and partition. Display a generated run identifier and data hash prefix. Put concise summaries and the key figures on the site; link full experiments and outputs in the repository. Do not hide essential analysis in downloads, tooltips, or collapsed code panels.

### Required interaction: threshold explorer

Use the frozen selected model's actual validation y_true and purchase scores exported to JSON. Clearly label the panel Validation exploration; distinguish it from final test results.

Controls: threshold slider 0.00–1.00 in steps of 0.01; reset to 0.50; mark the chosen validation threshold. Prediction rule: score >= threshold means purchase.

Update predicted purchase count, TP/FP/TN/FN, precision, recall and F1 immediately. Show the number of validation sessions and true purchases. Explain how lowering the threshold usually increases both detected buyers and false alarms; the observed counts demonstrate the tradeoff. Do not imply that changing the slider retrains the model or updates the official test result.

Provide numeric outputs and text equivalents for charts, keyboard-operable controls, clear focus states and a small-screen layout without horizontal overflow. Error states must explain missing data instead of displaying fake scores.

A live prediction form is optional only after all core work passes. It is unnecessary to satisfy the instructor's interaction requirement. If implemented, it must use a real exported model and its exact preprocessing, with Python/JS parity verification and honest input-availability limits. No made-up percentages or hidden default features.

## 10. Repository structure and artifact contract

Suggested structure; files may be combined if responsibilities stay clear:

```text
README.md
PRD.md
requirements.txt
.gitignore
.github/workflows/pages.yml
data/README.md
data/raw/online_shoppers_intention.csv
notebooks/01_data_and_reasoning.ipynb
notebooks/02_models_and_results.ipynb
src/__init__.py
src/config.py
src/data.py
src/preprocessing.py
src/train.py
src/evaluate.py
src/report.py
src/run_all.py
tests/
results/data_manifest.json
results/quality.json
results/splits.json
results/cv_results.csv
results/parameters.json
results/selection.json
results/validation_metrics.csv
results/test_metrics.csv
results/validation_predictions.csv
results/test_predictions.csv
results/pagevalues_comparison.csv
results/error_analysis.json
results/figures/
docs/index.html
docs/styles.css
docs/app.js
docs/.nojekyll
docs/assets/
docs/data/summary.json
docs/data/threshold_validation.json
```

Data manifest: filename, checksum, attribution/source/license, actual shape/schema, target mapping, environment versions and access date where known. For a supplied file whose download date is unknown, say supplied locally rather than inventing a retrieval date.

Splits: seed, method, source checksum, train/validation/test row IDs and actual class counts.

Predictions: source_row_id, partition, model, feature_set, y_true, purchase_score, threshold, predicted_label. Persist unrounded numeric values. Keep displayed rounding separate from computation.

Metrics: model, feature_set, partition, threshold, support, precision, recall, F1, AP, ROC-AUC, accuracy and confusion counts. Keep CV, validation and test outputs distinct.

Selection record: selected model/parameters/features, validation selection evidence, threshold rule and chosen threshold. Include code/run identifier; create before producing test scores.

Website JSON and figures must be generated from these artifacts, not manually typed metric values. Human-written result interpretation must be checked against the same artifacts. Notebooks should import shared source functions and show the reasoning; they must execute top to bottom without hidden cell state.

## 11. Setup, reproduction and deployment

Use a compatible Python version and record the exact tested dependencies after the implementation runs. Main packages: numpy, pandas, scikit-learn, matplotlib, jupyter and pytest. Avoid unnecessary frameworks and external runtime APIs.

Required documented commands, implemented exactly or replaced consistently:

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m src.run_all --data data/raw/online_shoppers_intention.csv
python -m pytest -q
jupyter nbconvert --to notebook --execute notebooks/01_data_and_reasoning.ipynb --output executed_01.ipynb
jupyter nbconvert --to notebook --execute notebooks/02_models_and_results.ipynb --output executed_02.ipynb
python -m http.server 8000 --directory docs
```

README must state the working directory and any notebook import-path setup. It must explain obtaining the canonical CSV if raw data are uncommitted, then verifying the expected checksum. Include Linux/macOS and Windows environment activation instructions. Document reproducibility tolerance and measured runtime only after an actual run.

The localhost address works on the machine running the server. It is a preview, not a public submission URL. The deployed site must load all summaries and interactions without a Python server.

Publish docs/ through GitHub Pages using a workflow or branch/folder publishing. Use relative asset URLs so repository subpaths work. Include deployment steps and a repository URL in the site configuration; do not invent repository names or URLs. Push/publish only with the user's authorization and available access. Otherwise deliver a ready-to-push repository and exact remaining steps.

README must contain question/scope, website and repo links, dataset access and license, prerequisites, run/test/notebook commands, exact evaluation procedure, outputs guide, limitations, actual contributions, citations and AI-use disclosure. Disclosure must reflect assistance actually used and human verification actually performed; do not invent review or contributions.

## 12. Verification and delivery gates

Meaningful automated checks:
- Schema and binary target mapping are correct; target and row IDs never become predictors.
- Split row IDs are stable, disjoint, complete and associated with the recorded data hash.
- Pipeline transforms learn only on intended training rows/folds; test outcomes cannot enter selection or threshold code.
- Probability/score arrays are finite, ordered with their row IDs and within [0,1].
- Metrics/confusion counts are correct on a small hand-checkable fixture; zero predicted positives is handled.
- Exported website values agree with result CSV/JSON, and all mandatory variants have artifacts.
- Slider calculations agree with Python confusion counts at 0, 0.5, 1 and selected threshold, including equality at the threshold.

Verification must include actual notebook execution, a full pipeline run, static serving, website controls, mobile/desktop layout, relative URLs and repository links. Record which checks ran and which could not run. Do not describe unexecuted checks as passed.

Scientific review: a grader should answer every instructor requirement from the website alone, understand why the baseline is weak despite accuracy, identify the frozen selection procedure, see real mistakes, and understand the limits of the findings.

Delivery order:
1. Inspect existing work, validate data and save splits.
2. Run baseline, model search and validation decisions.
3. Run prespecified feature experiment, freeze decisions, evaluate test and generate artifacts.
4. Author the complete website narrative using measured results.
5. Add and verify the threshold explorer.
6. Reproduce notebooks and pipeline, complete README/disclosure, validate deployment.

Completion checklist:
- [ ] Every instructor requirement appears visibly on the site.
- [ ] Baseline and three basic models have real, reproducible results.
- [ ] Preprocessing, splits, tuning, selection and metrics are justified.
- [ ] Errors, PageValues sensitivity and limitations are explained.
- [ ] One meaningful interaction works with actual validation outputs.
- [ ] Repository contains source, executable notebooks, exact splits, all experiments and full results.
- [ ] README contains setup/run, data access, sources and accurate AI disclosure.
- [ ] Website numbers agree with reproducible artifacts.
- [ ] Published links are checked, or deployment is explicitly reported as pending.

## 13. Instructions to the implementing Codex agent

Implement the core end to end using this PRD. Preserve existing user work. Make routine engineering decisions without repeated clarification. Do not spend time on visual luxury before completing scientific evidence and the report. Do not add advanced models to chase scores. If blocked, describe the specific missing dependency/access and deliver the rest honestly.

Final handoff must state: files and commands; verified dataset identity; models and experiments actually run; selected model, metric and test results; interaction checks; deployment status and real links; remaining limitations. Never claim results or publication before observing them.

## 14. Technical references

Verified reference pages for the specification on 16 November 2026; implementing authors must cite additional sources actually used:
- UCI dataset: https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset
- Scikit-learn metric definitions: https://scikit-learn.org/stable/modules/model_evaluation.html
- GitHub Pages static hosting: https://docs.github.com/en/pages/getting-started-with-github-pages/what-is-github-pages

The instructor message supplied in this conversation is authoritative for submission coverage. This PRD sets implementation choices; it does not assert additional grading conditions.
