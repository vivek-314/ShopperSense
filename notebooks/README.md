# ShopperSense notebooks

For the final project submission, start with **[03 — Six-input models, step by step](03_six_input_models_step_by_step.ipynb)**. It contains the actual estimator, grid-search and `.fit(...)` code in separate explained cells, rather than only importing a completed training run. Executed outputs are committed to GitHub.

| Notebook | What it demonstrates |
| --- | --- |
| [01 — Data and reasoning](01_data_and_reasoning.ipynb) | Supplied data identity, quality, source row partitions and training-only exploration of the original dataset |
| [02 — Original research results](02_models_and_results.ipynb) | Inspect and reproduce the earlier full-feature research decisions and saved evidence; this is not the current six-input app model |
| [03 — Six-input model training](03_six_input_models_step_by_step.ipynb) | Actual training of Dummy, Logistic Regression, Decision Tree and Random Forest on the six current inputs; preprocessing, CV, selection, evaluation, account-field removal comparison and a fresh prediction |

Notebook 03 has 20 numbered code cells, each preceded by its purpose:

1. Imports and project location.
2. Load the real CSV and verify its checksum.
3. Define the six inputs and binary Revenue target.
4. Reproduce the exact stratified training/validation/test rows.
5. Define the production preprocessing pipeline.
6. Define the production metrics.
7. Configure training-only cross-validation.
8. Fit the majority baseline.
9. Fit and tune Logistic Regression.
10. Fit and tune Decision Tree.
11. Fit and tune Random Forest.
12. Inspect every candidate and fold score.
13. Compare validation results.
14. Define the validation-only model and threshold rules.
15. Freeze the decision before the notebook's test evaluation.
16. Compare final test results and assert agreement with production.
17. Plot and explain the confusion matrix.
18. Measure six-versus-eight input performance on the same saved rows.
19. Predict a new six-input session.
20. Verify the fitted model matches the actual browser export.

Run from the repository root after installing requirements and activating the environment:

```bash
python -m src.predictor.train
python -m nbconvert --to notebook --execute notebooks/03_six_input_models_step_by_step.ipynb --output executed_03.ipynb --ExecutePreprocessor.timeout=300
node tests/check_predictor.cjs
python -m pytest -q
```

The notebook also works when launched from `notebooks/`; its first cell resolves the import path. The raw dataset is preserved. Administrative and Administrative_Duration are excluded from the six-input predictor and all four model fits in notebook 03. Earlier eight-input results are archived under `results/predictor_previous_eight_inputs/`; earlier full-feature research remains separately labelled.

Generator: `python scripts/create_training_notebook.py` refreshes the unexecuted notebook source, copying the shared preprocessing, metric and selection functions into visible code cells. Re-execute it before replacing the committed, executed notebook. Generated notebook reproduction outputs are stored under `results/predictor/` and do not overwrite the production selection decision.

Scores are not calibrated probabilities. The historical test split is reused, not a new external validation. New-shop, future-session and early-session performance remain unvalidated. The feature-removal comparison is descriptive; it does not prove that the removed fields have no predictive value.
