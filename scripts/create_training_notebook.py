"""Generate a submission notebook with the actual fit/search code for each model."""
from pathlib import Path
import inspect
import sys

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.predictor import train as product
from src.evaluate import metrics, choose_model, choose_threshold

cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text.strip()))


def code(text):
    cells.append(nbf.v4.new_code_cell(text.strip()))


md('''# ShopperSense · 03 / Six-input models, cell by cell

This is the **actual training code**, separated into explained cells for the final submission. It fits the majority baseline, Logistic Regression, Decision Tree and Random Forest on the supplied dataset; performs the production parameter searches; selects using validation only; and reproduces the deployed six-input model's metrics.

**Current features:** ProductRelated, ProductRelated_Duration, Informational, Informational_Duration, VisitorType and Weekend. Administrative / Administrative_Duration were removed from the practical predictor at the user's request. The original CSV and earlier 17-feature research are preserved.

**Question:** Can these six recorded browsing-session features help identify purchases? One row is a session; Revenue is the known historical outcome. This is not validated early-session or cross-store forecasting. The historical test partition is reused from the earlier project, not a new external validation.

Run from this repository's root or `notebooks/`, with the environment in README installed. Run `python -m src.predictor.train` first to generate the production artifacts this notebook checks. No hidden cell state is required. The source generator copies the production preprocessing, metric and selection functions into visible code cells; the model cells below explicitly show each estimator, parameter grid and `.fit(...)` call.''')
md('## Cell 1 — Imports and project location\nThese are standard tabular classification libraries. The root-path setup makes shared project definitions accessible when Jupyter starts inside `notebooks/`.')
code('''from pathlib import Path
import sys, json, hashlib, warnings
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)
ROOT = Path.cwd().resolve()
if ROOT.name == 'notebooks':
    ROOT = ROOT.parent
assert (ROOT / 'src').is_dir(), 'Start from the project root or notebooks/'
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.config import MODEL_ORDER
from src.predictor import train as production
SEED = 42
OUTPUT = ROOT / 'results' / 'predictor'
def saved(name):
    return json.loads((OUTPUT / name).read_text(encoding='utf-8'))
''')
md('## Cell 2 — Read the supplied real CSV\nThe checksum identifies the exact raw file. We do not create training data, delete rows or replace missing outcomes. UCI attribution: Sakar & Kastro (2018), DOI 10.24432/C5F88Q, CC BY 4.0.')
code('''data_path = ROOT / 'data/raw/online_shoppers_intention.csv'
raw = pd.read_csv(data_path)
data_hash = hashlib.sha256(data_path.read_bytes()).hexdigest()
manifest = json.loads((ROOT / 'results/data_manifest.json').read_text())
assert data_hash == manifest['sha256']
print('Rows and columns:', raw.shape)
print('SHA-256:', data_hash)
display(raw.head())''')
md('## Cell 3 — Explicit inputs and target\nOnly these six fields are passed to the pipelines. We do not merely hide account fields in the form: they are absent from X. Revenue and row IDs also never enter X. VisitorType is categorical; Weekend is explicit 0/1. The raw 18-column source stays unchanged.')
code('''NUMERIC = ['ProductRelated', 'ProductRelated_Duration', 'Informational',
           'Informational_Duration', 'Weekend']
CATEGORICAL = ['VisitorType']
FEATURES = NUMERIC + CATEGORICAL
assert FEATURES == production.FEATURES
assert not {'Administrative','Administrative_Duration','PageValues','Revenue','source_row_id'} & set(FEATURES)
X = raw[FEATURES].copy()
X['VisitorType'] = X['VisitorType'].astype(object)
boolean_map = {'true': 1, 'false': 0, '1': 1, '0': 0}
X['Weekend'] = X['Weekend'].astype(str).str.lower().map(boolean_map)
y = raw['Revenue'].astype(str).str.lower().map(boolean_map)
assert not X.isna().any().any() and not y.isna().any()
y = y.astype(int)
X['Weekend'] = X['Weekend'].astype(int)
display(pd.DataFrame({'source_column': FEATURES,
                      'meaning': ['Product page count','Product page seconds','Information page count',
                                  'Information page seconds','Weekend flag','Visitor category']}))
display(y.value_counts().rename(index={0:'No purchase',1:'Purchase'}))''')
md('## Cell 4 — Reproduce and verify the saved stratified splits\nFirst reserve 15% test, then take 15/85 of the remainder for validation, both stratified with seed 42. Assert the exact source row order matches the production split. The same rows make the six-versus-eight comparison interpretable.')
code('''remaining_ids, test_ids = train_test_split(X.index.to_numpy(), test_size=0.15,
                                           stratify=y, random_state=SEED)
train_ids, validation_ids = train_test_split(remaining_ids, test_size=15/85,
                                           stratify=y.loc[remaining_ids], random_state=SEED)
split = json.loads((ROOT / 'results/splits.json').read_text())
assert split['source_sha256'] == data_hash
parts = {'train': train_ids, 'validation': validation_ids, 'test': test_ids}
for name, ids in parts.items():
    assert ids.tolist() == split['partitions'][name]['source_row_ids']
all_ids = np.concatenate(list(parts.values()))
assert len(all_ids) == len(set(all_ids)) == len(raw)
X_train, y_train = X.loc[train_ids], y.loc[train_ids]
X_validation, y_validation = X.loc[validation_ids], y.loc[validation_ids]
X_test, y_test = X.loc[test_ids], y.loc[test_ids]
display(pd.DataFrame([{'partition':name,'sessions':len(ids),'purchases':int(y.loc[ids].sum())}
                      for name,ids in parts.items()]))''')
md('## Cell 5 — The production preprocessing code\nThis function is copied from `src/predictor/train.py`. Median imputation and categorical vocabulary are learned inside each training fold. StandardScaler is used only for Logistic Regression. The tree models retain numeric scale. No class weights or resampling are used.')
code(inspect.getsource(product.preprocess))
md('## Cell 6 — The production metric function\nAP ranks purchase scores across thresholds; it is not trapezoidal PR-AUC. Precision is TP/(TP+FP), recall TP/(TP+FN), and F1 balances them. Undefined precision with no predicted buyers is displayed as 0 by the declared convention.')
code(inspect.getsource(metrics))
md('## Cell 7 — Common training-only cross-validation\nEvery learned classifier uses the same three shuffled stratified folds and AP objective. An unresolved convergence warning stops execution. Model and threshold choices later use validation, never test outcomes.')
code('''warnings.filterwarnings('error', category=ConvergenceWarning)
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
models, searches = {}, {}
search_options = dict(scoring='average_precision', cv=cv, refit=True,
                      n_jobs=1, return_train_score=True, error_score='raise')''')
md('## Cell 8 — Majority baseline: actual fit code\nAlways predict no purchase. This exposes why accuracy can look high while buyer recall is zero. There is no hyperparameter search for this reference.')
code('''dummy_pipeline = Pipeline([
    ('preprocess', preprocess()),
    ('model', DummyClassifier(strategy='most_frequent', random_state=SEED))
])
dummy_pipeline.fit(X_train, y_train)
models['Dummy'] = dummy_pipeline
print('Baseline fitted only on training sessions.')''')
md('## Cell 9 — Logistic Regression: actual fit and search code\nA regularized linear log-odds model. Scaling makes regularization comparable across units. C=[0.1,1,10] is a deliberately small search; the best candidate is refitted on all training rows, with no validation refit.')
code('''logistic_pipeline = Pipeline([
    ('preprocess', preprocess(scale=True)),
    ('model', LogisticRegression(max_iter=2000, random_state=SEED))
])
logistic_search = GridSearchCV(logistic_pipeline, {'model__C':[0.1,1,10]}, **search_options)
logistic_search.fit(X_train, y_train)
models['Logistic Regression'] = logistic_search.best_estimator_
searches['Logistic Regression'] = logistic_search
print('Selected training-CV parameters:', logistic_search.best_params_)
display(pd.DataFrame(logistic_search.cv_results_)[['params','mean_test_score','std_test_score']])''')
md('## Cell 10 — Decision Tree: actual fit and search code\nA tree learns nonlinear feature-threshold rules. Limiting depth and requiring larger leaves reduce overfitting. Six candidates cover depths 3,5,8 and minimum leaf sizes 5,20. Trees do not require numeric scaling.')
code('''tree_pipeline = Pipeline([
    ('preprocess', preprocess()),
    ('model', DecisionTreeClassifier(random_state=SEED))
])
tree_grid = {'model__max_depth':[3,5,8], 'model__min_samples_leaf':[5,20]}
tree_search = GridSearchCV(tree_pipeline, tree_grid, **search_options)
tree_search.fit(X_train, y_train)
models['Decision Tree'] = tree_search.best_estimator_
searches['Decision Tree'] = tree_search
print('Selected training-CV parameters:', tree_search.best_params_)
display(pd.DataFrame(tree_search.cv_results_)[['params','mean_test_score','std_test_score']])''')
md('## Cell 11 — Random Forest: actual fit and search code\nAverage 150 randomized trees to reduce single-tree variance. This captures nonlinear patterns but is less directly interpretable. The bounded grid compares depth 8/unbounded and minimum leaf sizes 2/10.')
code('''forest_pipeline = Pipeline([
    ('preprocess', preprocess()),
    ('model', RandomForestClassifier(n_estimators=150, random_state=SEED, n_jobs=1))
])
forest_grid = {'model__max_depth':[8,None], 'model__min_samples_leaf':[2,10]}
forest_search = GridSearchCV(forest_pipeline, forest_grid, **search_options)
forest_search.fit(X_train, y_train)
models['Random Forest'] = forest_search.best_estimator_
searches['Random Forest'] = forest_search
print('Selected training-CV parameters:', forest_search.best_params_)
display(pd.DataFrame(forest_search.cv_results_)[['params','mean_test_score','std_test_score']])''')
md('## Cell 12 — Inspect every candidate and fold score\nAll 13 learned-model candidates and three test-fold AP scores are available. CV test folds are within training, not the held-out project test partition.')
code('''cv_tables = []
for name, search in searches.items():
    table = pd.DataFrame(search.cv_results_)
    table.insert(0,'model',name)
    cv_tables.append(table)
all_cv = pd.concat(cv_tables, ignore_index=True)
display(all_cv[['model','params','split0_test_score','split1_test_score','split2_test_score',
                'mean_test_score','std_test_score','rank_test_score']])
assert len(all_cv) == 13''')
md('## Cell 13 — Validation comparison\nEvaluate the four fitted models at 0.50. For compatibility with the shared selection helper, `feature_set=full` means all **six inputs in this condition**, not the earlier 17-feature research benchmark. Condition names distinguish these results.')
code('''validation_records = []
for name, fitted in models.items():
    score = fitted.predict_proba(X_validation)[:,1]
    validation_records.append({'model':name, 'feature_set':'full',
                               'condition':'six_observable_inputs','partition':'validation',
                               **metrics(y_validation, score, 0.5)})
validation_table = pd.DataFrame(validation_records)
display(validation_table[['model','ap','roc_auc','precision','recall','f1','accuracy','tp','fp','fn','tn']])
expected_validation = pd.read_csv(OUTPUT/'validation_metrics.csv')
np.testing.assert_allclose(validation_table.ap, expected_validation.ap, atol=1e-12)''')
md('## Cell 14 — The production validation-only decision functions\nChoose highest validation AP; within 0.005 of the highest, prefer LR, then Tree, then Forest. For the selected model maximize validation F1 on 0.05..0.95 with 0.01 steps; ties use the larger threshold. These functions accept no test data.')
code(inspect.getsource(choose_model)+'\n\n'+inspect.getsource(choose_threshold))
md('## Cell 15 — Freeze before test evaluation\nThe production decision was saved before its test evaluation. This notebook reproduces it and writes a separate notebook decision record; it does not overwrite production selection. No model is refitted on validation.')
code('''selected_name = choose_model(validation_records)
selected_model = models[selected_name]
validation_scores = selected_model.predict_proba(X_validation)[:,1]
threshold, threshold_grid = choose_threshold(y_validation, validation_scores)
production_decision = saved('selection.json')
assert selected_name == production_decision['model']
assert threshold == production_decision['threshold']
decision = {'model':selected_name, 'threshold':threshold, 'features':FEATURES,
            'parameters':searches[selected_name].best_params_,
            'frozen_at_utc':datetime.now(timezone.utc).isoformat(),
            'purpose':'Notebook reproduction; no changes to the production decision'}
(OUTPUT/'notebook_decision.json').write_text(json.dumps(decision,indent=2))
display(pd.Series(decision))''')
md('## Cell 16 — Final frozen test comparison\nCompare all four models at 0.50 and the selected model additionally at its frozen threshold. Scores are not calibrated purchase probabilities. This replays an existing historical split, not an independent new test. Exact agreement checks tie this notebook to the deployed model.')
code('''test_records = []
for name, fitted in models.items():
    scores = fitted.predict_proba(X_test)[:,1]
    cuts = [0.5,threshold] if name == selected_name and threshold != 0.5 else [0.5]
    for cut in cuts:
        test_records.append({'model':name,'feature_set':'six_observable_inputs','partition':'test',
                             **metrics(y_test,scores,cut)})
test_table = pd.DataFrame(test_records)
display(test_table[['model','threshold','ap','roc_auc','precision','recall','f1','accuracy','tp','fp','fn','tn']])
expected_test = pd.read_csv(OUTPUT/'test_metrics.csv')
for field in ['threshold','ap','roc_auc','precision','recall','f1','accuracy','tp','fp','fn','tn']:
    np.testing.assert_allclose(test_table[field], expected_test[field], atol=1e-12)
test_table.to_csv(OUTPUT/'notebook_metrics.csv',index=False)
official = test_table[(test_table.model==selected_name)&(test_table.threshold==threshold)].iloc[0]
print(f"Detected {int(official.tp)}/{int(official.purchases)} buyers; {int(official.fp)} false alarms among {int(official.predicted_purchases)} flagged sessions.")''')
md('## Cell 17 — Explain the mistakes visually\nFalse positives are flagged sessions that did not purchase. False negatives are missed purchases. Counts and denominators make the limitations understandable to a nontechnical audience.')
code('''matrix = [[int(official.tn),int(official.fp)],[int(official.fn),int(official.tp)]]
fig, ax = plt.subplots(figsize=(5,4))
ax.imshow(matrix,cmap='Greens')
for row in range(2):
    for col in range(2):
        ax.text(col,row,str(matrix[row][col]),ha='center',va='center',fontsize=16,
                color='white' if row==0 and col==0 else 'black')
ax.set(xticks=[0,1],yticks=[0,1],xticklabels=['No purchase','Purchase'],
       yticklabels=['No purchase','Purchase'],xlabel='Predicted',ylabel='Actual',
       title=f'{selected_name} — six inputs — threshold {threshold:.2f}')
plt.tight_layout();plt.show()''')
md('## Cell 18 — Measure removal of the account inputs\nDo not claim these features were unimportant merely because they were removed. Compare the saved earlier eight-input condition and current six-input condition on the same row partitions. Each used the declared bounded search. Differences combine feature removal and any resulting training-CV parameter change; no statistical significance is claimed.')
code('''comparison = []
previous = ROOT/'results/predictor_previous_eight_inputs'
for partition in ['validation','test']:
    before = pd.read_csv(previous/f'{partition}_metrics.csv')
    after = pd.read_csv(OUTPUT/f'{partition}_metrics.csv')
    for name in ['Logistic Regression','Decision Tree','Random Forest']:
        old = before[(before.model==name)&(before.threshold==0.5)].iloc[0]
        new = after[(after.model==name)&(after.threshold==0.5)].iloc[0]
        comparison.append({'partition':partition,'model':name,'eight_input_ap':old.ap,
                           'six_input_ap':new.ap,'ap_change':new.ap-old.ap,
                           'eight_input_f1_at_050':old.f1,'six_input_f1_at_050':new.f1})
comparison_table = pd.DataFrame(comparison)
display(comparison_table)
comparison_table.to_csv(OUTPUT/'feature_removal_comparison.csv',index=False)''')
md('## Cell 19 — Predict a newly supplied six-input session\nThis constructed example is an inference demonstration, not training data or a performance claim. Account fields are absent from both this input and the model. Real users can supply the same six fields by form or CSV.')
code('''new_session = pd.DataFrame([{'ProductRelated':23,'ProductRelated_Duration':620,
                             'Informational':2,'Informational_Duration':40,
                             'Weekend':1,'VisitorType':'New_Visitor'}])[FEATURES]
score = float(selected_model.predict_proba(new_session)[0,1])
display(new_session)
print('Purchase score:',score)
print('Prediction:', 'Purchase predicted' if score>=threshold else 'No purchase predicted')
print('This score is not a guaranteed or calibrated probability.')''')
md('## Cell 20 — Verify the browser export represents this fitted model\nThe export contains the exact preprocessing and fitted estimator used by the JavaScript engine. The separate Node check covers 3,703 Python/JS cases and CSV behavior. This cell verifies the notebook scores match the production test rows and the model uses only six original inputs.')
code('''exported = json.loads((ROOT/'docs/data/predictor.json').read_text())
notebook_export = production.export_model(selected_model)
assert notebook_export == exported['inference']
assert exported['feature_set'] == 'six_observable_inputs'
assert len(set(exported['inference']['transformed_feature_groups'])) == 6
predictions = pd.read_csv(OUTPUT/'predictions.csv',float_precision='round_trip')
test_predictions = predictions[predictions.partition=='test']
np.testing.assert_allclose(selected_model.predict_proba(X_test)[:,1],test_predictions.purchase_score,atol=1e-12)
print('Notebook model matches the deployed export and all saved test scores.')
print('Browser parity command: node tests/check_predictor.cjs')''')
md('''## What to explain in the submission

1. We predict a session's Revenue label from six observed browsing inputs.
2. The baseline, linear model, tree and forest have explicit fit/search cells above.
3. Preprocessing is learned inside training folds; validation selects model and threshold.
4. The fixed test comparison and error counts quantify performance without changing the decision.
5. Account/admin variables were removed to simplify the tool; measured before/after evidence is shown rather than an unsupported feature-importance claim.
6. The six-input predictor and original full-feature research benchmark are different conditions. They must not share performance claims.

**Limits:** one historical source, reused random split, no future-store or live-session validation, uncalibrated scores, and no causal/price/preference inference. False alarms and missed buyers remain.

**Credits:** the requester supplied the dataset, specification and feature-removal request. Codex authored this cell-by-cell reproduction and executed the checks recorded in the repository. No human verification or personal contributor names are asserted. Dataset: Sakar & Kastro (2018), [UCI DOI 10.24432/C5F88Q](https://doi.org/10.24432/C5F88Q), CC BY 4.0.''')

notebook = nbf.v4.new_notebook(cells=cells)
notebook.metadata['kernelspec'] = {'display_name':'Python 3','language':'python','name':'python3'}
notebook.metadata['language_info'] = {'name':'python','version':'3.12'}
path = ROOT/'notebooks/03_six_input_models_step_by_step.ipynb'
nbf.write(notebook,path)
print(f'Created {path.name}: {len(cells)} cells, {sum(c.cell_type=="code" for c in cells)} code cells')
