"""Create inspectable notebooks whose analysis calls the shared project modules."""
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]


def md(text):
    return nbf.v4.new_markdown_cell(text)


def code(text):
    return nbf.v4.new_code_cell(text)


setup = """from pathlib import Path
import sys, json
ROOT = Path.cwd().resolve()
if ROOT.name == 'notebooks':
    ROOT = ROOT.parent
assert (ROOT / 'src').is_dir(), 'Start Jupyter from the repository root or notebooks/.'
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import pandas as pd
from IPython.display import display, SVG
from src.data import load_data
from src.config import FEATURES, NUMERIC, CATEGORICAL
def saved(name):
    return json.loads((ROOT / 'results' / name).read_text(encoding='utf-8'))
raw, frame = load_data(ROOT / 'data/raw/online_shoppers_intention.csv')
splits = saved('splits.json')
train_ids = splits['partitions']['train']['source_row_ids']
train = frame.loc[train_ids]
"""

nb1 = nbf.v4.new_notebook(cells=[
    md("""# ShopperSense · 01 / Data and reasoning

**Question:** How well can recorded browsing-session characteristics identify sessions ending in a purchase, compared with always predicting no purchase?

One row is a session. Revenue maps explicitly to 0/1. These are historical session summaries, not proof of early-session availability. Run `python -m src.run_all` from the repository root first. This notebook executes structural inspection and training-only EDA through shared source code. It does not revise model choices using test results."""),
    code(setup),
    md('## Preserve the raw data and identify the file\nThe locally supplied download date is unknown. SHA-256 identifies the supplied bytes; no newly downloaded archive is assumed. Source: [UCI dataset 468](https://doi.org/10.24432/C5F88Q), Sakar & Kastro (2018), CC BY 4.0.'),
    code("""import hashlib
manifest = saved('data_manifest.json')
assert hashlib.sha256((ROOT / 'data/raw/online_shoppers_intention.csv').read_bytes()).hexdigest() == manifest['sha256']
display(pd.Series({k: manifest[k] for k in ['source_filename','sha256','rows','predictors','class_counts','provenance','license']}))
display(pd.Series(manifest['dtypes'], name='source dtype'))"""),
    md('## Whole-file structural audit\nDuplicates are excess occurrences after the first row. A repeated observation is not automatically erroneous. Preserve all source rows and audit cross-partition overlap separately. Keep large durations and counts; a plotting transform is not a model transform.'),
    code("""from src.data import inspect_data
_, measured_quality = inspect_data(ROOT / 'data/raw/online_shoppers_intention.csv', raw, frame)
display(pd.DataFrame({k: measured_quality[k] for k in ['missing_by_column','nonfinite_by_numeric_column','negative_by_numeric_column']}))
display(pd.Series({k: measured_quality[k] for k in ['complete_duplicate_rows','identical_predictor_extra_rows','unexpected_categories','source_expectations_match']}))
display(pd.DataFrame(measured_quality['numeric_summary']))"""),
    md('## Split before exploring\nFirst hold out 15% test, then split 15/85 of the remainder as validation. Both calls stratify Revenue with random_state=42. ID arrays cover the file exactly once. The CV preprocessing is fitted within each training fold.'),
    code("""from src.data import split_rows
parts, repeat = split_rows(frame, manifest['sha256'])
assert repeat == splits
display(pd.DataFrame({p: {'sessions': len(ids), 'purchases': int(frame.loc[ids].Revenue.sum())} for p, ids in parts.items()}).T)
display(pd.DataFrame(saved('quality.json')['cross_partition_duplicate_audit']['pairs']))"""),
    md('## Class imbalance: a reference, not a solution\nWhole-file counts describe the supplied source. Always predicting no purchase can have high accuracy while missing every buyer; AP, purchase recall and F1 are more revealing.'),
    code("""display(frame.Revenue.value_counts().rename(index={0: 'No purchase', 1: 'Purchase'}))
display(SVG(filename=str(ROOT / 'results/figures/class_balance.svg')))"""),
    md('## Training-only visitor rates and browsing intensity\nShow numerators and denominators, and avoid claims about motives. The `Other` category has little support. Plot log(1 + pages) to make skew visible; normalization is within target class.'),
    code("""display(train.groupby('VisitorType').Revenue.agg(sessions='size', purchases='sum', purchase_rate='mean'))
display(SVG(filename=str(ROOT / 'results/figures/visitor_rates.svg')))
display(train.groupby('Revenue').ProductRelated.agg(['size','median','max']))
display(SVG(filename=str(ROOT / 'results/figures/browsing_intensity.svg')))"""),
    md('## Preprocessing rationale\nNumeric training medians retain rows if missing values occur; fixed categorical missing tokens preserve missingness. One-hot encoding prevents ordered interpretation of numeric category codes, with unknown categories ignored. Logistic Regression uses numeric StandardScaler; trees keep the original scale. Weekend is explicit binary. Revenue and row IDs are not inputs. No SMOTE, class weights, or outlier deletion.'),
    code("""from src.preprocessing import preprocessor
display(preprocessor(scale=True))
assert 'Revenue' not in FEATURES and 'source_row_id' not in FEATURES
assert 'PageValues' not in preprocessor(without_pagevalues=True).transformers[0][2]
display(pd.DataFrame({'training_skew': train[NUMERIC].skew(), 'training_p99': train[NUMERIC].quantile(.99)}))"""),
    md('## Availability limitation\nPageValues is an analytics measure connected to pages visited before transactions. Its precise construction timing in this file is unresolved; do not assert proven leakage. Prespecify the same-row no-PageValues LR and RF experiment. Remaining counts, durations, bounce and exit summaries may also be unavailable at an early decision point. These observations are associations within one historical source, without price/quality preferences or causal outcomes.'),
])

nb2 = nbf.v4.new_notebook(cells=[
    md("""# ShopperSense · 02 / Models and results

Run the training pipeline first: `python -m src.run_all`. This notebook imports the shared estimators, evaluation and selection functions, audits their saved evidence and explains results. It does **not** train an alternative model or tune from test outcomes. The pipeline trains all models; notebook execution is a separate verification gate."""),
    code(setup),
    md('## A baseline and three standard classifiers\nDummy always predicts the majority. Logistic Regression models a regularized linear log-odds boundary. A Decision Tree gives nonlinear rules but can overfit. A Random Forest averages randomized trees to reduce variance, with less direct interpretability. All learned models use bounded, training-only 3-fold stratified CV ranked by purchase average precision (AP).'),
    code("""from src.train import definitions
display(pd.DataFrame([{'model': name, 'estimator': str(estimator), 'grid': grid} for name, (estimator, grid) in definitions().items()]))
cv = pd.read_csv(ROOT / 'results/cv_results.csv')
display(cv[['model','feature_set','params','split0_test_score','split1_test_score','split2_test_score','mean_test_score','std_test_score','rank_test_score']])
assert not any(w['category'] == 'ConvergenceWarning' for w in saved('warnings.json'))"""),
    md('## Validation chooses; test evaluates\nTake the highest validation AP with a 0.005 practical tolerance favoring LR, then Tree, then Forest. Full features are primary. For the selected model maximize validation F1 on 0.05..0.95 in 0.01 steps; ties prefer the larger threshold. These functions have no test-data arguments. AP is not trapezoidal PR-AUC; scores are not claimed calibrated.'),
    code("""from src.evaluate import choose_model, choose_threshold, metrics
selection = saved('selection.json')
validation = pd.read_csv(ROOT / 'results/validation_metrics.csv')
display(validation)
assert choose_model(validation.to_dict('records')) == selection['selected_model']
predictions = pd.read_csv(ROOT / 'results/validation_predictions.csv')
selected_rows = predictions[(predictions.model == selection['selected_model']) & (predictions.feature_set == 'full')]
threshold, grid = choose_threshold(selected_rows.y_true, selected_rows.purchase_score)
assert threshold == selection['threshold']
display(pd.Series({k: selection[k] for k in ['selected_model','selected_parameters','threshold','model_rule','threshold_rule','frozen_at_utc','run_id']}))
display(pd.DataFrame(saved('events.json')))"""),
    md('## Held-out test: report without revising the choice\nAt 0.50 compare all frozen variants; additionally report the selected model at the frozen threshold. Accuracy counts correct non-buyers too; purchase precision is TP/(TP+FP), recall TP/(TP+FN), F1 their harmonic mean. AP summarizes ranking, ROC-AUC complements it. Zero predicted positives gives undefined precision displayed as 0 by convention.'),
    code("""test = pd.read_csv(ROOT / 'results/test_metrics.csv')
display(test)
official = test[(test.model == selection['selected_model']) & (test.feature_set == 'full') & (test.threshold == selection['threshold'])].iloc[0]
print(f\"Selected {selection['selected_model']}; test AP {official.ap:.6f}; F1 {official.f1:.6f}; recall {official.recall:.6f}.\")
print(f\"Detected {int(official.tp)} / {int(official.purchases)} buyers; false alarms {int(official.fp)} / {int(official.support-official.purchases)} non-buyers.\")
display(SVG(filename=str(ROOT / 'results/figures/precision_recall.svg')))
display(SVG(filename=str(ROOT / 'results/figures/confusion.svg')))"""),
    md('## Tree capacity diagnostic\nKeep min_samples_leaf=5 and threshold=0.50 fixed. Compare training and validation F1 at several depths; this diagnostic did not replace the declared model selection rule.'),
    code("""display(pd.read_csv(ROOT / 'results/tree_depth_comparison.csv'))
display(SVG(filename=str(ROOT / 'results/figures/tree_depth.svg')))"""),
    md('## PageValues sensitivity\nLR and RF are independently retrained without PageValues, on the same rows and original small grids. Inspect validation dependence before freezing, then report prespecified test variants. Full features stay primary; removal is not proof of leakage or of real-time availability.'),
    code("""comparison = pd.read_csv(ROOT / 'results/pagevalues_comparison.csv')
display(comparison)
from src.report import sensitivity_text
print(sensitivity_text({'validation_metrics': validation.to_dict('records'), 'test_metrics': test.to_dict('records')}))
display(SVG(filename=str(ROOT / 'results/figures/pagevalues.svg')))"""),
    md('## Mistakes and denominators\nFP means a predicted buyer did not purchase; FN means a real buyer was missed. The first four source-row IDs per category are shown deterministically. ProductRelated bins use training tertiles. Small subgroup support precludes strong conclusions.'),
    code("""errors = saved('error_analysis.json')
display(pd.Series(errors['bin_boundaries']))
display(pd.DataFrame(errors['subgroups']))
for kind, examples in errors['examples'].items():
    print(kind)
    display(pd.DataFrame(examples))"""),
    md('## Duplicate sensitivity\nThis separate group-aware split keeps identical predictor vectors together. It reuses the primary selected model parameters and threshold without group-test selection. Different rows and supports make it a sensitivity check rather than a replacement for the primary result.'),
    code("""display(pd.read_csv(ROOT / 'results/group_sensitivity.csv'))
display(pd.Series(saved('run.json')))"""),
    md('## A meaningful validation interaction\nThe site exports exact validation labels/scores in row-ID order. Rule: score >= threshold. Changing the slider changes counts only; no retraining and no official test changes. The JS/Python parity check also covers equality, endpoint thresholds and zero positives.'),
    code("""web = json.loads((ROOT / 'docs/data/threshold_validation.json').read_text())
assert web['partition'] == 'validation'
display(pd.DataFrame([{'threshold': t, **metrics(web['y_true'], web['purchase_score'], t)} for t in [0., .5, 1., selection['threshold']]]))"""),
    md('## Conclusion and disclosure\nSession summaries contain predictive signal beyond a majority baseline, but there are missed purchases and dependence on PageValues. One random historical split does not validate future shops, early-session intervention, causal uplift, preferences or calibration. No statistical significance or confidence interval is claimed.\n\nWe developed ShopperSense as a course project to explore how browsing behaviour relates to online purchase intent. We used AI tools, including ChatGPT and Codex, to assist with planning, coding, debugging, documentation, and website development. AI-generated code and suggestions supported the project. Human verification has been completed, and I have a complete understanding of the project. Responsibility for the submitted work, its interpretation, and its limitations remains mine. Sources: UCI [doi:10.24432/C5F88Q](https://doi.org/10.24432/C5F88Q), [scikit-learn metric documentation](https://scikit-learn.org/stable/modules/model_evaluation.html).'),
])

for name, notebook in [('01_data_and_reasoning', nb1), ('02_models_and_results', nb2)]:
    notebook.metadata['kernelspec'] = {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}
    notebook.metadata['language_info'] = {'name': 'python', 'version': '3.12'}
    nbf.write(notebook, ROOT/'notebooks'/f'{name}.ipynb')
    print(f'Created {name}.ipynb')
