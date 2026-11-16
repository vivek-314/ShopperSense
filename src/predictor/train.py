"""Train the eight-input product model, freeze its decision, and export real JS inference.

Run: python -m src.predictor.train
Original research results are preserved. This is a new, separately labelled condition.
"""
import hashlib
import json
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ..config import ROOT, SEED
from ..data import load_data
from ..evaluate import choose_model, choose_threshold, metrics
from ..run_all import write_json
from ..train import definitions

FIELDS = [
    ('product_pages', 'ProductRelated', 'Product pages viewed', True),
    ('product_seconds', 'ProductRelated_Duration', 'Time on product pages', False),
    ('information_pages', 'Informational', 'Information pages viewed', True),
    ('information_seconds', 'Informational_Duration', 'Time on information pages', False),
    ('account_pages', 'Administrative', 'Account / admin pages viewed', True),
    ('account_seconds', 'Administrative_Duration', 'Time on account / admin pages', False),
    ('weekend', 'Weekend', 'Weekend visit', True),
]
NUMERIC = [field[1] for field in FIELDS]
CATEGORICAL = ['VisitorType']
FEATURES = NUMERIC + CATEGORICAL
FIELD_BY_SOURCE = {source: public for public, source, _, _ in FIELDS} | {'VisitorType': 'visitor_type'}


def preprocess(scale=False):
    steps = [('impute', SimpleImputer(strategy='median'))]
    if scale:
        steps.append(('scale', StandardScaler()))
    return ColumnTransformer([
        ('numeric', Pipeline(steps), NUMERIC),
        ('category', Pipeline([('impute', SimpleImputer(strategy='constant', fill_value='__missing__')), ('onehot', OneHotEncoder(handle_unknown='ignore'))]), CATEGORICAL),
    ], remainder='drop')


def to_public(row):
    return {FIELD_BY_SOURCE[source]: (str(row[source]) if source == 'VisitorType' else float(row[source])) for source in FEATURES}


def export_model(pipeline):
    prep, estimator = pipeline.named_steps['preprocess'], pipeline.named_steps['model']
    numeric = prep.named_transformers_['numeric']
    vocabulary = prep.named_transformers_['category'].named_steps['onehot'].categories_[0].tolist()
    groups = [FIELD_BY_SOURCE[source] for source in NUMERIC] + ['visitor_type']*len(vocabulary)
    result = {
        'numeric_fields': [FIELD_BY_SOURCE[source] for source in NUMERIC],
        'numeric_medians': numeric.named_steps['impute'].statistics_.tolist(),
        'category_field': 'visitor_type', 'categories': vocabulary, 'transformed_feature_groups': groups,
        'scale_mean': numeric.named_steps['scale'].mean_.tolist() if 'scale' in numeric.named_steps else None,
        'scale_std': numeric.named_steps['scale'].scale_.tolist() if 'scale' in numeric.named_steps else None,
    }
    if hasattr(estimator, 'coef_'):
        result.update({'kind': 'logistic', 'coefficient': estimator.coef_[0].tolist(), 'intercept': float(estimator.intercept_[0]), 'explanation_units': 'log odds'})
    else:
        estimators = estimator.estimators_ if hasattr(estimator, 'estimators_') else [estimator]
        trees = []
        for fitted in estimators:
            tree = fitted.tree_
            value = tree.value[:, 0, :]
            purchase = value[:, 1] / value.sum(axis=1)
            trees.append({'left': tree.children_left.tolist(), 'right': tree.children_right.tolist(), 'feature': tree.feature.tolist(), 'threshold': tree.threshold.tolist(), 'purchase': purchase.tolist()})
        result.update({'kind': 'forest' if len(trees) > 1 else 'tree', 'trees': trees, 'tree_input_dtype': 'float32', 'explanation_units': 'score points'})
    return result


def run():
    start = time.perf_counter()
    output = ROOT/'results/predictor'
    output.mkdir(parents=True, exist_ok=True)
    raw, frame = load_data(ROOT/'data/raw/online_shoppers_intention.csv')
    source_hash = hashlib.sha256((ROOT/'data/raw/online_shoppers_intention.csv').read_bytes()).hexdigest()
    split = json.loads((ROOT/'results/splits.json').read_text())
    if source_hash != split['source_sha256']:
        raise ValueError('Supplied data hash differs from saved row partition')
    ids = {p: v['source_row_ids'] for p, v in split['partitions'].items()}
    train = frame.loc[ids['train'], FEATURES]
    train_y = frame.loc[ids['train'], 'Revenue']
    models, records, cv_frames, warnings_log = {}, [], [], []
    parameters = {}
    for name, (estimator, grid) in definitions().items():
        print(f'Training eight-input model: {name}', flush=True)
        model = Pipeline([('preprocess', preprocess(name == 'Logistic Regression')), ('model', estimator)])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            if name == 'Dummy':
                model.fit(train, train_y)
                params = {}
            else:
                search = GridSearchCV(model, grid, scoring='average_precision', cv=StratifiedKFold(3, shuffle=True, random_state=SEED), n_jobs=1, refit=True, return_train_score=True, error_score='raise')
                search.fit(train, train_y)
                model, params = search.best_estimator_, search.best_params_
                cv = pd.DataFrame(search.cv_results_)
                cv.insert(0, 'model', name)
                cv_frames.append(cv)
        for warn in caught:
            warnings_log.append({'model': name, 'category': warn.category.__name__, 'message': str(warn.message)})
        if any(issubclass(w.category, ConvergenceWarning) for w in caught):
            raise RuntimeError('Resolve estimator convergence before exporting product model')
        models[name] = model
        parameters[name] = params
        scores = model.predict_proba(frame.loc[ids['validation'], FEATURES])[:, 1]
        records.append({'model': name, 'feature_set': 'full', 'condition': 'eight_observable_inputs', 'partition': 'validation', **metrics(frame.loc[ids['validation'], 'Revenue'], scores)})
        print(f'  validation AP {records[-1]["ap"]:.6f}', flush=True)
    selected = choose_model(records)
    validation_scores = models[selected].predict_proba(frame.loc[ids['validation'], FEATURES])[:, 1]
    threshold, grid = choose_threshold(frame.loc[ids['validation'], 'Revenue'], validation_scores)
    source_digest = hashlib.sha256()
    for path in sorted((ROOT/'src/predictor').glob('*.py')):
        source_digest.update(path.read_bytes())
    code_hash = source_digest.hexdigest()
    run_id = f'predictor-{source_hash[:8]}-{code_hash[:8]}'
    frozen = {
        'run_id': run_id, 'source_code_sha256': code_hash, 'data_sha256': source_hash,
        'model': selected, 'threshold': threshold, 'parameters': parameters[selected],
        'features': FEATURES, 'public_fields': list(FIELD_BY_SOURCE.values()), 'feature_set': 'eight_observable_inputs',
        'frozen_at_utc': datetime.now(timezone.utc).isoformat(),
        'selection_rule': 'Training-only 3-fold AP grid search; highest validation AP with 0.005 simplicity tolerance (LR, Tree, Forest). Threshold maximizes validation F1 on 0.05..0.95, ties prefer higher.',
        'test_status': 'Same historical research split reused for this separately specified condition; not an independent new external validation. No product-model tuning from test.',
        'validation_evidence': records,
    }
    # Freeze BEFORE accessing product-model test outcomes.
    write_json(output/'selection.json', frozen)
    pd.concat(cv_frames, ignore_index=True).to_csv(output/'cv_results.csv', index=False)
    pd.DataFrame(records).to_csv(output/'validation_metrics.csv', index=False)
    pd.DataFrame(grid).to_csv(output/'threshold_grid.csv', index=False)
    write_json(output/'warnings.json', warnings_log)
    tests, predictions = [], []
    for name, model in models.items():
        scores = model.predict_proba(frame.loc[ids['test'], FEATURES])[:, 1]
        for cut in ([.5, threshold] if name == selected and threshold != .5 else [.5]):
            tests.append({'model': name, 'feature_set': 'eight_observable_inputs', 'partition': 'test', **metrics(frame.loc[ids['test'], 'Revenue'], scores, cut)})
    fitted = models[selected]
    for partition in ['train', 'validation', 'test']:
        scores = fitted.predict_proba(frame.loc[ids[partition], FEATURES])[:, 1]
        predictions.append(pd.DataFrame({'source_row_id': ids[partition], 'partition': partition, 'y_true': frame.loc[ids[partition], 'Revenue'].to_numpy(), 'purchase_score': scores, 'threshold': threshold, 'predicted_label': (scores >= threshold).astype(int)}))
    pd.concat(predictions).to_csv(output/'predictions.csv', index=False)
    pd.DataFrame(tests).to_csv(output/'test_metrics.csv', index=False)
    (ROOT/'results/models').mkdir(exist_ok=True)
    joblib.dump(fitted, ROOT/'results/models/marketer_predictor.joblib')
    official = next(row for row in tests if row['model'] == selected and row['threshold'] == threshold)
    numeric_contract = []
    for public, source, label, integer in FIELDS:
        observed = train[source]
        numeric_contract.append({'name': public, 'source': source, 'label': label, 'integer': integer, 'min': 0, 'training_min': float(observed.min()), 'training_max': float(observed.max()), 'training_median': float(observed.median()), 'training_p99': float(observed.quantile(.99))})
    payload = {
        'schema_version': 1, 'run_id': run_id, 'data_sha256': source_hash, 'model_name': selected,
        'feature_set': 'eight_observable_inputs', 'threshold': threshold, 'trained_on': len(train),
        'input_contract': numeric_contract, 'visitor_types': ['New_Visitor', 'Returning_Visitor', 'Other'],
        'inference': export_model(fitted), 'validation_metrics': metrics(frame.loc[ids['validation'], 'Revenue'], validation_scores, threshold),
        'test_metrics': official, 'test_baseline': next(r for r in tests if r['model'] == 'Dummy'),
        'source': 'https://doi.org/10.24432/C5F88Q',
        'limits': 'Historical session summaries, not validated early-session or cross-store performance. Eight-input scores are not calibrated purchase probabilities. Imported/new sessions are scored, not used for retraining. No automatic analytics connection.',
        'explanation_method': 'Trees: average changes in purchase proportion along each actual decision path, grouped by original input; contributions sum to the score minus root average. Path-dependent model attribution, not a causal explanation. Logistic: signed contributions in log odds.',
    }
    write_json(ROOT/'docs/data/predictor.json', payload)
    # Every existing split row checks Python/JS parity; extra manually constructed inputs are NOT training data.
    fixture_rows = []
    for partition in ['validation', 'test']:
        data = frame.loc[ids[partition], FEATURES]
        scores = fitted.predict_proba(data)[:, 1]
        fixture_rows.extend({'name': f'{partition}:{idx}', 'input': to_public(row), 'score': float(score)} for (idx, row), score in zip(data.iterrows(), scores))
    new_inputs = [
        {'product_pages': 7, 'product_seconds': 311, 'information_pages': 2, 'information_seconds': 55, 'account_pages': 1, 'account_seconds': 31, 'weekend': 1, 'visitor_type': 'New_Visitor'},
        {'product_pages': 43, 'product_seconds': 1567.25, 'information_pages': 3, 'information_seconds': 110.5, 'account_pages': 4, 'account_seconds': 80.5, 'weekend': 0, 'visitor_type': 'Returning_Visitor'},
        {'product_pages': 0, 'product_seconds': 0, 'information_pages': 0, 'information_seconds': 0, 'account_pages': 1, 'account_seconds': 20, 'weekend': 0, 'visitor_type': 'Other'},
    ]
    for n, public in enumerate(new_inputs):
        values = {source: public[FIELD_BY_SOURCE[source]] for source in FEATURES}
        score = fitted.predict_proba(pd.DataFrame([values]))[0, 1]
        fixture_rows.append({'name': f'new-input:{n}', 'input': public, 'score': float(score)})
    write_json(output/'parity_fixtures.json', fixture_rows)
    # Clearly labelled historical examples only; an empty template supports fresh external records.
    example_ids = ids['train'][:5]
    example_rows = [{'session_id': f'historical-example-{i+1}', **to_public(frame.loc[idx])} for i, idx in enumerate(example_ids)]
    pd.DataFrame(example_rows).to_csv(ROOT/'docs/assets/example-sessions.csv', index=False)
    columns = ['session_id'] + [field[0] for field in FIELDS[:-1]] + ['visitor_type', 'weekend']
    (ROOT/'docs/assets/session-template.csv').write_text(','.join(columns)+'\n', encoding='utf-8')
    write_json(output/'run.json', {'run_id': run_id, 'seconds': time.perf_counter()-start, 'test_evaluation_started_after': frozen['frozen_at_utc'], 'export_bytes': (ROOT/'docs/data/predictor.json').stat().st_size})
    print(json.dumps({'selected': selected, 'threshold': threshold, 'test': official, 'export_bytes': (ROOT/'docs/data/predictor.json').stat().st_size}), flush=True)


if __name__ == '__main__':
    run()
