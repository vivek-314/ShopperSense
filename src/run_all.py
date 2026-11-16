"""Offline evidence generator. Run from the repository root: python -m src.run_all."""
import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from .config import FEATURES, ROOT, SEED
from .data import duplicate_audit, inspect_data, load_data, split_rows
from .evaluate import choose_model, choose_threshold, metrics
from .preprocessing import preprocessor
from .train import train_variant


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False, default=lambda x: x.item() if hasattr(x, 'item') else str(x)), encoding='utf-8')


def code_hash():
    digest = hashlib.sha256()
    for path in sorted((ROOT/'src').glob('*.py')):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def error_analysis(frame, train_ids, test_ids, scores, threshold):
    observed = frame.loc[test_ids].copy()
    observed['purchase_score'] = scores
    observed['predicted_label'] = (scores >= threshold).astype(int)
    observed['error'] = np.select([(observed.Revenue == 0) & (observed.predicted_label == 1), (observed.Revenue == 1) & (observed.predicted_label == 0)], ['FP', 'FN'], default='correct')
    q1, q2 = frame.loc[train_ids, 'ProductRelated'].quantile([1/3, 2/3]).tolist()
    observed['ProductRelated_bin'] = pd.cut(observed.ProductRelated, [-np.inf, q1, q2, np.inf], labels=['low', 'medium', 'high'], include_lowest=True)
    groups = []
    for key in ['VisitorType', 'ProductRelated_bin']:
        for label, sub in observed.groupby(key, observed=True):
            groups.append({'grouping': key, 'group': str(label), **metrics(sub.Revenue, sub.purchase_score, threshold)})
    examples = {}
    for error in ['FP', 'FN']:
        cols = ['VisitorType', 'ProductRelated', 'ProductRelated_Duration', 'PageValues', 'Revenue', 'purchase_score', 'predicted_label']
        examples[error] = observed[observed.error == error].sort_index().head(4)[cols].reset_index().to_dict('records')
    return {'partition': 'test', 'threshold': threshold, 'example_rule': 'First four source_row_id values in each error category; no cherry-picking.', 'bin_boundaries': {'low': f'ProductRelated <= {q1:g}', 'medium': f'{q1:g} < ProductRelated <= {q2:g}', 'high': f'ProductRelated > {q2:g}', 'training_quantiles': [q1, q2]}, 'overall': metrics(observed.Revenue, scores, threshold), 'subgroups': groups, 'examples': examples}


def run(data, output=ROOT/'results'):
    started = time.perf_counter()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output/'models').mkdir(exist_ok=True)
    raw, frame = load_data(data)
    manifest, quality = inspect_data(data, raw, frame)
    parts, split_record = split_rows(frame, manifest['sha256'])
    overlaps, group_parts = duplicate_audit(frame, parts)
    quality['cross_partition_duplicate_audit'] = overlaps
    group_record = {'method': overlaps['group_sensitivity_method'], 'source_sha256': manifest['sha256'], 'seed': SEED, 'partitions': {p: {'source_row_ids': ids.tolist(), 'support': len(ids), 'purchases': int(frame.loc[ids].Revenue.sum())} for p, ids in group_parts.items()}}
    write_json(output/'data_manifest.json', manifest)
    write_json(output/'quality.json', quality)
    write_json(output/'splits.json', split_record)
    write_json(output/'group_splits.json', group_record)
    source_code_hash = code_hash()
    run_id = f'ss-{manifest["sha256"][:8]}-{source_code_hash[:8]}'
    events = [{'event': 'splits_persisted', 'utc': datetime.now(timezone.utc).isoformat()}]
    models, parameters, validation_records, predictions, candidates, warning_records = {}, {}, [], [], [], []
    train_x, train_y = frame.loc[parts['train'], FEATURES], frame.loc[parts['train'], 'Revenue']
    for feature_set, names in [('full', ['Dummy', 'Logistic Regression', 'Decision Tree', 'Random Forest']), ('no_PageValues', ['Logistic Regression', 'Random Forest'])]:
        for name in names:
            print(f'Training {name} / {feature_set}', flush=True)
            fitted, params, cv, warns = train_variant(name, feature_set, train_x, train_y)
            key = (name, feature_set)
            models[key] = fitted
            parameters[f'{name}/{feature_set}'] = {'selected_grid_params': params, 'estimator_params': fitted.named_steps['model'].get_params(), 'numeric_features': fitted.named_steps['preprocess'].transformers_[0][2], 'categorical_vocabulary': [v.tolist() for v in fitted.named_steps['preprocess'].named_transformers_['categorical'].named_steps['onehot'].categories_]}
            joblib.dump(fitted, output/'models'/f'{name.replace(" ", "_")}_{feature_set}.joblib')
            if not cv.empty:
                candidates.append(cv)
            warning_records.extend(warns)
            ids = parts['validation']
            scores = fitted.predict_proba(frame.loc[ids, FEATURES])[:, 1]
            record = {'model': name, 'feature_set': feature_set, 'partition': 'validation', **metrics(frame.loc[ids, 'Revenue'], scores)}
            validation_records.append(record)
            predictions.append(pd.DataFrame({'source_row_id': ids, 'partition': 'validation', 'model': name, 'feature_set': feature_set, 'y_true': frame.loc[ids, 'Revenue'].to_numpy(), 'purchase_score': scores, 'threshold': .5, 'predicted_label': (scores >= .5).astype(int)}))
            print(f'  validation AP={record["ap"]:.6f}', flush=True)
    validation_predictions = pd.concat(predictions, ignore_index=True)
    selected = choose_model(validation_records)
    selected_validation = validation_predictions[(validation_predictions.model == selected) & (validation_predictions.feature_set == 'full')]
    threshold, threshold_grid = choose_threshold(selected_validation.y_true, selected_validation.purchase_score)
    # A fixed 0.5 threshold makes depth comparisons comparable; no selection from these curves.
    depth_records = []
    for depth in [3, 5, 8, 12, None]:
        tree = Pipeline([('preprocess', preprocessor()), ('model', DecisionTreeClassifier(max_depth=depth, min_samples_leaf=5, random_state=SEED))]).fit(train_x, train_y)
        for partition in ['train', 'validation']:
            ids = parts[partition]
            depth_records.append({'max_depth': 'unbounded' if depth is None else str(depth), 'partition': partition, **metrics(frame.loc[ids, 'Revenue'], tree.predict_proba(frame.loc[ids, FEATURES])[:, 1], .5)})
    pd.DataFrame(depth_records).to_csv(output/'tree_depth_comparison.csv', index=False)
    selection = {
        'run_id': run_id, 'source_code_sha256': source_code_hash, 'data_sha256': manifest['sha256'],
        'frozen_at_utc': datetime.now(timezone.utc).isoformat(), 'selected_model': selected, 'feature_set': 'full',
        'selected_parameters': parameters[f'{selected}/full']['selected_grid_params'], 'threshold': threshold,
        'model_rule': 'Highest validation AP; within 0.005 of highest, prefer Logistic Regression, Decision Tree, Random Forest in that order. Dummy excluded.',
        'threshold_rule': 'Maximize validation purchase F1 on 0.05..0.95 by 0.01, ties use larger threshold. Illustrative balance; no known cost ratio.',
        'validation_evidence': validation_records, 'feature_condition': 'Full features are prespecified primary; no-PageValues LR/RF are secondary and not selection candidates.',
        'training_policy': 'All models fitted only on training; no refit on validation after selection.',
        'duplicate_sensitivity_policy': overlaps['group_sensitivity_method'],
        'frozen_variants': [{'model': n, 'feature_set': f, 'parameters': parameters[f'{n}/{f}']['selected_grid_params']} for n, f in models],
    }
    # This write intentionally occurs BEFORE the first test prediction.
    write_json(output/'selection.json', selection)
    events.append({'event': 'decisions_frozen', 'utc': selection['frozen_at_utc']})
    write_json(output/'parameters.json', parameters)
    write_json(output/'warnings.json', warning_records)
    pd.concat(candidates, ignore_index=True).to_csv(output/'cv_results.csv', index=False)
    pd.DataFrame(validation_records).to_csv(output/'validation_metrics.csv', index=False)
    validation_predictions.to_csv(output/'validation_predictions.csv', index=False)
    pd.DataFrame(threshold_grid).to_csv(output/'threshold_grid.csv', index=False)
    test_records, test_predictions = [], []
    events.append({'event': 'test_evaluation_started', 'utc': datetime.now(timezone.utc).isoformat()})
    for (name, feature_set), fitted in models.items():
        ids = parts['test']
        scores = fitted.predict_proba(frame.loc[ids, FEATURES])[:, 1]
        for cut in ([.5, threshold] if name == selected and feature_set == 'full' and threshold != .5 else [.5]):
            test_records.append({'model': name, 'feature_set': feature_set, 'partition': 'test', **metrics(frame.loc[ids, 'Revenue'], scores, cut)})
            test_predictions.append(pd.DataFrame({'source_row_id': ids, 'partition': 'test', 'model': name, 'feature_set': feature_set, 'y_true': frame.loc[ids, 'Revenue'].to_numpy(), 'purchase_score': scores, 'threshold': cut, 'predicted_label': (scores >= cut).astype(int)}))
    pd.DataFrame(test_records).to_csv(output/'test_metrics.csv', index=False)
    pd.concat(test_predictions, ignore_index=True).to_csv(output/'test_predictions.csv', index=False)
    comparison = [r for r in validation_records+test_records if r['model'] in ['Logistic Regression', 'Random Forest'] and r['threshold'] == .5]
    pd.DataFrame(comparison).to_csv(output/'pagevalues_comparison.csv', index=False)
    selected_test_scores = models[selected, 'full'].predict_proba(frame.loc[parts['test'], FEATURES])[:, 1]
    errors = error_analysis(frame, parts['train'], parts['test'], selected_test_scores, threshold)
    errors.update({'model': selected, 'feature_set': 'full'})
    write_json(output/'error_analysis.json', errors)
    group_model = clone(models[selected, 'full']).fit(frame.loc[group_parts['train'], FEATURES], frame.loc[group_parts['train'], 'Revenue'])
    group_records, group_predictions = [], []
    for partition in ['validation', 'test']:
        ids = group_parts[partition]
        scores = group_model.predict_proba(frame.loc[ids, FEATURES])[:, 1]
        group_records.append({'model': selected, 'feature_set': 'full', 'partition': f'group_{partition}', **metrics(frame.loc[ids, 'Revenue'], scores, threshold)})
        group_predictions.append(pd.DataFrame({'source_row_id': ids, 'partition': f'group_{partition}', 'model': selected, 'feature_set': 'full', 'y_true': frame.loc[ids, 'Revenue'].to_numpy(), 'purchase_score': scores, 'threshold': threshold, 'predicted_label': (scores >= threshold).astype(int)}))
    pd.DataFrame(group_records).to_csv(output/'group_sensitivity.csv', index=False)
    pd.concat(group_predictions).to_csv(output/'group_predictions.csv', index=False)
    write_json(output/'events.json', events)
    run_record = {'run_id': run_id, 'data_sha256': manifest['sha256'], 'source_code_sha256': source_code_hash, 'training_and_evaluation_seconds': time.perf_counter()-started, 'test_policy': 'No test-based model or threshold changes', 'environment': manifest['environment']}
    write_json(output/'run.json', run_record)
    from .report import generate_report
    generate_report(output, ROOT/'docs', frame, parts)
    print(json.dumps({'selected': selected, 'threshold': threshold, 'run_id': run_id, 'seconds': time.perf_counter()-started}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT/'data/raw/online_shoppers_intention.csv')
    args = parser.parse_args()
    run(args.data)


if __name__ == '__main__':
    main()
