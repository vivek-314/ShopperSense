import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from src.config import CATEGORICAL, FEATURES, NUMERIC, ROOT
from src.data import binary, load_data, split_rows
from src.evaluate import choose_model, choose_threshold, metrics
from src.preprocessing import preprocessor
from src.train import definitions


def read(name):
    return json.loads((ROOT/'results'/name).read_text(encoding='utf-8'))


@pytest.fixture(scope='module')
def study():
    raw, frame = load_data(ROOT/'data/raw/online_shoppers_intention.csv')
    return raw, frame, read('splits.json'), read('selection.json')


def test_schema_and_explicit_target(study):
    raw, frame, _, _ = study
    assert set(frame.columns) == set(FEATURES+['Revenue'])
    assert 'Revenue' not in FEATURES and 'source_row_id' not in FEATURES
    assert set(frame.Revenue.unique()) == {0, 1}
    assert frame.Revenue.sum() == 1908
    assert binary(pd.Series(['FALSE', 'TRUE', '0', '1'])).tolist() == [0, 1, 0, 1]
    with pytest.raises(ValueError):
        binary(pd.Series(['yes', None]))
    raw_hash = hashlib.sha256((ROOT/'online_shoppers_intention.csv').read_bytes()).hexdigest() if (ROOT/'online_shoppers_intention.csv').exists() else hashlib.sha256((ROOT/'data/raw/online_shoppers_intention.csv').read_bytes()).hexdigest()
    assert raw_hash == read('data_manifest.json')['sha256']


def test_exact_saved_splits(study):
    _, frame, splits, _ = study
    parts, again = split_rows(frame, read('data_manifest.json')['sha256'])
    assert again == splits
    combined = np.concatenate(list(parts.values()))
    assert len(set(combined)) == len(frame) == len(combined)
    assert sorted(combined.tolist()) == list(range(len(frame)))
    for key, ids in parts.items():
        assert splits['partitions'][key]['purchases'] == int(frame.loc[ids].Revenue.sum())


def test_preprocessing_fit_isolation_and_unknown_category(study):
    _, frame, splits, _ = study
    ids = splits['partitions']['train']['source_row_ids'][:100]
    small = frame.loc[ids, FEATURES].copy()
    fitted = preprocessor(scale=True).fit(small)
    scaler = fitted.named_transformers_['numeric'].named_steps['scale']
    np.testing.assert_allclose(scaler.mean_, small[NUMERIC].mean())
    unseen = small.iloc[:1].copy()
    unseen['VisitorType'] = 'HOLDOUT_ONLY_CATEGORY'
    unseen['ProductRelated'] = 1e9
    before = scaler.mean_.copy()
    transformed = fitted.transform(unseen)
    assert transformed.shape[0] == 1
    np.testing.assert_array_equal(before, scaler.mean_)
    vocab = fitted.named_transformers_['categorical'].named_steps['onehot'].categories_[-1]
    assert 'HOLDOUT_ONLY_CATEGORY' not in vocab
    assert 'PageValues' not in preprocessor(without_pagevalues=True).transformers[0][2]


def test_hand_checked_metrics_and_equality():
    # Equality at .5 counts as purchase. One of each confusion category.
    m = metrics([1, 0, 1, 0], [.5, .8, .2, .1], .5)
    assert [m[k] for k in ['tp', 'fp', 'tn', 'fn']] == [1, 1, 1, 1]
    assert m['precision'] == m['recall'] == m['f1'] == m['accuracy'] == .5
    zero = metrics([0, 1], [.1, .2], 1)
    assert zero['precision'] == zero['recall'] == zero['f1'] == 0
    assert zero['precision_defined'] is False
    with pytest.raises(ValueError):
        metrics([0], [float('nan')])
    with pytest.raises(ValueError):
        metrics([0], [.4], float('nan'))


def test_validation_selection_and_threshold(study):
    _, _, _, selected = study
    records = pd.read_csv(ROOT/'results/validation_metrics.csv').to_dict('records')
    assert choose_model(records) == selected['selected_model']
    # A more complex model within 0.005 is intentionally not preferred.
    fixture = [{'model': name, 'feature_set': 'full', 'ap': ap} for name, ap in [('Logistic Regression', .70), ('Decision Tree', .703), ('Random Forest', .704)]]
    assert choose_model(fixture) == 'Logistic Regression'
    # An all-zero F1 tie chooses the highest threshold in the declared grid.
    assert choose_threshold([1, 0], [0., 0.])[0] == .95
    predictions = pd.read_csv(ROOT/'results/validation_predictions.csv', float_precision='round_trip')
    sub = predictions[(predictions.model == selected['selected_model']) & (predictions.feature_set == 'full')]
    assert choose_threshold(sub.y_true, sub.purchase_score)[0] == selected['threshold']
    events = read('events.json')
    assert [r['event'] for r in events] == ['splits_persisted', 'decisions_frozen', 'test_evaluation_started']
    assert events[1]['utc'] <= events[2]['utc']


def test_all_bounded_searches_and_no_unresolved_warnings():
    cv = pd.read_csv(ROOT/'results/cv_results.csv')
    counts = cv.groupby(['model', 'feature_set']).size().to_dict()
    assert counts == {('Logistic Regression', 'full'): 3, ('Logistic Regression', 'no_PageValues'): 3, ('Decision Tree', 'full'): 6, ('Random Forest', 'full'): 4, ('Random Forest', 'no_PageValues'): 4}
    assert np.isfinite(cv[['split0_test_score', 'split1_test_score', 'split2_test_score']]).all().all()
    assert not any(w['category'] == 'ConvergenceWarning' for w in read('warnings.json'))


@pytest.mark.parametrize('partition', ['validation', 'test'])
def test_predictions_metrics_and_website_agree(partition, study):
    _, frame, splits, _ = study
    preds = pd.read_csv(ROOT/f'results/{partition}_predictions.csv', float_precision='round_trip')
    saved = pd.read_csv(ROOT/f'results/{partition}_metrics.csv')
    web = json.loads((ROOT/'docs/data/summary.json').read_text())
    assert set(zip(preds.model, preds.feature_set)) == {('Dummy', 'full'), ('Logistic Regression', 'full'), ('Decision Tree', 'full'), ('Random Forest', 'full'), ('Logistic Regression', 'no_PageValues'), ('Random Forest', 'no_PageValues')}
    for _, row in saved.iterrows():
        p = preds[(preds.model == row.model) & (preds.feature_set == row.feature_set) & (preds.threshold == row.threshold)]
        assert p.source_row_id.tolist() == splits['partitions'][partition]['source_row_ids']
        np.testing.assert_array_equal(p.y_true, frame.loc[p.source_row_id, 'Revenue'])
        assert np.isfinite(p.purchase_score).all() and p.purchase_score.between(0, 1).all()
        assert ((p.purchase_score >= row.threshold).astype(int) == p.predicted_label).all()
        calculated = metrics(p.y_true, p.purchase_score, row.threshold)
        exported = next(v for v in web[f'{partition}_metrics'] if v['model'] == row.model and v['feature_set'] == row.feature_set and v['threshold'] == row.threshold)
        for key in ['support', 'purchases', 'tp', 'fp', 'tn', 'fn', 'precision', 'recall', 'f1', 'ap', 'roc_auc', 'accuracy']:
            assert calculated[key] == pytest.approx(row[key], abs=1e-12)
            assert exported[key] == pytest.approx(row[key], abs=1e-12)


def test_fitted_model_only_learned_training_rows(study):
    _, frame, splits, selection = study
    ids = splits['partitions']['train']['source_row_ids']
    # These joblib files were created locally by this project; never load untrusted pickle files.
    pipeline = joblib.load(ROOT/'results/models/Logistic_Regression_full.joblib')
    prep = pipeline.named_steps['preprocess']
    scaler = prep.named_transformers_['numeric'].named_steps['scale']
    np.testing.assert_allclose(scaler.mean_, frame.loc[ids, NUMERIC].mean(), rtol=1e-12)
    vocabularies = prep.named_transformers_['categorical'].named_steps['onehot'].categories_
    for col, vocabulary in zip(CATEGORICAL, vocabularies):
        assert set(vocabulary) == set(frame.loc[ids, col].dropna())
    actual = pd.read_csv(ROOT/'results/test_predictions.csv', float_precision='round_trip')
    sub = actual[(actual.model == 'Logistic Regression') & (actual.feature_set == 'full') & (actual.threshold == .5)]
    np.testing.assert_allclose(pipeline.predict_proba(frame.loc[sub.source_row_id, FEATURES])[:, 1], sub.purchase_score, atol=1e-15)


def test_group_sensitivity_has_no_vector_overlap(study):
    _, frame, _, _ = study
    groups = pd.util.hash_pandas_object(frame[FEATURES], index=False)
    saved = read('group_splits.json')['partitions']
    all_ids = [i for part in saved.values() for i in part['source_row_ids']]
    assert sorted(all_ids) == list(range(len(frame)))
    for a, b in [('train', 'test'), ('train', 'validation'), ('validation', 'test')]:
        assert not (set(groups.loc[saved[a]['source_row_ids']]) & set(groups.loc[saved[b]['source_row_ids']]))


def test_site_payload_exact_scores_and_required_sections(study):
    _, _, _, selection = study
    payload = json.loads((ROOT/'docs/data/threshold_validation.json').read_text())
    predictions = pd.read_csv(ROOT/'results/validation_predictions.csv', float_precision='round_trip')
    sub = predictions[(predictions.model == selection['selected_model']) & (predictions.feature_set == 'full')]
    assert payload['partition'] == 'validation'
    assert payload['source_row_id'] == sub.source_row_id.tolist()
    assert payload['purchase_score'] == sub.purchase_score.tolist()
    html = (ROOT/'docs/index.html').read_text(encoding='utf-8')
    assert '{{' not in html
    for section in ['overview', 'data', 'preprocessing', 'methods', 'evaluation', 'results', 'mistakes', 'threshold', 'conclusions', 'reproduce']:
        assert f'id="{section}"' in html
