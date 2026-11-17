import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import joblib

from src.config import ROOT
from src.data import load_data
from src.evaluate import choose_model, choose_threshold, metrics
from src.predictor.train import FEATURES, NUMERIC


def read(path):
    return json.loads((ROOT/path).read_text(encoding='utf-8'))


def test_product_model_has_only_observable_explicit_inputs():
    model=read('docs/data/predictor.json')
    selection=read('results/predictor/selection.json')
    assert selection['features']==FEATURES
    assert len(FEATURES)==6
    assert not {'Administrative','Administrative_Duration','Revenue','PageValues','BounceRates','ExitRates','source_row_id'} & set(FEATURES)
    assert model['feature_set']=='six_observable_inputs'
    assert model['run_id']==selection['run_id']
    assert model['data_sha256']==read('results/splits.json')['source_sha256']


def test_product_freeze_and_test_metrics():
    selection=read('results/predictor/selection.json')
    validation=pd.read_csv(ROOT/'results/predictor/validation_metrics.csv')
    assert choose_model(validation.to_dict('records'))==selection['model']
    predictions=pd.read_csv(ROOT/'results/predictor/predictions.csv',float_precision='round_trip')
    val=predictions[predictions.partition=='validation']
    assert choose_threshold(val.y_true,val.purchase_score)[0]==selection['threshold']
    test=predictions[predictions.partition=='test']
    exported=read('docs/data/predictor.json')['test_metrics']
    actual=metrics(test.y_true,test.purchase_score,selection['threshold'])
    for key in ['tp','fp','tn','fn','precision','recall','f1','ap','accuracy']:
        assert actual[key]==pytest.approx(exported[key],abs=1e-12)
    assert 'reused' in selection['test_status']


def test_product_preprocessing_learned_only_training():
    _,frame=load_data(ROOT/'data/raw/online_shoppers_intention.csv')
    train_ids=read('results/splits.json')['partitions']['train']['source_row_ids']
    pipeline=joblib.load(ROOT/'results/models/marketer_predictor.joblib')
    prep=pipeline.named_steps['preprocess']
    np.testing.assert_allclose(prep.named_transformers_['numeric'].named_steps['impute'].statistics_,frame.loc[train_ids,NUMERIC].median())
    assert set(prep.named_transformers_['category'].named_steps['onehot'].categories_[0])==set(frame.loc[train_ids,'VisitorType'])
    cv=pd.read_csv(ROOT/'results/predictor/cv_results.csv')
    assert cv.groupby('model').size().to_dict()=={'Decision Tree':6,'Logistic Regression':3,'Random Forest':4}
    assert not any(w['category']=='ConvergenceWarning' for w in read('results/predictor/warnings.json'))


def test_product_scores_stable_order_and_export_fixtures():
    split=read('results/splits.json')['partitions']
    predictions=pd.read_csv(ROOT/'results/predictor/predictions.csv')
    for partition in ['train','validation','test']:
        sub=predictions[predictions.partition==partition]
        assert sub.source_row_id.tolist()==split[partition]['source_row_ids']
        assert sub.purchase_score.between(0,1).all() and np.isfinite(sub.purchase_score).all()
    fixtures=read('results/predictor/parity_fixtures.json')
    assert len(fixtures)==3703
    assert sum(row['name'].startswith('new-input:') for row in fixtures)==3
    index=(ROOT/'docs/index.html').read_text(encoding='utf-8')
    assert 'session-form' in index and 'csv-file' in index
    assert 'href="#research"' in index
    assert 'name="account_pages"' not in index and 'name="account_seconds"' not in index
    assert '03_six_input_models_step_by_step.ipynb' in index
