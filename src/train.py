import warnings

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from .config import SEED
from .preprocessing import preprocessor


def definitions():
    return {
        'Dummy': (DummyClassifier(strategy='most_frequent', random_state=SEED), {}),
        'Logistic Regression': (LogisticRegression(max_iter=2000, random_state=SEED), {'model__C': [.1, 1, 10]}),
        'Decision Tree': (DecisionTreeClassifier(random_state=SEED), {'model__max_depth': [3, 5, 8], 'model__min_samples_leaf': [5, 20]}),
        'Random Forest': (RandomForestClassifier(n_estimators=150, random_state=SEED, n_jobs=1), {'model__max_depth': [8, None], 'model__min_samples_leaf': [2, 10]}),
    }


def train_variant(name, feature_set, train_x, train_y, jobs=1):
    estimator, grid = definitions()[name]
    pipeline = Pipeline([('preprocess', preprocessor(name == 'Logistic Regression', feature_set == 'no_PageValues')), ('model', estimator)])
    # Sequential by default: warnings from each fold remain observable in this process.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        if name == 'Dummy':
            pipeline.fit(train_x, train_y)
            candidates, params = pd.DataFrame(), {}
        else:
            search = GridSearchCV(pipeline, grid, scoring='average_precision', cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED), refit=True, n_jobs=jobs, error_score='raise', return_train_score=True)
            search.fit(train_x, train_y)
            pipeline, params = search.best_estimator_, search.best_params_
            candidates = pd.DataFrame(search.cv_results_)
    records = [{'category': w.category.__name__, 'message': str(w.message), 'model': name, 'feature_set': feature_set} for w in caught]
    if any(issubclass(w.category, ConvergenceWarning) for w in caught):
        raise RuntimeError(f'Unresolved convergence warning in {name}; see configuration before publishing')
    if not candidates.empty:
        candidates.insert(0, 'model', name)
        candidates.insert(1, 'feature_set', feature_set)
    return pipeline, params, candidates, records
