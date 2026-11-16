from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import CATEGORICAL, NUMERIC


def preprocessor(scale=False, without_pagevalues=False):
    numeric = [c for c in NUMERIC if not (without_pagevalues and c == 'PageValues')]
    steps = [('imputer', SimpleImputer(strategy='median'))]
    if scale:
        steps.append(('scale', StandardScaler()))
    categories = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='__missing__')),
        ('onehot', OneHotEncoder(handle_unknown='ignore')),
    ])
    return ColumnTransformer([
        ('numeric', Pipeline(steps), numeric),
        ('categorical', categories, CATEGORICAL),
    ], remainder='drop')
