from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
NUMERIC = [
    'Administrative', 'Administrative_Duration', 'Informational',
    'Informational_Duration', 'ProductRelated', 'ProductRelated_Duration',
    'BounceRates', 'ExitRates', 'PageValues', 'SpecialDay', 'Weekend',
]
CATEGORICAL = ['Month', 'OperatingSystems', 'Browser', 'Region', 'TrafficType', 'VisitorType']
FEATURES = NUMERIC + CATEGORICAL
MODEL_ORDER = ['Logistic Regression', 'Decision Tree', 'Random Forest']
SOURCE_URL = 'https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset'
