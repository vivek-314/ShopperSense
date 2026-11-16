"""Read immutable source bytes, audit quality, and persist source-row partitions."""
import hashlib
import platform
from datetime import datetime, timezone
from importlib.metadata import version

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

from .config import CATEGORICAL, FEATURES, NUMERIC, SEED, SOURCE_URL


def binary(series):
    mapping = {'true': 1, 'false': 0, '1': 1, '0': 0}
    values = series.astype(str).str.lower().map(mapping)
    if values.isna().any():
        raise ValueError(f'{series.name} must contain only explicit true/false or 0/1 values')
    return values.astype(int)


def load_data(path):
    raw = pd.read_csv(path)
    expected = set(FEATURES + ['Revenue'])
    if set(raw.columns) != expected:
        raise ValueError(f'Schema mismatch: missing={expected-set(raw.columns)}, extra={set(raw.columns)-expected}')
    frame = raw.copy()
    frame['Revenue'] = binary(raw['Revenue'])
    frame['Weekend'] = binary(raw['Weekend'])
    for col in CATEGORICAL:
        frame[col] = raw[col].map(lambda x: str(x) if pd.notna(x) else np.nan).astype(object)
    for col in NUMERIC:
        frame[col] = pd.to_numeric(frame[col], errors='raise')
        if np.isinf(frame[col]).any():
            raise ValueError(f'Nonfinite value in {col}; investigate source before training')
    frame.index.name = 'source_row_id'
    return raw, frame


def inspect_data(path, raw, frame):
    checksum = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {
        'source_filename': path.name, 'sha256': checksum,
        'source': SOURCE_URL, 'doi': '10.24432/C5F88Q',
        'attribution': 'Sakar, C. & Kastro, Y. (2018). Online Shoppers Purchasing Intention Dataset. UCI Machine Learning Repository.',
        'license': 'CC BY 4.0', 'license_url': 'https://creativecommons.org/licenses/by/4.0/',
        'provenance': 'Supplied locally; original download date unknown. Raw bytes preserved.',
        'source_metadata_verified_on': '2026-11-16',
        'inspection_utc': datetime.now(timezone.utc).isoformat(),
        'rows': len(raw), 'predictors': len(FEATURES), 'columns': raw.columns.tolist(),
        'dtypes': {c: str(t) for c, t in raw.dtypes.items()},
        'target_mapping': {'FALSE': 0, 'TRUE': 1},
        'class_counts': {str(k): int(v) for k, v in frame.Revenue.value_counts().sort_index().items()},
        'environment': {'python': platform.python_version(), **{p: version(p) for p in ['numpy', 'pandas', 'scikit-learn', 'matplotlib', 'jupyter', 'pytest', 'joblib']}},
    }
    numeric = frame[NUMERIC]
    # Code ranges are the source's documented category inventories, not ordered values.
    allowed = {
        'Month': ['Feb', 'Mar', 'May', 'June', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
        'VisitorType': ['Returning_Visitor', 'New_Visitor', 'Other'],
        'OperatingSystems': list(map(str, range(1, 9))), 'Browser': list(map(str, range(1, 14))),
        'Region': list(map(str, range(1, 10))), 'TrafficType': list(map(str, range(1, 21))),
    }
    quality = {
        'scope': 'Whole-file structural audit; exploratory choices use training only.',
        'missing_by_column': raw.isna().sum().astype(int).to_dict(),
        'nonfinite_by_numeric_column': {c: int(np.isinf(numeric[c]).sum()) for c in NUMERIC},
        'negative_by_numeric_column': (numeric < 0).sum().astype(int).to_dict(),
        'outside_unit_interval': {c: int(((frame[c] < 0) | (frame[c] > 1)).sum()) for c in ['BounceRates', 'ExitRates', 'SpecialDay', 'Weekend']},
        'noninteger_counts': {c: int((frame[c].dropna() % 1 != 0).sum()) for c in ['Administrative', 'Informational', 'ProductRelated']},
        'categories': {c: sorted(frame[c].dropna().unique().tolist()) for c in CATEGORICAL},
        'unexpected_categories': {c: sorted(set(frame[c].dropna()) - set(allowed[c])) for c in CATEGORICAL},
        'numeric_summary': numeric.describe(percentiles=[.5, .95, .99]).to_dict(),
        'complete_duplicate_rows': int(raw.duplicated().sum()),
        'identical_predictor_extra_rows': int(frame[FEATURES].duplicated().sum()),
        'duplicate_decision': 'Preserve all source rows: repeated observed sessions are not automatically erroneous. Quantify cross-partition overlap and also report a group-aware sensitivity.',
        'outlier_decision': 'Retain extreme values. No automatic clipping or removal. Training summaries and intensity distributions show skew.',
        'source_expectations_match': len(raw) == 12330 and frame.Revenue.sum() == 1908 and len(FEATURES) == 17,
    }
    return manifest, quality


def split_rows(frame, checksum):
    train_val, test = train_test_split(frame.index.to_numpy(), test_size=.15, stratify=frame.Revenue, random_state=SEED)
    train, validation = train_test_split(train_val, test_size=15/85, stratify=frame.loc[train_val, 'Revenue'], random_state=SEED)
    parts = {'train': train, 'validation': validation, 'test': test}
    result = {'seed': SEED, 'source_sha256': checksum, 'method': 'stratified 70/15/15; test=.15 then validation=15/85', 'partitions': {}}
    for name, ids in parts.items():
        result['partitions'][name] = {'source_row_ids': ids.tolist(), 'support': len(ids), 'purchases': int(frame.loc[ids, 'Revenue'].sum())}
    joined = np.concatenate(list(parts.values()))
    assert len(joined) == len(frame) == len(np.unique(joined))
    return parts, result


def duplicate_audit(frame, parts):
    groups = pd.util.hash_pandas_object(frame[FEATURES], index=False)
    overlaps = []
    for a, b in [('train', 'validation'), ('train', 'test'), ('validation', 'test')]:
        shared = set(groups.loc[parts[a]]) & set(groups.loc[parts[b]])
        a_rows = frame.loc[parts[a]].loc[groups.loc[parts[a]].isin(shared)]
        b_rows = frame.loc[parts[b]].loc[groups.loc[parts[b]].isin(shared)]
        overlaps.append({'a': a, 'b': b, 'shared_vectors': len(shared), 'a_rows': len(a_rows), 'b_rows': len(b_rows), 'a_purchases': int(a_rows.Revenue.sum()), 'b_purchases': int(b_rows.Revenue.sum())})
    # Always run the secondary split so even a small overlap is visible to the reader.
    folds = list(StratifiedGroupKFold(n_splits=20, shuffle=True, random_state=SEED).split(frame[FEATURES], frame.Revenue, groups))
    ids = [fold[1] for fold in folds]
    group_parts = {'train': np.concatenate(ids[:14]), 'validation': np.concatenate(ids[14:17]), 'test': np.concatenate(ids[17:])}
    for a, b in [('train', 'validation'), ('train', 'test'), ('validation', 'test')]:
        assert not (set(groups.loc[group_parts[a]]) & set(groups.loc[group_parts[b]]))
    return {'pairs': overlaps, 'group_sensitivity_method': '20 stratified group folds: 14 train, 3 validation, 3 test; identical predictor hashes kept together. Approximate 70/15/15; secondary only. Frozen primary hyperparameters and threshold reused; no group-test tuning.'}, group_parts
