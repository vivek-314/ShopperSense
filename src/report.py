"""Generate static evidence, figures, and the readable report from saved results."""
import html
import json
import shutil
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

from .config import ROOT

COLORS = ['#263b42', '#ba694b', '#397b68']


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def fmt(value, digits=3):
    return 'undefined' if value is None else f'{value:.{digits}f}'


def pct(value):
    return f'{value:.1%}'


def table(headers, rows, caption='', classes=''):
    head = ''.join(f'<th scope="col">{html.escape(str(h))}</th>' for h in headers)
    body = ''.join('<tr>'+''.join(f'<{"th scope=\"row\"" if i == 0 else "td"}>{html.escape(str(v))}</{"th" if i == 0 else "td"}>' for i, v in enumerate(row))+'</tr>' for row in rows)
    return f'<div class="table-wrap {classes}" tabindex="0" role="region" aria-label="{html.escape(caption)}"><table><caption>{html.escape(caption)}</caption><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def figure(name, caption, alt):
    return f'<figure><img src="assets/{name}.svg" alt="{html.escape(alt)}" loading="lazy" width="900" height="540"><figcaption>{caption}</figcaption></figure>'


def make_figures(output, docs, frame, parts, summary):
    dest = output/'figures'
    dest.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': '#c6ccc5', 'axes.labelcolor': '#344b4c', 'text.color': '#243b3d', 'xtick.color': '#526463', 'ytick.color': '#526463', 'figure.facecolor': '#fbfcf8', 'axes.facecolor': '#fbfcf8', 'svg.fonttype': 'none'})

    def save(name, fig):
        fig.tight_layout(pad=1.5)
        fig.savefig(dest/f'{name}.svg', metadata={'Date': None}, bbox_inches='tight')
        fig.savefig(dest/f'{name}.png', dpi=160, bbox_inches='tight')
        shutil.copyfile(dest/f'{name}.svg', docs/'assets'/f'{name}.svg')
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.4))
    counts = frame.Revenue.value_counts().reindex([0, 1])
    ax.barh(['No purchase', 'Purchase'], counts, color=['#cdd5cb', '#397b68'], height=.5)
    for i, count in enumerate(counts):
        ax.text(count+100, i, f'{count:,} ({count/len(frame):.1%})', va='center')
    ax.set_xlim(0, counts.max()*1.3)
    ax.set_xlabel('Sessions · whole-file structural count')
    ax.invert_yaxis()
    save('class_balance', fig)

    train = frame.loc[parts['train']]
    visitor = train.groupby('VisitorType').Revenue.agg(['size', 'sum', 'mean']).reset_index()
    visitor.to_csv(output/'eda_visitor_type.csv', index=False)
    fig, ax = plt.subplots(figsize=(8, 4.4))
    ax.bar(visitor.VisitorType.str.replace('_', ' '), visitor['mean'], color=COLORS, width=.5)
    for i, row in visitor.iterrows():
        ax.text(i, row['mean']+.008, f'{row["sum"]:,}/{row["size"]:,} · {row["mean"]:.1%}', ha='center', fontsize=10)
    ax.set_ylim(0, max(visitor['mean'])*1.3)
    ax.set_ylabel('Purchase rate · training only')
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
    save('visitor_rates', fig)

    intensity_rows = []
    fig, ax = plt.subplots(figsize=(8, 4.4))
    bins = np.linspace(0, np.log1p(train.ProductRelated.max()), 32)
    for y, color in [(0, '#7a9393'), (1, '#397b68')]:
        subset = train[train.Revenue == y].ProductRelated
        ax.hist(np.log1p(subset), bins=bins, density=True, alpha=.5, color=color, label=f'{"Purchase" if y else "No purchase"} (n={len(subset):,})')
        intensity_rows.append({'Revenue': y, 'support': len(subset), 'median': float(subset.median()), 'p95': float(subset.quantile(.95)), 'max': int(subset.max())})
    ax.set_xlabel('log(1 + product-related pages) · training only')
    ax.set_ylabel('Density within each target class')
    ax.legend(frameon=False)
    save('browsing_intensity', fig)
    pd.DataFrame(intensity_rows).to_csv(output/'eda_intensity.csv', index=False)

    predictions = pd.read_csv(output/'test_predictions.csv')
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for name, color in zip(['Logistic Regression', 'Decision Tree', 'Random Forest'], COLORS):
        sub = predictions[(predictions.model == name) & (predictions.feature_set == 'full') & (predictions.threshold == .5)]
        precision, recall, _ = precision_recall_curve(sub.y_true, sub.purchase_score)
        ap = next(r['ap'] for r in summary['test_metrics'] if r['model'] == name and r['feature_set'] == 'full' and r['threshold'] == .5)
        ax.step(recall, precision, where='post', color=color, label=f'{name} · AP {ap:.3f}', lw=1.8)
    prevalence = predictions[(predictions.model == 'Dummy')].y_true.mean()
    ax.axhline(prevalence, color='#83908a', ls='--', label=f'Purchase prevalence {prevalence:.3f}')
    ax.set(xlabel='Recall · purchase class', ylabel='Precision · purchase class', xlim=(0, 1), ylim=(0, 1.04))
    ax.legend(frameon=False, fontsize=9, loc='lower left')
    save('precision_recall', fig)

    selected = summary['selection']['selected_model']
    threshold = summary['selection']['threshold']
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
    for ax, cut in zip(axes, [.5, threshold]):
        r = next(r for r in summary['test_metrics'] if r['model'] == selected and r['feature_set'] == 'full' and r['threshold'] == cut)
        matrix = [[r['tn'], r['fp']], [r['fn'], r['tp']]]
        ax.imshow(matrix, cmap='Greens', vmin=0, vmax=max(r['tn'], r['tp']))
        for y in range(2):
            for x in range(2):
                ax.text(x, y, f'{["TN", "FP", "FN", "TP"][y*2+x]}\n{matrix[y][x]:,}', ha='center', va='center', fontsize=14, color='white' if y == 0 and x == 0 else '#173f35')
        ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=['No purchase', 'Purchase'], yticklabels=['No purchase', 'Purchase'], xlabel='Predicted label', ylabel='Actual label', title=f'Threshold {cut:.2f}')
    save('confusion', fig)

    depth = pd.read_csv(output/'tree_depth_comparison.csv')
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for partition, color in [('train', '#397b68'), ('validation', '#ba694b')]:
        sub = depth[depth.partition == partition]
        ax.plot(sub.max_depth, sub.f1, marker='o', label=partition.title(), color=color, lw=2)
    ax.set(xlabel='Tree maximum depth · min_samples_leaf=5', ylabel='Purchase F1 · fixed threshold 0.50', ylim=(0, 1))
    ax.legend(frameon=False)
    save('tree_depth', fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for offset, feature_set, color, label in [(-.18, 'full', '#397b68', 'Full features'), (.18, 'no_PageValues', '#ba694b', 'Without PageValues')]:
        values = [next(r['ap'] for r in summary['test_metrics'] if r['model'] == name and r['feature_set'] == feature_set and r['threshold'] == .5) for name in ['Logistic Regression', 'Random Forest']]
        bars = ax.bar(np.arange(2)+offset, values, .34, color=color, label=label)
        ax.bar_label(bars, fmt='%.3f', padding=4)
    ax.set(xticks=[0, 1], xticklabels=['Logistic Regression', 'Random Forest'], ylabel='Average precision · held-out test', ylim=(0, 1))
    ax.legend(frameon=False)
    save('pagevalues', fig)


def generate_report(output, docs, frame, parts):
    docs.mkdir(exist_ok=True)
    (docs/'assets').mkdir(exist_ok=True)
    (docs/'data').mkdir(exist_ok=True)
    selection = read_json(output/'selection.json')
    summary = {key: read_json(output/f'{key}.json') for key in ['data_manifest', 'quality', 'selection', 'error_analysis', 'run']}
    summary.update({name: pd.read_csv(output/f'{name}.csv').to_dict('records') for name in ['validation_metrics', 'test_metrics', 'group_sensitivity']})
    summary['split_counts'] = {key: {'support': len(ids), 'purchases': int(frame.loc[ids, 'Revenue'].sum())} for key, ids in parts.items()}
    from .run_all import write_json
    write_json(docs/'data/summary.json', summary)
    predictions = pd.read_csv(output/'validation_predictions.csv', float_precision='round_trip')
    selected = selection['selected_model']
    sub = predictions[(predictions.model == selected) & (predictions.feature_set == 'full')]
    threshold_data = {'run_id': selection['run_id'], 'model': selected, 'feature_set': 'full', 'partition': 'validation', 'chosen_threshold': selection['threshold'], 'source_row_id': sub.source_row_id.tolist(), 'y_true': sub.y_true.tolist(), 'purchase_score': sub.purchase_score.tolist(), 'score_label': 'Uncalibrated model score'}
    write_json(docs/'data/threshold_validation.json', threshold_data)
    make_figures(output, docs, frame, parts, summary)
    manifest, quality, errors = summary['data_manifest'], summary['quality'], summary['error_analysis']
    cut = selection['threshold']
    main = next(r for r in summary['test_metrics'] if r['model'] == selected and r['feature_set'] == 'full' and r['threshold'] == cut)
    baseline = next(r for r in summary['test_metrics'] if r['model'] == 'Dummy')
    val = next(r for r in summary['validation_metrics'] if r['model'] == selected and r['feature_set'] == 'full')
    best_test = max((r for r in summary['test_metrics'] if r['feature_set'] == 'full' and r['threshold'] == .5 and r['model'] != 'Dummy'), key=lambda r: r['ap'])
    repo = read_json(ROOT/'site_config.json').get('repository_url')
    if repo and not repo.startswith('https://github.com/'):
        raise ValueError('Configure a real https://github.com/ repository URL or leave null')
    replacements = {
        'run_id': selection['run_id'], 'hash_prefix': manifest['sha256'][:12], 'data_hash': manifest['sha256'],
        'selected_model': selected, 'chosen_threshold': f'{cut:.2f}', 'test_ap': fmt(main['ap']), 'test_f1': fmt(main['f1']), 'test_recall': pct(main['recall']),
        'session_count': f'{len(frame):,}', 'purchase_count': f'{int(frame.Revenue.sum()):,}', 'purchase_rate': pct(frame.Revenue.mean()),
        'brief_answer': f'{selected}, selected on validation, achieved {main["ap"]:.3f} average precision on the held-out test with full features, compared with {baseline["ap"]:.3f} for the majority baseline. At the frozen {cut:.2f} threshold, it detected {main["tp"]} of {main["purchases"]} purchases, with {main["fp"]} false alarms.',
        'quality_summary': f'{sum(quality["missing_by_column"].values())} missing cells · {sum(quality["nonfinite_by_numeric_column"].values())} infinite values · {sum(quality["negative_by_numeric_column"].values())} negative numeric values · {quality["complete_duplicate_rows"]} complete duplicates preserved.',
        'quality_details': table(['Check · whole file', 'Observed finding / decision'], [
            ['Schema & target', '12,330 rows, 17 predictors + binary Revenue; source expectations match.' if quality['source_expectations_match'] else 'Supplied file differs from source expectations; see manifest.'],
            ['Missing / infinite / negative', f'{sum(quality["missing_by_column"].values())} / {sum(quality["nonfinite_by_numeric_column"].values())} / {sum(quality["negative_by_numeric_column"].values())} numeric negative values'],
            ['Invalid values', f'{sum(quality["outside_unit_interval"].values())} values outside [0,1] in rates, SpecialDay or Weekend; {sum(quality["noninteger_counts"].values())} noninteger page counts; {sum(len(v) for v in quality["unexpected_categories"].values())} unexpected categories'],
            ['Repeated observations', f'{quality["complete_duplicate_rows"]} extra complete duplicates; {quality["identical_predictor_extra_rows"]} extra identical predictor vectors. All retained.'],
            ['Extreme browsing', f'ProductRelated max {quality["numeric_summary"]["ProductRelated"]["max"]:,.0f} pages; duration max {quality["numeric_summary"]["ProductRelated_Duration"]["max"]:,.1f}. Descriptive whole-file audit; no clipping.'],
            ['Provenance', 'Supplied locally; original download date unknown. SHA-256 recorded; original bytes preserved.'],
        ], 'Measured data quality'),
        'class_figure': figure('class_balance', f'Whole-file class distribution. Separately, the majority baseline achieves {baseline["accuracy"]:.1%} on the test partition while detecting zero purchases. The imbalance makes purchase-focused metrics necessary.', f'Whole file: {len(frame)-int(frame.Revenue.sum()):,} non-purchase and {int(frame.Revenue.sum()):,} purchase sessions.'),
        'visitor_figure': figure('visitor_rates', 'Training only. Labels show purchases / sessions and the observed rate. These are associations, and the small Other group deserves caution.', 'Training purchase rates by VisitorType; numeric denominators appear in the following table.'),
        'intensity_figure': figure('browsing_intensity', 'Training only. Both distributions are normalized within class; log(1 + pages) makes skew visible without dropping large sessions. Overlap means page count alone cannot perfectly separate purchases.', 'Training distributions of log transformed product-related page counts by target; numeric summaries follow.'),
        'split_table': table(['Partition', 'Sessions', 'Purchases', 'Purpose'], [[p.title(), f'{r["support"]:,}', f'{r["purchases"]:,}', {'train': 'Fit preprocessing and tune with 3-fold CV', 'validation': 'Select model and threshold', 'test': 'Final frozen comparison only'}[p]] for p, r in summary['split_counts'].items()], 'Exact stratified split · random_state = 42'),
        'selection_text': f'{selected} was selected with validation AP {val["ap"]:.3f}. We take the highest validation AP, then prefer the simpler model when it is within 0.005 of the highest: Logistic Regression → Decision Tree → Random Forest. This practical tolerance is not a significance test. The validation F1 grid selected {cut:.2f}; ties use the larger threshold. All six fitted variants and this decision were frozen before any test predictions. No model was refitted on validation.',
        'validation_table': metric_table(summary['validation_metrics'], 'Validation selection · all prespecified variants · threshold 0.50'),
        'test_table': metric_table(summary['test_metrics'], 'Final held-out test · full and no-PageValues variants'),
        'test_ranking_text': model_comparison_text(summary, best_test, selected),
        'pr_figure': figure('precision_recall', 'Held-out test, full features. Curves show precision/recall over scores; the table reports average precision (AP), not trapezoidal area under this curve. A constant-score reference has AP equal to purchase prevalence.', 'Test purchase precision-recall curves for Logistic Regression, Decision Tree, and Random Forest. AP values are in the comparison table.'),
        'confusion_figure': figure('confusion', f'Held-out test, {selected}, full features. The left panel uses 0.50; the right uses the frozen validation-selected {cut:.2f}. Counts are also listed in the text and table.', f'Selected model confusion counts at thresholds 0.50 and {cut:.2f}; text table follows.'),
        'confusion_table': table(['Test threshold', 'TN', 'FP', 'FN', 'TP', 'Recall', 'Precision'], [[f'{r["threshold"]:.2f}', r['tn'], r['fp'], r['fn'], r['tp'], pct(r['recall']), pct(r['precision'])] for r in summary['test_metrics'] if r['model'] == selected and r['feature_set'] == 'full'], f'{selected} · full features · test confusion counts'),
        'depth_figure': figure('tree_depth', 'Training versus validation, full features, min_samples_leaf=5 and threshold fixed at 0.50. Deeper trees can fit training details that do not carry to validation; this diagnostic did not alter the declared selection protocol.', 'Decision Tree train and validation F1 at depths 3, 5, 8, 12 and unbounded; exact values follow.'),
        'depth_table': table(['Maximum depth', 'Train F1', 'Validation F1'], [[d, fmt(g[g.partition == 'train'].iloc[0].f1), fmt(g[g.partition == 'validation'].iloc[0].f1)] for d, g in pd.read_csv(output/'tree_depth_comparison.csv').groupby('max_depth', sort=False)], 'Depth diagnostic · full features · fixed threshold 0.50'),
        'error_text': f'For {selected}, full features, test at {cut:.2f}: false negatives missed {main["fn"]}/{main["purchases"]} actual buyers ({main["fn"]/main["purchases"]:.1%}); false positives flagged {main["fp"]}/{main["support"]-main["purchases"]} non-buyers ({main["fp"]/(main["support"]-main["purchases"]):.1%}). Of {main["predicted_purchases"]} predicted buyers, {main["fp"]} did not purchase. Errors describe recorded outcomes, not visitor motives.',
        'error_examples': example_tables(errors),
        'subgroup_table': table(['Grouping', 'Group', 'Sessions', 'Buyers', 'FP / non-buyers', 'FN / buyers', 'Precision', 'Recall'], [[r['grouping'], r['group'], r['support'], r['purchases'], f'{r["fp"]} / {r["support"]-r["purchases"]}', f'{r["fn"]} / {r["purchases"]}', fmt(r['precision']), fmt(r['recall'])] for r in errors['subgroups']], f'{selected} · full features · test at {cut:.2f} · subgroup support'),
        'bin_text': '; '.join(f'{key}: {errors["bin_boundaries"][key]}' for key in ['low', 'medium', 'high']) + '. Boundaries come from training tertiles and were reused on test.',
        'overlap_table': table(['Partitions', 'Shared vectors', 'Rows in first', 'Rows in second', 'Buyers in first / second'], [[f'{r["a"]} ↔ {r["b"]}', r['shared_vectors'], r['a_rows'], r['b_rows'], f'{r["a_purchases"]} / {r["b_purchases"]}'] for r in quality['cross_partition_duplicate_audit']['pairs']], 'Identical predictor overlap in the primary split'),
        'group_table': metric_table(summary['group_sensitivity'], 'Secondary group-aware split · selected primary model / parameters / threshold reused'),
        'group_text': group_interpretation(summary),
        'pagevalues_figure': figure('pagevalues', 'Held-out test, threshold 0.50. Both feature conditions used the same primary row partitions and the same small training-only parameter searches. Full features were prespecified primary before test access.', 'Test AP with and without PageValues for Logistic Regression and Random Forest; exact values are in the comparison tables.'),
        'pagevalues_text': sensitivity_text(summary),
        'conclusion': f'In this historical dataset, recorded session summaries identify purchases better than always predicting no purchase. Validation-selected {selected} with full features achieved held-out test AP {main["ap"]:.3f} versus baseline {baseline["ap"]:.3f}, and purchase F1 {main["f1"]:.3f} at the frozen {cut:.2f} threshold. It still missed {main["fn"]} of {main["purchases"]} test purchases. This supports session-level discrimination on this split, with clear dependence on the recorded feature set; it does not establish early-session utility or conversion uplift.',
        'repo_access': f'<a class="button secondary" href="{html.escape(repo, quote=True)}">View repository ↗</a>' if repo else '<p class="muted">Repository URL is not configured. The complete source and evidence are available in the download below. Public deployment is pending.</p>',
        'runtime': f'{summary["run"]["training_and_evaluation_seconds"]:.1f}',
    }
    visitor = pd.read_csv(output/'eda_visitor_type.csv')
    replacements['visitor_table'] = table(['Visitor type', 'Training sessions', 'Purchases', 'Purchase rate'], [[r.VisitorType, r['size'], r['sum'], pct(r['mean'])] for _, r in visitor.iterrows()], 'Training-only purchase rates with denominators')
    intensity = pd.read_csv(output/'eda_intensity.csv')
    replacements['intensity_table'] = table(['Training target', 'Sessions', 'Median pages', '95th percentile', 'Maximum'], [['Purchase' if r.Revenue else 'No purchase', int(r.support), f'{r["median"]:g}', f'{r.p95:g}', int(r['max'])] for _, r in intensity.iterrows()], 'Training-only product page count summaries')
    template = (ROOT/'src/report_template.html').read_text(encoding='utf-8')
    for key, value in replacements.items():
        template = template.replace('{{'+key+'}}', str(value))
    if '{{' in template:
        raise ValueError('Unresolved report template variable')
    (docs/'index.html').write_text(template, encoding='utf-8')
    (docs/'.nojekyll').touch()
    for name in ['data_manifest.json', 'quality.json', 'selection.json', 'test_metrics.csv', 'validation_metrics.csv', 'pagevalues_comparison.csv', 'error_analysis.json', 'cv_results.csv', 'splits.json', 'threshold_grid.csv']:
        shutil.copyfile(output/name, docs/'data'/name)
    write_json(output/'website_export.json', {'run_id': selection['run_id'], 'summary': 'docs/data/summary.json', 'threshold_data': 'docs/data/threshold_validation.json', 'generated_from_saved_artifacts': True})
    make_bundle(output, docs)


def metric_table(records, caption):
    return table(['Model / features', 'Partition · threshold', 'n / buyers', 'AP', 'ROC-AUC', 'Precision', 'Recall', 'F1', 'Accuracy', 'TN / FP / FN / TP'], [[f'{r["model"]} · {r["feature_set"]}', f'{r["partition"]} · {r["threshold"]:.2f}', f'{r["support"]} / {r["purchases"]}', *[fmt(r[k]) for k in ['ap', 'roc_auc', 'precision', 'recall', 'f1', 'accuracy']], f'{r["tn"]} / {r["fp"]} / {r["fn"]} / {r["tp"]}'] for r in records], caption)


def model_comparison_text(summary, best_test, selected):
    values = {r['model']: r for r in summary['test_metrics'] if r['feature_set'] == 'full' and r['threshold'] == .5}
    linear = values['Logistic Regression']['ap']
    return f'The highest test AP among full-feature learned models belongs to {best_test["model"]} ({best_test["ap"]:.3f}). Decision Tree improves test AP over Logistic Regression by {values["Decision Tree"]["ap"]-linear:.3f}; Random Forest by {values["Random Forest"]["ap"]-linear:.3f}. These are descriptive differences on this split, without significance or operational-value claims. The selected model remains {selected}; test rankings never revise the validation decision.'


def example_tables(errors):
    content = ''
    for error, rows in errors['examples'].items():
        content += f'<h4>{"False positives · predicted buyers who did not purchase" if error == "FP" else "False negatives · missed actual buyers"}</h4>'
        content += table(['Source row ID', 'Visitor type', 'Product pages', 'Product duration', 'PageValues', 'Actual', 'Score', 'Predicted'], [[r['source_row_id'], r['VisitorType'], r['ProductRelated'], f'{r["ProductRelated_Duration"]:.1f}', f'{r["PageValues"]:.2f}', r['Revenue'], fmt(r['purchase_score'], 6), r['predicted_label']] for r in rows], f'{error} examples · first four source row IDs') if rows else '<p>No errors in this category.</p>'
    return content


def sensitivity_text(summary):
    texts = []
    for name in ['Logistic Regression', 'Random Forest']:
        pieces = []
        for partition in ['validation', 'test']:
            records = summary[f'{partition}_metrics']
            full = next(r['ap'] for r in records if r['model'] == name and r['feature_set'] == 'full' and r['threshold'] == .5)
            reduced = next(r['ap'] for r in records if r['model'] == name and r['feature_set'] == 'no_PageValues' and r['threshold'] == .5)
            pieces.append(f'{partition} AP {full:.3f} → {reduced:.3f} (change {reduced-full:+.3f})')
        texts.append(name+': '+ '; '.join(pieces)+'.')
    return ' '.join(texts)+' This measures dependence on PageValues, not proof of target leakage. Removing it also does not establish real-time availability of the remaining session summaries.'


def group_interpretation(summary):
    record = next(r for r in summary['group_sensitivity'] if r['partition'] == 'group_test')
    return f'The secondary group test has {record["support"]} sessions and {record["purchases"]} purchases; AP {record["ap"]:.3f}, F1 {record["f1"]:.3f}. Twenty stratified group folds allocate 14/3/3 folds to train/validation/test. No identical predictor vectors cross these boundaries. Different rows and supports mean this is a sensitivity check, not a replacement estimate or a pure causal estimate of duplicate effects. The model, parameters and threshold were reused from the primary validation decision.'


def make_bundle(output, docs):
    with zipfile.ZipFile(docs/'assets/reproducibility.zip', 'w', zipfile.ZIP_DEFLATED) as bundle:
        for folder in ['src', 'tests', 'scripts', 'notebooks', 'data', 'results', '.github', 'docs']:
            for path in (ROOT/folder).rglob('*'):
                if not path.is_file() or any(p in path.parts for p in ['__pycache__', 'models', 'browser-qa', '.ipynb_checkpoints']) or path.name == 'reproducibility.zip' or path.name.startswith('executed_'):
                    continue
                bundle.write(path, path.relative_to(ROOT))
        for name in ['README.md', 'PRD.md', 'requirements.txt', 'requirements-lock.txt', 'site_config.json', '.gitignore', '.gitattributes']:
            path = ROOT/name
            if path.exists():
                bundle.write(path, name)


if __name__ == '__main__':
    from .data import load_data
    raw, frame = load_data(ROOT/'data/raw/online_shoppers_intention.csv')
    split = read_json(ROOT/'results/splits.json')
    parts = {k: np.asarray(v['source_row_ids']) for k, v in split['partitions'].items()}
    generate_report(ROOT/'results', ROOT/'docs', frame, parts)
