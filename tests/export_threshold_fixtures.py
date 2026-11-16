"""Generate independent Python expectations for the shared browser/Node function."""
import json
from src.config import ROOT
from src.evaluate import metrics

data = json.loads((ROOT/'docs/data/threshold_validation.json').read_text())
fixtures = []
for threshold in [0., .5, 1., data['chosen_threshold']]:
    fixtures.append({'name': f'validation threshold {threshold}', 'threshold': threshold, 'metrics': metrics(data['y_true'], data['purchase_score'], threshold)})
for name, y, scores, threshold in [('equality', [1, 0, 1, 0], [.5, .8, .2, .1], .5), ('no positives', [0, 1], [.1, .2], 1.)]:
    fixtures.append({'name': name, 'y_true': y, 'purchase_score': scores, 'threshold': threshold, 'metrics': metrics(y, scores, threshold)})
(ROOT/'results/threshold_parity.json').write_text(json.dumps(fixtures, indent=2))
