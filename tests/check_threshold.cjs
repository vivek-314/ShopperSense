// Run with node tests/check_threshold.cjs; compares JS to independent Python exports.
const fs = require('node:fs');
const assert = require('node:assert/strict');
const path = require('node:path');
const { calculateMetrics } = require('../docs/threshold.js');
const root = path.resolve(__dirname, '..');
const data = JSON.parse(fs.readFileSync(path.join(root, 'docs/data/threshold_validation.json'), 'utf8'));
const expected = JSON.parse(fs.readFileSync(path.join(root, 'results/threshold_parity.json'), 'utf8'));
for (const fixture of expected) {
  const actual = calculateMetrics(fixture.y_true || data.y_true, fixture.purchase_score || data.purchase_score, fixture.threshold);
  for (const key of ['tp', 'fp', 'tn', 'fn', 'predicted_purchases', 'precision', 'recall', 'f1']) {
    assert.ok(Math.abs(actual[key] - fixture.metrics[key]) < 1e-12, `${fixture.name}: ${key}`);
  }
}
assert.throws(() => calculateMetrics([1], [NaN], .5));
assert.throws(() => calculateMetrics([2], [.3], .5));
console.log(`Passed ${expected.length} JS/Python threshold parity fixtures, including equality and zero predicted positives.`);
