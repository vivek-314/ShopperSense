/* One pure function shared by the browser and Node parity checks. */
(function (root) {
  'use strict';
  function calculateMetrics(labels, scores, threshold) {
    if (!Array.isArray(labels) || !Array.isArray(scores) || labels.length !== scores.length || labels.length === 0 || !Number.isFinite(threshold) || threshold < 0 || threshold > 1) throw new Error('Invalid threshold data.');
    let tp = 0, fp = 0, tn = 0, fn = 0;
    for (let i = 0; i < labels.length; i++) {
      const y = labels[i], score = scores[i];
      if ((y !== 0 && y !== 1) || !Number.isFinite(score) || score < 0 || score > 1) throw new Error('Labels or scores are invalid.');
      const predicted = score >= threshold;
      if (predicted && y === 1) tp++;
      else if (predicted) fp++;
      else if (y === 1) fn++;
      else tn++;
    }
    const precision = tp + fp ? tp / (tp + fp) : 0;
    const recall = tp + fn ? tp / (tp + fn) : 0;
    return { tp, fp, tn, fn, predicted_purchases: tp + fp, precision, recall, f1: 2 * tp + fp + fn ? 2 * tp / (2 * tp + fp + fn) : 0, precision_defined: tp + fp > 0 };
  }
  root.ShopperThreshold = { calculateMetrics };
  if (typeof module !== 'undefined' && module.exports) module.exports = { calculateMetrics };
})(typeof globalThis !== 'undefined' ? globalThis : this);
