(function () {
  'use strict';
  const byId = id => document.getElementById(id);
  const slider = byId('threshold-slider');
  const reset = byId('reset-threshold');
  const lab = byId('threshold-lab');
  let validation;
  const percent = value => (100 * value).toFixed(1) + '%';
  function updateThreshold() {
    const threshold = Number(slider.value);
    const m = window.ShopperThreshold.calculateMetrics(validation.y_true, validation.purchase_score, threshold);
    byId('threshold-value').textContent = threshold.toFixed(2);
    slider.setAttribute('aria-valuetext', threshold.toFixed(2));
    for (const key of ['tp', 'fp', 'tn', 'fn']) byId('lab-' + key).textContent = m[key].toLocaleString();
    byId('lab-predicted').textContent = m.predicted_purchases.toLocaleString();
    byId('lab-precision').textContent = m.precision_defined ? percent(m.precision) : '0%*';
    byId('lab-recall').textContent = percent(m.recall);
    byId('lab-f1').textContent = m.f1.toFixed(3);
    byId('lab-interpretation').textContent = `At ${threshold.toFixed(2)}, ${m.tp} actual buyers are detected, ${m.fn} are missed, and ${m.fp} non-buyers trigger false alarms.${m.precision_defined ? '' : ' *Precision is undefined with no predicted buyers; displayed as zero by convention.'}`;
  }
  async function initLab() {
    try {
      const response = await fetch('data/threshold_validation.json');
      if (!response.ok) throw new Error('HTTP ' + response.status);
      const candidate = await response.json();
      if (candidate.partition !== 'validation' || candidate.feature_set !== 'full' || !Number.isFinite(candidate.chosen_threshold) || candidate.chosen_threshold < 0 || candidate.chosen_threshold > 1 || !Array.isArray(candidate.source_row_id) || candidate.source_row_id.length !== candidate.y_true?.length) throw new Error('Invalid validation metadata.');
      window.ShopperThreshold.calculateMetrics(candidate.y_true, candidate.purchase_score, .5);
      validation = candidate;
      const purchases = validation.y_true.reduce((a, b) => a + b, 0);
      byId('lab-support').textContent = `${validation.y_true.length.toLocaleString()} validation sessions · ${purchases.toLocaleString()} true purchases · frozen training-only model`;
      byId('chosen-marker').style.setProperty('--chosen-position', (100 * validation.chosen_threshold) + '%');
      slider.disabled = false;
      reset.disabled = false;
      slider.addEventListener('input', updateThreshold);
      reset.addEventListener('click', () => { slider.value = '0.50'; updateThreshold(); });
      updateThreshold();
      lab.setAttribute('aria-busy', 'false');
    } catch (error) {
      byId('lab-error').hidden = false;
      byId('lab-error').textContent = 'The actual validation scores could not be loaded. Serve docs/ with a local HTTP server or use the deployed site, then reload. No illustrative scores are substituted.';
      byId('lab-support').textContent = 'Validation data unavailable.';
      lab.setAttribute('aria-busy', 'false');
      console.error('Threshold lab unavailable:', error.message);
    }
  }
  const links = Array.from(document.querySelectorAll('.sidebar nav a'));
  if ('IntersectionObserver' in window) {
    const observer = new IntersectionObserver(entries => {
      for (const entry of entries) if (entry.isIntersecting) {
        links.forEach(link => {
          const active = link.hash === '#' + entry.target.id;
          link.classList.toggle('active', active);
          if (active) link.setAttribute('aria-current', 'location');
          else link.removeAttribute('aria-current');
        });
      }
    }, { rootMargin: '-8% 0px -70% 0px' });
    document.querySelectorAll('main > section').forEach(section => observer.observe(section));
  }
  fetch('data/summary.json').then(response => { if (!response.ok) throw new Error('Summary unavailable'); return response.json(); }).then(data => {
    const selected = data.test_metrics.find(r => r.model === data.selection.selected_model && r.feature_set === 'full');
    if (selected) document.querySelector('.score-scale').style.setProperty('--score-width', Math.max(0, Math.min(100, selected.ap * 100)) + '%');
  }).catch(() => { /* The full generated report remains readable without this decorative bar. */ });
  initLab();
})();
