// Apply the saved/device theme before CSS paints; session inputs are never stored.
(function () {
  'use strict';
  const root = document.documentElement;
  const preferenceKey = 'shoppersense-theme';
  const deviceLight = window.matchMedia('(prefers-color-scheme: light)');
  let preference = null;
  let button = null;
  try {
    const saved = localStorage.getItem(preferenceKey);
    if (saved === 'light' || saved === 'dark') preference = saved;
  } catch { /* A blocked storage API must not prevent theme switching. */ }

  function applyTheme() {
    const theme = preference || (deviceLight.matches ? 'light' : 'dark');
    root.dataset.theme = theme;
    const meta = document.querySelector('meta[name="theme-color"]');
    if (meta) meta.content = theme === 'light' ? '#f7f8fa' : '#040506';
    if (button) {
      const next = theme === 'light' ? 'dark' : 'light';
      button.querySelector('.theme-label').textContent = next === 'light' ? 'Light mode' : 'Dark mode';
      button.setAttribute('aria-label', `Switch to ${next} mode`);
      button.title = `Current theme: ${theme}. Switch to ${next} mode.`;
    }
  }
  applyTheme();

  document.addEventListener('DOMContentLoaded', () => {
    button = document.getElementById('theme-toggle');
    if (!button) return;
    button.hidden = false;
    applyTheme();
    button.addEventListener('click', () => {
      preference = root.dataset.theme === 'light' ? 'dark' : 'light';
      try { localStorage.setItem(preferenceKey, preference); } catch { /* Keep the in-page choice. */ }
      applyTheme();
    });
  });
  deviceLight.addEventListener('change', () => { if (!preference) applyTheme(); });
  window.addEventListener('storage', event => {
    if (event.key !== preferenceKey && event.key !== null) return;
    preference = event.key === preferenceKey && ['light', 'dark'].includes(event.newValue) ? event.newValue : null;
    applyTheme();
  });
})();
