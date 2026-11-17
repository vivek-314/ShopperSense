(function () {
  'use strict';
  if (location.pathname.endsWith('/research.html')) {
    location.replace('./' + (location.hash || '#research'));
  }
})();
