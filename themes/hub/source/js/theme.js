(function () {
  'use strict';

  var root = document.documentElement;
  var header = document.getElementById('site-header');
  var themeBtn = document.getElementById('theme-toggle');
  var lightLink = document.getElementById('hljs-light');
  var darkLink = document.getElementById('hljs-dark');
  var media = window.matchMedia ? window.matchMedia('(prefers-color-scheme: dark)') : null;
  var manualTheme = false;

  function storedTheme() {
    try {
      var t = localStorage.getItem('hub-theme');
      return t === 'light' || t === 'dark' ? t : null;
    } catch (e) {
      return null;
    }
  }

  function currentTheme() {
    return root.getAttribute('data-theme') === 'dark' ? 'dark' : 'light';
  }

  function applyTheme(t) {
    root.setAttribute('data-theme', t);
    if (lightLink) lightLink.disabled = t === 'dark';
    if (darkLink) darkLink.disabled = t !== 'dark';
    var meta = document.querySelector('meta[name="theme-color"]');
    if (meta) meta.setAttribute('content', t === 'dark' ? '#111c2b' : '#f4f8fd');
    if (themeBtn) {
      var next = t === 'dark' ? '浅色' : '深色';
      themeBtn.setAttribute('aria-pressed', t === 'dark' ? 'true' : 'false');
      themeBtn.setAttribute('aria-label', '切换到' + next + '模式');
      themeBtn.setAttribute('title', '切换到' + next + '模式');
    }
  }

  var saved = storedTheme();
  manualTheme = saved !== null;
  applyTheme(saved || (media && media.matches ? 'dark' : 'light'));

  if (themeBtn) {
    themeBtn.addEventListener('click', function () {
      var next = currentTheme() === 'dark' ? 'light' : 'dark';
      applyTheme(next);
      manualTheme = true;
      try { localStorage.setItem('hub-theme', next); } catch (e) {}
    });
  }

  if (media) {
    var onSchemeChange = function (e) {
      if (!manualTheme) applyTheme(e.matches ? 'dark' : 'light');
    };
    if (media.addEventListener) media.addEventListener('change', onSchemeChange);
    else if (media.addListener) media.addListener(onSchemeChange);
  }

  /* ---------- fixed header ---------- */
  function syncHeaderOffset() {
    if (!header) return;
    var h = header.offsetHeight;
    if (h > 0) root.style.setProperty('--header-offset', h + 'px');
  }

  function updateHeader() {
    if (!header) return;
    var scrolled = (window.pageYOffset || root.scrollTop) > 24;
    header.classList.toggle('is-scrolled', scrolled);
  }

  if (header && window.ResizeObserver) {
    try {
      new ResizeObserver(syncHeaderOffset).observe(header, { box: 'border-box' });
    } catch (e) {
      new ResizeObserver(syncHeaderOffset).observe(header);
    }
  }

  /* ---------- reading progress ---------- */
  function updateProgress() {
    var bar = document.getElementById('reading-progress-bar');
    if (!bar) return;
    var article = document.querySelector('.article');
    if (!article) return;
    var headerH = header ? header.offsetHeight : 0;
    var top = article.getBoundingClientRect().top + (window.pageYOffset || root.scrollTop);
    var viewport = window.innerHeight - headerH;
    var scrollable = article.offsetHeight - viewport;
    var scrolled = (window.pageYOffset || root.scrollTop) - top + headerH;
    var ratio = scrollable > 0 ? scrolled / scrollable : (scrolled > 0 ? 1 : 0);
    ratio = Math.max(0, Math.min(1, ratio));
    bar.style.width = (ratio * 100).toFixed(2) + '%';
  }

  /* ---------- TOC scrollspy ---------- */
  var spyLinks = Array.prototype.slice.call(document.querySelectorAll('.toc-panel .toc-link'));
  var spyTargets = spyLinks.map(function (link) {
    var id = link.getAttribute('href');
    if (!id || id.charAt(0) !== '#') return null;
    try { id = decodeURIComponent(id.slice(1)); } catch (e) { id = id.slice(1); }
    return document.getElementById(id);
  });
  var spyActive = null;

  function updateSpy() {
    if (!spyLinks.length) return;
    var line = (header ? header.offsetHeight : 0) + 96;
    var active = null;
    for (var i = 0; i < spyTargets.length; i++) {
      var el = spyTargets[i];
      if (el && el.getBoundingClientRect().top <= line) active = spyLinks[i];
    }
    if (active === spyActive) return;
    if (spyActive) spyActive.classList.remove('active');
    if (active) {
      active.classList.add('active');
      spyActive = active;
    } else {
      spyActive = null;
    }
  }

  /* ---------- frame-throttled scroll ---------- */
  var ticking = false;

  function onScroll() {
    if (ticking) return;
    ticking = true;
    requestAnimationFrame(function () {
      ticking = false;
      updateHeader();
      updateProgress();
      updateSpy();
    });
  }

  window.addEventListener('scroll', onScroll, { passive: true });
  window.addEventListener('resize', function () {
    syncHeaderOffset();
    onScroll();
  });
  window.addEventListener('load', function () {
    syncHeaderOffset();
    onScroll();
  });
  if (document.fonts && document.fonts.ready) {
    document.fonts.ready.then(function () {
      syncHeaderOffset();
      onScroll();
    });
  }

  syncHeaderOffset();
  updateHeader();
  updateProgress();
  updateSpy();

  /* ---------- home topic filter ---------- */
  var articleList = document.getElementById('article-list');

  if (articleList) {
    var cards = Array.prototype.slice.call(articleList.querySelectorAll('.post'));
    var topics = Array.prototype.slice.call(document.querySelectorAll('.topic[data-filter]'));
    var countEl = document.getElementById('article-count');
    var stateEl = document.getElementById('filter-state');
    var labelEl = document.getElementById('filter-label');
    var clearBtn = document.getElementById('filter-clear');
    var emptyEl = document.getElementById('empty-state');

    var applyFilter = function (filter, label) {
      var visible = 0;
      cards.forEach(function (card) {
        var match = !filter || card.getAttribute('data-topic') === filter;
        card.hidden = !match;
        if (match) visible++;
      });
      topics.forEach(function (btn) {
        var isActive = (btn.getAttribute('data-filter') || '') === filter;
        btn.classList.toggle('active', isActive);
        btn.setAttribute('aria-pressed', isActive ? 'true' : 'false');
      });
      if (countEl) {
        countEl.textContent = filter ? visible + ' / ' + cards.length + ' 篇笔记' : cards.length + ' 篇笔记';
      }
      if (stateEl) {
        stateEl.hidden = !filter;
        if (filter && labelEl) labelEl.textContent = '筛选：' + label;
      }
      if (emptyEl) emptyEl.hidden = visible !== 0;
    };

    topics.forEach(function (btn) {
      btn.addEventListener('click', function () {
        applyFilter(btn.getAttribute('data-filter') || '', btn.getAttribute('data-label') || '');
      });
    });

    if (clearBtn) {
      clearBtn.addEventListener('click', function () {
        applyFilter('', '');
      });
    }
  }

  /* ---------- back to top ---------- */
  var backTop = document.getElementById('back-to-top');

  if (backTop) {
    backTop.addEventListener('click', function (e) {
      e.preventDefault();
      var reduce = media && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
      window.scrollTo({ top: 0, behavior: reduce ? 'auto' : 'smooth' });
      var wordmark = document.querySelector('.wordmark');
      if (wordmark) wordmark.focus({ preventScroll: true });
    });
  }

  onScroll();
})();
