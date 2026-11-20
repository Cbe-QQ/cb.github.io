(function () {
  'use strict';

  var shell = document.querySelector('.search-shell');
  if (!shell) return;

  var input = document.getElementById('search-input');
  var statusEl = document.getElementById('search-status');
  var resultsEl = document.getElementById('search-results');
  var indexUrl = shell.getAttribute('data-index-url');
  var posts = [];
  var tokens = [];
  var loaded = false;
  var failed = false;
  var timer = null;
  var MAX_RESULTS = 20;

  function escapeRegExp(str) {
    return str.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  }

  function setStatus(text) {
    if (statusEl) statusEl.textContent = text;
  }

  function clearResults() {
    while (resultsEl && resultsEl.firstChild) resultsEl.removeChild(resultsEl.firstChild);
  }

  function appendHighlighted(parent, text, needleList) {
    if (!needleList.length) {
      parent.appendChild(document.createTextNode(text));
      return;
    }
    var re;
    try {
      re = new RegExp('(' + needleList.map(escapeRegExp).join('|') + ')', 'gi');
    } catch (e) {
      parent.appendChild(document.createTextNode(text));
      return;
    }
    var last = 0;
    var match;
    while ((match = re.exec(text)) !== null) {
      if (match.index > last) {
        parent.appendChild(document.createTextNode(text.slice(last, match.index)));
      }
      var mark = document.createElement('mark');
      mark.textContent = match[0];
      parent.appendChild(mark);
      last = match.index + match[0].length;
      if (match[0].length === 0) re.lastIndex++;
    }
    if (last < text.length) {
      parent.appendChild(document.createTextNode(text.slice(last)));
    }
  }

  function tokenize(query) {
    return query.toLowerCase().split(/\s+/).filter(function (t) { return t.length > 0; });
  }

  function scorePost(post, needleList) {
    var title = (post.title || '').toLowerCase();
    var series = (post.series || '').toLowerCase();
    var cats = (post.categories || []).join(' ').toLowerCase();
    var tags = (post.tags || []).join(' ').toLowerCase();
    var desc = (post.description || '').toLowerCase();
    var content = (post.content || '').toLowerCase();
    var score = 0;

    for (var i = 0; i < needleList.length; i++) {
      var t = needleList[i];
      var found = false;

      if (title.indexOf(t) !== -1) {
        score += 100;
        if (title.indexOf(t) === 0) score += 40;
        found = true;
      }
      if (series.indexOf(t) !== -1) { score += 55; found = true; }
      if (cats.indexOf(t) !== -1 || tags.indexOf(t) !== -1) { score += 35; found = true; }
      if (desc.indexOf(t) !== -1) { score += 18; found = true; }
      if (content.indexOf(t) !== -1) {
        score += 8;
        if (content.indexOf(t) < 200) score += 4;
        found = true;
      }

      if (!found) return -1;
    }

    return score;
  }

  function buildSnippet(post) {
    var desc = (post.description || '').trim();
    if (desc) return desc.length > 180 ? desc.slice(0, 180).trim() + '…' : desc;

    // 没有摘要时从正文取一段，并去掉其中的 TeX 源码
    var text = (post.content || '')
      .replace(/\$\$[\s\S]*?\$\$/g, ' ')
      .replace(/\$[^$\n]*\$/g, ' ')
      .replace(/\s+/g, ' ')
      .trim();
    var lower = text.toLowerCase();
    var first = -1;

    tokens.forEach(function (t) {
      var idx = lower.indexOf(t);
      if (idx !== -1 && (first === -1 || idx < first)) first = idx;
    });

    var start = first > 60 ? first - 60 : 0;
    var end = Math.min(text.length, (first === -1 ? 0 : first) + 160);
    var snippet = text.slice(start, end).trim();
    return (start > 0 ? '…' : '') + snippet + (end < text.length ? '…' : '');
  }

  function render(matches, query) {
    clearResults();
    if (!resultsEl) return;

    if (!matches.length) {
      setStatus('没有找到与“' + query + '”相关的笔记。');
      return;
    }

    var fragment = document.createDocumentFragment();

    matches.slice(0, MAX_RESULTS).forEach(function (post) {
      var card = document.createElement('article');
      card.className = 'search-result';

      var meta = document.createElement('div');
      meta.className = 'search-result-meta';
      var metaParts = [];
      if (post.series) metaParts.push(post.series);
      if (post.date) metaParts.push(post.date);
      if (!post.series && post.categories && post.categories.length) metaParts.push(post.categories.join(' / '));
      meta.textContent = metaParts.join(' · ');
      if (meta.textContent) card.appendChild(meta);

      var heading = document.createElement('h2');
      var link = document.createElement('a');
      link.href = post.url;
      appendHighlighted(link, post.title || '', tokens);
      heading.appendChild(link);
      card.appendChild(heading);

      var snippet = document.createElement('p');
      appendHighlighted(snippet, buildSnippet(post), tokens);
      card.appendChild(snippet);

      fragment.appendChild(card);
    });

    resultsEl.appendChild(fragment);
    setStatus('找到 ' + matches.length + ' 篇相关笔记' +
      (matches.length > MAX_RESULTS ? '，显示前 ' + MAX_RESULTS + ' 篇。' : '。'));
  }

  function search(query) {
    var q = query.trim();
    tokens = tokenize(q);

    if (!q) {
      clearResults();
      setStatus(loaded ? '共 ' + posts.length + ' 篇笔记，输入关键词开始搜索。' : '正在加载索引…');
      return;
    }

    if (!loaded) {
      if (failed) setStatus('搜索索引加载失败，请稍后重试。');
      return;
    }

    var matches = [];
    posts.forEach(function (post) {
      var score = scorePost(post, tokens);
      if (score >= 0) {
        matches.push({ post: post, score: score });
      }
    });

    matches.sort(function (a, b) {
      if (b.score !== a.score) return b.score - a.score;
      return String(b.post.date || '').localeCompare(String(a.post.date || ''));
    });

    render(matches.map(function (item) { return item.post; }), q);
  }

  function scheduleSearch() {
    clearTimeout(timer);
    timer = setTimeout(function () {
      search(input ? input.value : '');
    }, 140);
  }

  function initQueryFromUrl() {
    if (!input || !window.URLSearchParams) return;
    var q = new URLSearchParams(window.location.search).get('q');
    if (q) {
      input.value = q;
      search(q);
    }
  }

  fetch(indexUrl, { credentials: 'same-origin' })
    .then(function (res) {
      if (!res.ok) throw new Error('HTTP ' + res.status);
      return res.json();
    })
    .then(function (data) {
      posts = Array.isArray(data) ? data : [];
      loaded = true;
      if (input && input.value.trim()) search(input.value);
      else {
        setStatus('共 ' + posts.length + ' 篇笔记，输入关键词开始搜索。');
        initQueryFromUrl();
      }
    })
    .catch(function (err) {
      failed = true;
      setStatus('搜索索引加载失败，请稍后重试。');
      if (window.console && console.error) console.error('[search]', err);
    });

  if (input) {
    input.addEventListener('input', scheduleSearch);
    input.addEventListener('search', function () { search(input.value); });
    input.addEventListener('keydown', function (e) {
      if (e.key === 'Escape') {
        input.value = '';
        search('');
      }
    });
  }

  document.addEventListener('keydown', function (e) {
    if (e.key !== '/' || e.ctrlKey || e.metaKey || e.altKey) return;
    var el = e.target;
    var tag = el && el.tagName ? el.tagName.toLowerCase() : '';
    if (tag === 'input' || tag === 'textarea' || tag === 'select' || (el && el.isContentEditable)) return;
    if (!input) return;
    e.preventDefault();
    input.focus();
  });
})();
