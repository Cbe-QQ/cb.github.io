(function () {
  'use strict';

  // 复制文章链接
  document.querySelectorAll('[data-copy]').forEach(function (button) {
    button.addEventListener('click', function () {
      var url = button.getAttribute('data-copy');
      if (!url || !navigator.clipboard) return;

      navigator.clipboard.writeText(url).then(function () {
        button.classList.add('is-copied');
        button.setAttribute('aria-label', '已复制链接');
        window.setTimeout(function () {
          button.classList.remove('is-copied');
          button.setAttribute('aria-label', '复制文章链接');
        }, 1600);
      });
    });
  });

  // 搜索页的简单客户端筛选
  var input = document.getElementById('search-input');
  var list = document.getElementById('search-list');
  var empty = document.getElementById('search-empty');

  if (input && list) {
    var items = Array.prototype.slice.call(list.querySelectorAll('li'));

    input.addEventListener('input', function () {
      var keyword = input.value.trim().toLowerCase();
      var visible = 0;

      items.forEach(function (item) {
        var match = !keyword || (item.getAttribute('data-search') || '').indexOf(keyword) !== -1;
        item.hidden = !match;
        if (match) visible += 1;
      });

      if (empty) empty.hidden = visible !== 0;
    });
  }
})();
