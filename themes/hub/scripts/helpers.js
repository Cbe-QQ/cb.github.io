'use strict';

const { stripHTML, unescapeHTML } = require('hexo-util');

const SERIES_NAMES = {
  deepseek: 'DeepSeek 系列',
  ascend: 'Ascend 系列'
};

function plainText(html) {
  return unescapeHTML(stripHTML(html || ''))
    .replace(/\s+/g, ' ')
    .trim();
}

// 同一 group 的 docs 文章,按 front-matter order 升序
hexo.extend.helper.register('series_info', function (group) {
  if (!group) return [];
  return hexo.locals.get('posts').toArray()
    .filter(function (p) { return p.layout === 'docs' && p.group === group; })
    .sort(function (a, b) { return (a.order || 0) - (b.order || 0); });
});

hexo.extend.helper.register('series_name', function (group) {
  if (!group) return '';
  return SERIES_NAMES[group] || group;
});

// 阅读时长:中文按 300 字/分钟,英文按 200 词/分钟
hexo.extend.helper.register('reading_time', function (post) {
  const text = plainText(post && (post.content || post.excerpt));
  const cjk = (text.match(/[一-鿿]/g) || []).length;
  const words = (text.replace(/[一-鿿]/g, ' ').match(/[A-Za-z0-9_]+/g) || []).length;
  return Math.max(1, Math.ceil(cjk / 300 + words / 200));
});

// 纯文本摘要,用于没有 description 的文章
hexo.extend.helper.register('post_excerpt', function (post, length) {
  const limit = length || 180;
  const text = plainText(post && post.content);
  return text.length > limit ? text.slice(0, limit).trim() + '…' : text;
});

// 卡片左上角展示的主题名:优先系列名,其次第一个分类
hexo.extend.helper.register('post_topic', function (post) {
  if (!post) return '';
  if (post.group) return SERIES_NAMES[post.group] || post.group;
  const cats = post.categories && post.categories.toArray ? post.categories.toArray() : [];
  return cats.length ? cats[0].name : 'NOTE';
});
