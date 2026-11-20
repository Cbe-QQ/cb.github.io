'use strict';

const { stripHTML, unescapeHTML } = require('hexo-util');

const PALETTE = ['yellow', 'blue', 'green', 'pink', 'orange'];

const GROUP_LABELS = {
  deepseek: { 1: 'FA', 2: 'MLA', 3: 'MoE' },
  ascend: { 1: 'CANN', 2: 'PTOAS' }
};

const GROUP_COLORS = {
  deepseek: { 1: 'yellow', 2: 'blue', 3: 'green' },
  ascend: { 1: 'pink', 2: 'orange' }
};

function plainText(html) {
  return unescapeHTML(stripHTML(String(html || '')))
    .replace(/\s+/g, ' ')
    .trim();
}

hexo.extend.helper.register('card_cover', function (post, index) {
  const group = post.group || '';
  const order = parseInt(post.order, 10);
  let label = post.cover_text;
  let color = post.cover_color;

  if (!label && GROUP_LABELS[group] && GROUP_LABELS[group][order]) {
    label = GROUP_LABELS[group][order];
  }
  if (!color && GROUP_COLORS[group] && GROUP_COLORS[group][order]) {
    color = GROUP_COLORS[group][order];
  }

  if (!label) {
    const latin = String(post.title || '').match(/[A-Za-z][A-Za-z0-9-]*/);
    label = latin ? latin[0].slice(0, 6) : String(post.title || '笔记').slice(0, 4);
    if (/^[a-z]/.test(label)) label = label.toUpperCase();
  }

  if (!color || PALETTE.indexOf(color) === -1) {
    color = PALETTE[index % PALETTE.length];
  }

  return { label: label, color: color };
});

hexo.extend.helper.register('card_summary', function (post, length) {
  const text = plainText(post.description || post.excerpt || post.content || '');
  return text.length > length ? text.slice(0, length) + '…' : text;
});

hexo.extend.helper.register('card_category', function (post) {
  if (!post.categories || !post.categories.length) return 'NOTE';
  return post.categories.toArray()[0].name;
});
