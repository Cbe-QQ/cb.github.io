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

hexo.extend.generator.register('search-index', function (locals) {
  const config = hexo.config;
  const base = config.root.endsWith('/') ? config.root : config.root + '/';

  const data = locals.posts.sort('-date').toArray().map(function (post) {
    const categories = post.categories ? post.categories.toArray().map(function (cat) { return cat.name; }) : [];
    const tags = post.tags ? post.tags.toArray().map(function (tag) { return tag.name; }) : [];
    const content = plainText(post.content).slice(0, 4000);

    return {
      title: post.title || '',
      url: base + post.path.replace(/^\//, ''),
      date: post.date ? post.date.format('YYYY-MM-DD') : '',
      categories: categories,
      tags: tags,
      series: post.group ? (SERIES_NAMES[post.group] || post.group) : '',
      group: post.group || '',
      description: post.description || '',
      content: content
    };
  });

  return {
    path: 'search.json',
    data: JSON.stringify(data)
  };
});
