'use strict';

const { stripHTML, unescapeHTML, escapeHTML } = require('hexo-util');

function plainText(html) {
  return unescapeHTML(stripHTML(html || ''))
    .replace(/\s+/g, ' ')
    .trim();
}

hexo.extend.generator.register('rss-feed', function (locals) {
  const config = hexo.config;
  const siteUrl = config.url.replace(/\/$/, '');
  const channelLink = siteUrl + '/';
  const posts = locals.posts.sort('-date').limit(20).toArray();

  const items = posts.map(function (post) {
    const link = siteUrl + '/' + post.path.replace(/^\//, '');
    const summary = post.description || plainText(post.content).slice(0, 300);

    return [
      '<item>',
      '<title>' + escapeHTML(post.title || '') + '</title>',
      '<link>' + escapeHTML(link) + '</link>',
      '<guid isPermaLink="true">' + escapeHTML(link) + '</guid>',
      '<pubDate>' + post.date.toDate().toUTCString() + '</pubDate>',
      '<description>' + escapeHTML(summary) + '</description>',
      '</item>'
    ].join('');
  }).join('');

  const xml = '<?xml version="1.0" encoding="UTF-8"?>' +
    '<rss version="2.0" xmlns:atom="http://www.w3.org/2005/Atom">' +
    '<channel>' +
    '<title>' + escapeHTML(config.title || '') + '</title>' +
    '<link>' + escapeHTML(channelLink) + '</link>' +
    '<atom:link href="' + escapeHTML(channelLink + 'rss.xml') + '" rel="self" type="application/rss+xml"/>' +
    '<description>' + escapeHTML(config.description || '') + '</description>' +
    '<language>zh-CN</language>' +
    '<lastBuildDate>' + new Date().toUTCString() + '</lastBuildDate>' +
    items +
    '</channel>' +
    '</rss>';

  return {
    path: 'rss.xml',
    data: xml
  };
});
