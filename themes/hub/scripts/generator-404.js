'use strict';

// hexo 7 核心不生成 404.html,这里注册一个
hexo.extend.generator.register('404', function () {
  return {
    path: '404.html',
    layout: ['404', 'page', 'post', 'index'],
    data: { title: '404 · 页面不存在' }
  };
});
