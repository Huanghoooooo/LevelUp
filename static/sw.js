// Service Worker for LevelUp PWA
// 更新版本号会强制更新缓存
const CACHE_NAME = 'levelup-v2';

// 只缓存真正的静态资源（不包含动态数据的）
const STATIC_ASSETS = [
  '/manifest.json',
  '/icon.svg'
];

// 安装 Service Worker
self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => {
      return cache.addAll(STATIC_ASSETS);
    })
  );
  // 立即激活新的 Service Worker
  self.skipWaiting();
});

// 激活 Service Worker - 清除旧缓存
self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((cacheNames) => {
      return Promise.all(
        cacheNames
          .filter((name) => name !== CACHE_NAME)
          .map((name) => caches.delete(name))
      );
    })
  );
  self.clients.claim();
});

// 请求拦截策略
self.addEventListener('fetch', (event) => {
  const url = new URL(event.request.url);
  
  // 所有 API 请求和 HTML 页面：始终从网络获取，不缓存
  // 这确保了认证状态和数据始终是最新的
  if (url.pathname.startsWith('/boards') || 
      url.pathname.startsWith('/records') ||
      url.pathname.startsWith('/auth') ||
      url.pathname === '/' ||
      url.pathname === '/index.html' ||
      event.request.method !== 'GET') {
    event.respondWith(
      fetch(event.request).catch(() => {
        // 离线时 API 请求返回错误
        if (url.pathname.startsWith('/auth') || 
            url.pathname.startsWith('/boards') || 
            url.pathname.startsWith('/records')) {
          return new Response(JSON.stringify({ error: '离线状态，无法访问' }), {
            status: 503,
            headers: { 'Content-Type': 'application/json' }
          });
        }
        // 离线时尝试返回缓存的页面
        return caches.match('/');
      })
    );
    return;
  }
  
  // 只有静态资源使用缓存策略
  event.respondWith(
    caches.match(event.request).then((cachedResponse) => {
      if (cachedResponse) {
        return cachedResponse;
      }
      return fetch(event.request).then((response) => {
        // 只缓存成功的静态资源请求
        if (response.ok && STATIC_ASSETS.some(asset => url.pathname.endsWith(asset))) {
          const responseClone = response.clone();
          caches.open(CACHE_NAME).then((cache) => {
            cache.put(event.request, responseClone);
          });
        }
        return response;
      });
    })
  );
});

