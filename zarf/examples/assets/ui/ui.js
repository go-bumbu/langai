// What the pages share: the navigation between them, filled into the
// <nav id="nav"> of every header, and window.ui, the handful of helpers the
// pages are written with (no framework: elements, fetch, a toast).
(() => {
  // the pages, by their path under the service's root
  const pages = [
    ['mediasearch/', 'media search'],
    ['textsearch/', 'text search'],
    ['duplicates/', 'duplicates'],
    ['autotag/', 'auto tagging'],
  ];

  const $ = (sel, root = document) => root.querySelector(sel);

  // el builds an element: attrs set attributes, "class" the class name and
  // "on…" a listener; null, undefined and false children are left out
  const el = (tag, attrs = {}, ...children) => {
    const e = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (k === 'class') e.className = v;
      else if (k.startsWith('on')) e.addEventListener(k.slice(2), v);
      else if (v !== null && v !== undefined && v !== false) e.setAttribute(k, v);
    }
    for (const c of children.flat()) if (c !== null && c !== undefined && c !== false) e.append(c);
    return e;
  };

  let toastTimer;
  const toast = (msg) => {
    let t = $('.toast');
    if (!t) { t = el('div', { class: 'toast' }); document.body.append(t); }
    t.textContent = msg;
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => t.remove(), 6000);
  };

  // api fetches JSON; opts.json is sent as the JSON body. A failed request
  // throws an Error with the server's message, its status in .status and
  // the whole body in .body
  const api = async (url, opts = {}) => {
    if (opts.json !== undefined) {
      opts = { ...opts, body: JSON.stringify(opts.json), headers: { 'Content-Type': 'application/json' } };
    }
    const r = await fetch(url, opts);
    const body = r.status === 204 ? {} : await r.json().catch(() => ({}));
    if (!r.ok) {
      const e = new Error(body.error || r.statusText);
      e.status = r.status;
      e.body = body;
      throw e;
    }
    return body;
  };

  // busy disables a button while fn runs and reports a failure
  const busy = async (btn, fn) => {
    if (btn) btn.disabled = true;
    try { await fn(); } catch (e) { toast(e.message); } finally { if (btn) btn.disabled = false; }
  };

  // ids are relative paths: every segment is encoded on its own, so that
  // the slashes stay slashes
  const encPath = (id) => id.split('/').map(encodeURIComponent).join('/');
  const secs = (ms) => (ms / 1000).toFixed(2) + ' s';
  const bytes = (n) => n >= 1 << 20 ? (n / (1 << 20)).toFixed(1) + ' MB' : n >= 1024 ? (n / 1024).toFixed(1) + ' KB' : n + ' B';
  // bar draws a score from 0 to 1, with a mark at cut when one is given
  const bar = (score, cut) => el('div', { class: 'scorebar' },
    el('i', { style: `width:${Math.max(0, Math.min(1, score)) * 100}%` }),
    cut === undefined || cut === null ? null : el('span', { class: 'cut', style: `left:${Math.max(0, Math.min(1, cut)) * 100}%` }));

  window.ui = { $, el, toast, api, busy, encPath, secs, bytes, bar };

  // the header is sticky and as tall as its text makes it: --header tells
  // a page's own sticky panels where it ends
  const header = document.querySelector('header');
  if (header && window.ResizeObserver) {
    // on the next frame: setting it moves the layout the observer watches
    const measure = () => document.documentElement.style.setProperty('--header', header.offsetHeight + 'px');
    new ResizeObserver(() => requestAnimationFrame(measure)).observe(header);
  }

  const nav = document.getElementById('nav');
  if (!nav) return;
  // every page but the home page lives one level under the root
  const root = nav.dataset.root || '../';
  const here = nav.dataset.page || '';
  nav.replaceChildren(el('a', { class: 'home' + (here === '' ? ' here' : ''), href: root }, 'langai examples'),
    ...pages.map(([path, name]) => el('a', { href: root + path, class: path === here ? 'here' : '' }, name)));
})();
