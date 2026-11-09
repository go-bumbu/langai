// The media search page: a grid of the catalog, and for the clicked item or
// an uploaded file the items that look like it, with the numbers behind
// every hit.
(() => {
  const $ = (sel) => document.querySelector(sel);
  const grid = $('#grid');
  const detail = $('#detail');
  const infoEl = $('#info');
  let items = [];
  let active = null;

  // ids are relative paths; every segment is encoded on its own so that
  // the slashes stay slashes
  const enc = (id) => id.split('/').map(encodeURIComponent).join('/');
  const ms = (n) => (n / 1000).toFixed(2) + ' s';
  const bytes = (n) => n > 1 << 20 ? (n / (1 << 20)).toFixed(1) + ' MB' : (n / 1024).toFixed(0) + ' KB';
  const el = (tag, attrs = {}, ...children) => {
    const e = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (k === 'class') e.className = v;
      else if (k.startsWith('on')) e.addEventListener(k.slice(2), v);
      else if (v !== null && v !== undefined) e.setAttribute(k, v);
    }
    for (const c of children) if (c !== null && c !== undefined) e.append(c);
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

  const api = async (url, opts) => {
    const r = await fetch(url, opts);
    const body = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(body.error || r.statusText);
    return body;
  };

  let info = {};
  async function loadInfo() {
    const d = await api('api/info');
    info = d.info;
    const scan = d.lastScan && d.lastScan.at && !d.lastScan.at.startsWith('0001')
      ? ` · scan ${ms(d.lastScan.elapsedMs)}: ${d.lastScan.indexed} indexed, ${d.lastScan.skipped} unchanged, ${d.lastScan.failed} failed, ${d.lastScan.removed} removed`
      : (d.scanning ? ' · scanning…' : '');
    const model = d.modelState ? ` · model ${d.modelState}` : '';
    const videos = info.mode === 'segments' ? `videos as ${ms(info.segmentMs)} segments` : 'videos as sampled frames';
    infoEl.innerHTML = `<b>${info.model}</b> · ${info.dimensions} dims · ${info.provider}${model} · ${videos} · min score ${info.minScore} · limit ${d.limit}<br>` +
      `<b>${d.images}</b> images, <b>${d.videos}</b> videos in ${d.dir}${scan}`;
    $('#textsearch').hidden = !info.text;
    banner(d);
    return d;
  }

  // banner offers the reset when the database was built by another model:
  // vectors of two models never mix, so nothing is indexed or searched
  // until the collection is emptied
  function banner(d) {
    const b = $('#banner');
    b.hidden = !(d.lastScan && d.lastScan.mismatch);
    if (b.hidden) return;
    const btn = el('button', { class: 'button danger' }, 'reset the index and rescan');
    btn.addEventListener('click', async () => {
      if (!confirm('Empty the index and embed every file again with the model that runs now?')) return;
      btn.disabled = true;
      try {
        await api('api/reset', { method: 'POST' });
        toast('index emptied; scanning…');
        const s = await api('api/rescan', { method: 'POST' });
        toast(`rescan: ${s.indexed} indexed, ${s.skipped} unchanged, ${s.failed} failed, ${s.removed} removed in ${ms(s.elapsedMs)}`);
      } catch (err) {
        toast(err.message);
      }
      await Promise.all([loadInfo(), loadItems()]).catch(() => {});
    });
    b.replaceChildren(el('div', { class: 'what' }, el('b', {}, 'The index was built by another model. '),
      d.storedModel ? `The database holds this collection's vectors under ${d.storedModel}; ` : 'The database holds this collection under another model; ',
      `the page runs ${info.model}. Vectors of two models never mix, so nothing is indexed or searched until the collection is emptied (MediaIndex.Reset).`), btn);
  }

  // segmentOf names the segment a time falls in, in the segments mode
  const segmentOf = (t) => info.mode === 'segments' && info.segmentMs > 0 ? ` · segment ${Math.floor(t / info.segmentMs) + 1}` : '';

  async function loadItems() {
    items = await api('api/items');
    grid.replaceChildren(...items.map(card));
  }

  function card(it) {
    const c = el('div', { class: 'card' + (active === it.id ? ' active' : ''), title: it.id, onclick: () => open(it.id) },
      el('img', { src: 'thumb/' + enc(it.id), loading: 'lazy', alt: it.name }),
      it.kind === 'video' ? el('span', { class: 'badge video' }, 'video') : null,
      it.error ? el('span', { class: 'badge ' + (it.reason ? 'skipped' : 'error'), title: it.error }, it.reason || 'failed') : null,
      el('div', { class: 'name' }, it.name));
    c.dataset.id = it.id;
    return c;
  }

  function mark(id) {
    active = id;
    for (const c of grid.children) c.classList.toggle('active', c.dataset.id === id);
  }

  // open shows a catalog item and its look-alikes
  async function open(id) {
    mark(id);
    const it = items.find((x) => x.id === id);
    detail.replaceChildren(preview(it), facts(it), el('p', { class: 'hint' }, 'searching…'));
    const jobs = [api('api/similar/' + enc(id) + ($('#reembed').checked ? '?embed=1' : ''))];
    if (it.kind === 'video') jobs.push(api('api/frames/' + enc(id)).catch((e) => ({ error: e.message })));
    try {
      const [res, board] = await Promise.all(jobs);
      detail.replaceChildren(...[preview(it), facts(it), board ? storyboard(it, board) : null, hits(res, it)].filter(Boolean));
    } catch (e) {
      detail.replaceChildren(preview(it), facts(it), el('p', { class: 'error' }, e.message));
    }
  }

  function preview(it, src) {
    const url = src || 'file/' + enc(it.id);
    const node = it.kind === 'video'
      ? el('video', { src: url, controls: '', preload: 'metadata', id: 'preview', poster: src ? null : 'thumb/' + enc(it.id) })
      : el('img', { src: url, alt: it.name });
    return el('div', { class: 'preview' }, node);
  }

  function facts(it) {
    const rows = [['path', it.id], ['kind', it.kind], ['size', bytes(it.size)],
      ['indexed in', it.indexedMs + ' ms' + (it.skipped ? ' (unchanged, no-op)' : '')]];
    if (it.error) rows.push([it.reason ? 'not indexed' : 'error', it.error]);
    const f = el('div', { class: 'facts' });
    for (const [k, v] of rows) f.append(el('span', {}, k), el('b', { class: k === 'error' ? 'error' : '' }, v));
    return f;
  }

  // storyboard lays the frame times over the strip; a click seeks the preview
  function storyboard(it, board) {
    if (board.error) return el('div', { class: 'board' }, el('h3', {}, 'frames the index saw'), el('p', { class: 'error' }, board.error));
    const scale = 90 / board.cellH;
    const cells = el('div', { class: 'cells' }, ...board.times.map((t) =>
      el('div', { class: 'cell', style: `width:${board.cellW * scale}px`, title: 'seek to ' + ms(t) + segmentOf(t), onclick: () => seek(t) },
        el('span', {}, ms(t) + segmentOf(t).replace(' · segment ', ' s')))));
    const title = info.mode === 'segments'
      ? `sampled frames (${board.times.length}); the index saw ${ms(info.segmentMs)} segments`
      : `frames the index saw (${board.times.length})`;
    return el('div', { class: 'board' },
      el('h3', {}, title),
      el('div', { class: 'strip', style: `width:${board.cellW * scale * board.times.length}px` },
        el('img', { src: 'storyboard/' + enc(it.id), alt: 'storyboard' }), cells));
  }

  function seek(t) {
    const v = $('#preview');
    if (v) { v.currentTime = t / 1000; v.pause(); }
  }

  // hits renders a Result; query is the open item, or null for an upload
  function hits(res, query) {
    const list = el('div', { class: 'hits' },
      el('h3', {}, 'looks like'),
      el('div', { class: 'summary' }, res.stored
        ? `stored vectors reused, nothing embedded, ${res.hits.length} hits in ${res.elapsedMs} ms`
        : `${res.frames} query frame${res.frames === 1 ? '' : 's'} embedded, ${res.hits.length} hits in ${res.elapsedMs} ms`));
    if (!res.hits.length) list.append(el('p', { class: 'hint' }, 'nothing above the min score'));
    for (const h of res.hits) {
      const times = [];
      const segments = info.mode === 'segments';
      if (h.item.kind === 'video') {
        times.push(segments ? 'matched in the segment at ' : 'matched at ');
        times.push(h.self || (query && h.item.id === query.id)
          ? el('a', { href: '#', onclick: (e) => { e.preventDefault(); seek(h.atMs); } }, ms(h.atMs))
          : ms(h.atMs));
      }
      if (query && query.kind === 'video') {
        times.push((times.length ? ' · ' : '') + (segments ? 'query segment at ' : 'query frame '));
        times.push(el('a', { href: '#', onclick: (e) => { e.preventDefault(); seek(h.queryAtMs); } }, ms(h.queryAtMs)));
      }
      if (res.frames > 1 && !query) times.push((segments ? 'query segment at ' : 'query frame ') + ms(h.queryAtMs));
      list.append(el('div', { class: 'hit' + (h.self ? ' self' : ''), onclick: () => open(h.item.id) },
        el('img', { src: 'thumb/' + enc(h.item.id), loading: 'lazy', alt: '' }),
        el('div', { class: 'meta' },
          el('div', { class: 'title' }, el('span', { title: h.item.id }, (h.self ? 'self · ' : '') + h.item.name), el('span', { class: 'score' }, h.score.toFixed(4))),
          el('div', { class: 'bar' }, el('i', { style: `width:${Math.max(0, Math.min(1, h.score)) * 100}%` })),
          el('div', { class: 'times' }, ...times))));
    }
    return list;
  }

  async function upload(file) {
    mark(null);
    const kind = file.type.startsWith('video/') || /\.(mp4|m4v|mov|mkv|webm|avi)$/i.test(file.name) ? 'video' : 'image';
    const fake = { id: file.name, name: file.name, kind, size: file.size, indexedMs: 0 };
    const url = URL.createObjectURL(file);
    detail.replaceChildren(preview(fake, url), el('p', { class: 'hint' }, `uploading and searching ${file.name}…`));
    const form = new FormData();
    form.append('file', file);
    try {
      const res = await api('api/search', { method: 'POST', body: form });
      const f = el('div', { class: 'facts' }, el('span', {}, 'upload'), el('b', {}, file.name), el('span', {}, 'size'), el('b', {}, bytes(file.size)));
      detail.replaceChildren(preview(fake, url), f, hits(res, null));
    } catch (e) {
      detail.replaceChildren(preview(fake, url), el('p', { class: 'error' }, e.message));
    }
  }

  async function searchText(q) {
    mark(null);
    detail.replaceChildren(el('p', { class: 'hint' }, `searching for “${q}”…`));
    try {
      const res = await api('api/text?q=' + encodeURIComponent(q));
      const f = el('div', { class: 'facts' }, el('span', {}, 'text'), el('b', {}, q));
      const list = hits(res, null);
      list.querySelector('.summary').textContent = `text embedded, ${res.hits.length} hits in ${res.elapsedMs} ms`;
      detail.replaceChildren(f, list);
    } catch (e) {
      detail.replaceChildren(el('p', { class: 'error' }, e.message));
    }
  }

  $('#textsearch').addEventListener('submit', (e) => { e.preventDefault(); const q = $('#q').value.trim(); if (q) searchText(q); });
  // switching the mode runs the open item's search again, to compare
  $('#reembed').addEventListener('change', () => { if (active) open(active); });
  $('#upload').addEventListener('change', (e) => { if (e.target.files[0]) upload(e.target.files[0]); e.target.value = ''; });
  $('#rescan').addEventListener('click', async (e) => {
    e.target.disabled = true;
    try {
      const s = await api('api/rescan', { method: 'POST' });
      toast(`rescan: ${s.indexed} indexed, ${s.skipped} unchanged, ${s.failed} failed, ${s.removed} removed in ${ms(s.elapsedMs)}`);
      await Promise.all([loadInfo(), loadItems()]);
    } catch (err) {
      toast(err.message);
      // a scan that met another model's vectors: show the reset
      await Promise.all([loadInfo(), loadItems()]).catch(() => {});
    } finally {
      e.target.disabled = false;
    }
  });

  // while the startup scan runs, the catalog fills: look again until it is
  // done
  async function start() {
    const [d] = await Promise.all([loadInfo(), loadItems()]);
    if (d.scanning) setTimeout(() => start().catch(() => {}), 2000);
  }
  start().catch((e) => toast(e.message));
})();
