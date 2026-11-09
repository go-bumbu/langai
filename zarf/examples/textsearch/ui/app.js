// The text search page: the documents the scan found on the left, and on
// the right the hits of a search (the chunk that matched, with its score)
// or a document cut into the chunks that were embedded, with the documents
// most like it.
(() => {
  const { $, el, api, toast, busy, encPath, secs, bytes, bar } = window.ui;
  const detail = $('#detail');
  let docs = [];
  let active = null;
  let info = {};

  const took = (us) => us < 1000 ? us + ' µs' : us < 1e6 ? (us / 1000).toFixed(1) + ' ms' : (us / 1e6).toFixed(2) + ' s';
  const statePill = (d) => el('span', { class: 'pill ' + ({ indexed: 'busy', unchanged: 'ok', failed: 'bad' }[d.state] || 'warn'), title: 'what the last scan did with it' }, d.state);

  async function loadInfo() {
    const d = await api('api/info');
    info = d.info;
    const model = d.modelState ? ` · model ${d.modelState}` : '';
    $('#info').replaceChildren(el('b', { title: 'the identity the store records with the vectors' }, info.model),
      ` · ${info.dimensions} dims${model} · chunker `, el('b', { title: 'the chunker\'s fingerprint, hashed with every document: changing it indexes them again' }, info.chunker),
      ` · min score ${info.minScore} · limit ${d.limit} · ${info.store}`, el('br'),
      el('b', {}, String(d.docs)), ' documents, ', el('b', {}, String(d.chunks)), ` chunks in ${d.dir}` + (d.scanning ? ' · scanning…' : ''));
    renderScan(d);
    renderBanner(d);
    $('#examples').replaceChildren(...(d.examples && d.examples.length ? ['try: ', ...d.examples.map((q) =>
      el('button', { class: 'chip', title: 'a question in one language whose answer is in another', onclick: () => { $('#q').value = q; search(q); } }, q))] : []));
    return d;
  }

  // renderScan shows the BatchResult of the last scan
  function renderScan(d) {
    const s = d.lastScan;
    const box = $('#scan');
    if (!s || !s.at || s.at.startsWith('0001')) {
      box.replaceChildren(d.scanning ? 'the first scan is running…' : (d.docs ? 'not indexed: rescan to build the index' : `no .md or .txt files in ${d.dir}: run make prepare, or drop some in and rescan`));
      return;
    }
    const parts = [
      'last scan ', el('b', {}, secs(s.elapsedMs)), ` (IndexBatch ${secs(s.indexMs)}): `,
      el('b', {}, String(s.indexed)), ' indexed, ', el('b', {}, String(s.unchanged)), ' unchanged, ',
      el('b', {}, String(s.failed)), ' failed, ', el('b', {}, String(s.removed)), ' removed',
      s.pending ? [', ', el('b', {}, String(s.pending)), ' not reached'] : null,
    ];
    if (s.error) parts.push(el('div', { class: 'failure' }, 'stopped: ' + s.error));
    for (const [id, why] of Object.entries(s.failures || {})) parts.push(el('div', { class: 'failure' }, `${id}: ${why}`));
    box.replaceChildren(...parts.flat().filter(Boolean));
  }

  // renderBanner offers the reset when the collection holds another model's
  // vectors
  function renderBanner(d) {
    const b = $('#banner');
    const legacy = d.lastScan && d.lastScan.legacy;
    b.hidden = !(d.lastScan && (d.lastScan.mismatch || legacy));
    if (b.hidden) return;
    const btn = el('button', { class: 'button danger' }, 'reset the index and rescan');
    btn.addEventListener('click', () => {
      if (!confirm(legacy ? 'Drop the index\'s old tables and embed every document again?' : 'Empty the index and embed every document again with the model that runs now?')) return;
      busy(btn, async () => {
        await api('api/reset', { method: 'POST' });
        await rescan();
      });
    });
    if (legacy) {
      b.replaceChildren(el('div', { class: 'what' }, el('b', {}, 'The index is in the tables of an older version. '),
        'The store refuses every read and write of it (vector.ErrLegacySchema). The reset drops those tables (Store.DeleteIndex), and the rescan embeds every document again.'), btn);
      return;
    }
    b.replaceChildren(el('div', { class: 'what' }, el('b', {}, 'The index was built by another model. '),
      d.storedModel ? `The store holds this collection's vectors under ${d.storedModel}; ` : 'The store holds this collection under another model; ',
      `the page runs ${info.model}. Vectors of two models never mix, so nothing is indexed or searched until the collection is emptied (Index.Reset).`), btn);
  }

  async function loadDocs() {
    docs = await api('api/docs');
    const list = $('#doclist');
    const rows = [];
    let group = null;
    for (const d of docs) {
      const dir = d.id.includes('/') ? d.id.slice(0, d.id.lastIndexOf('/')) : '';
      if (dir !== group) { group = dir; rows.push(el('div', { class: 'group' }, dir || '.')); }
      const row = el('div', { class: 'doc' + (active === d.id ? ' active' : ''), title: d.id, onclick: () => open(d.id) },
        el('span', { class: 'title' }, d.title),
        el('span', { class: 'meta' }, `${d.chunks} chunk${d.chunks === 1 ? '' : 's'} · ${bytes(d.bytes)} `, statePill(d)),
        d.error ? el('span', { class: 'why' }, d.error) : null);
      row.dataset.id = d.id;
      rows.push(row);
    }
    list.replaceChildren(...rows);
  }

  function mark(id) {
    active = id;
    for (const r of document.querySelectorAll('.doc')) r.classList.toggle('active', r.dataset.id === id);
  }

  // hits renders a Result's hits; a click on a title opens the document at
  // the chunk that matched
  function hits(res) {
    if (!res.hits.length) return [el('p', { class: 'hint' }, 'nothing above the min score')];
    return res.hits.map((h) => {
      const text = el('div', { class: 'text', title: 'click to expand' }, h.text || '(a heading with no text of its own)');
      text.addEventListener('click', () => text.classList.toggle('open'));
      return el('div', { class: 'hit' },
        el('div', { class: 'head' },
          el('span', { class: 'title', title: 'open the document at this chunk', onclick: () => open(h.doc.id, h.seq) }, h.doc.title),
          bar(h.score, info.minScore > 0 ? info.minScore : null),
          el('span', { class: 'score num' }, h.score.toFixed(4))),
        el('div', { class: 'where' }, `${h.doc.id} · chunk ${h.seq + 1} of ${h.doc.chunks || '?'}`, h.heading ? [' · ', el('b', {}, h.heading)] : null),
        text);
    });
  }

  async function search(q) {
    mark(null);
    detail.replaceChildren(el('p', { class: 'hint' }, `searching for “${q}”…`));
    try {
      const res = await api('api/search?q=' + encodeURIComponent(q));
      detail.replaceChildren(
        el('h2', {}, `“${q}”`),
        el('div', { class: 'summary' }, `query embedded, ${res.hits.length} documents in ${took(res.elapsedUs)}; each is scored by its best chunk, shown below it`),
        ...hits(res));
      loadInfo().catch(() => {}); // the first search loads the model
    } catch (e) {
      detail.replaceChildren(el('p', { class: 'error' }, e.message));
      if (e.body && (e.body.mismatch || e.body.legacy)) loadInfo().catch(() => {});
    }
  }

  // open shows a document: its chunks as the index embedded them, and the
  // documents most like it; seq marks the chunk a search hit matched
  async function open(id, seq) {
    mark(id);
    detail.replaceChildren(el('p', { class: 'hint' }, 'loading…'));
    try {
      const [view, similar] = await Promise.all([
        api('api/doc/' + encPath(id)),
        api('api/similar/' + encPath(id)).catch((e) => ({ error: e.message })),
      ]);
      const d = view.doc;
      const facts = el('div', { class: 'facts' });
      const rows = [['path', d.id], ['size', bytes(d.bytes)], ['chunks', `${view.chunks.length} (one vector each)`], ['last scan', d.state + (d.error ? ': ' + d.error : '')]];
      if (view.changed) rows.push(['note', 'the file changed since the last scan: these are the chunks the next scan will embed']);
      for (const [k, v] of rows) facts.append(el('span', {}, k), el('b', {}, v));
      const chunks = view.chunks.map((c) => el('div', { class: 'chunk' + (c.seq === seq ? ' match' : ''), id: 'chunk-' + c.seq },
        el('div', { class: 'head' }, el('span', { class: 'seq' }, `chunk ${c.seq + 1}`), `${c.bytes} bytes`,
          c.heading ? el('b', { title: 'the heading path the chunker puts in front of the chunk: it is embedded with it' }, c.heading) : null,
          c.seq === seq ? el('span', { class: 'pill busy' }, 'matched') : null),
        el('div', { class: 'text' }, c.text || '(a heading with no text of its own)')));
      detail.replaceChildren(
        el('h2', {}, d.title), facts,
        el('h3', {}, 'more like this'),
        similar.error
          ? el('p', { class: 'error' }, similar.error)
          : el('div', { class: 'summary' }, `stored chunk vectors reused, nothing embedded, ${similar.hits.length} documents in ${took(similar.elapsedUs)}`),
        ...(similar.error ? [] : hits(similar)),
        el('h3', {}, `chunks (${view.chunks.length}) · ${info.chunker}`),
        ...(chunks.length ? chunks : [el('p', { class: 'hint' }, 'nothing to embed in this file')]));
      const target = seq === undefined ? null : document.getElementById('chunk-' + seq);
      if (target) target.scrollIntoView({ block: 'center' });
      else window.scrollTo({ top: 0 });
    } catch (e) {
      detail.replaceChildren(el('p', { class: 'error' }, e.message));
    }
  }

  async function rescan() {
    const s = await api('api/rescan', { method: 'POST' });
    toast(s.error
      ? `rescan stopped: ${s.error}`
      : `rescan: ${s.indexed} indexed, ${s.unchanged} unchanged, ${s.failed} failed, ${s.removed} removed in ${secs(s.elapsedMs)}`);
    await Promise.all([loadInfo(), loadDocs()]);
  }

  $('#searchform').addEventListener('submit', (e) => { e.preventDefault(); const q = $('#q').value.trim(); if (q) search(q); });
  $('#rescan').addEventListener('click', (e) => busy(e.target, rescan));

  // while the startup scan runs, look again until it is done
  async function start() {
    const [d] = await Promise.all([loadInfo(), loadDocs()]);
    if (d.scanning) setTimeout(() => start().catch(() => {}), 1500);
  }
  start().catch((e) => toast(e.message));
})();
