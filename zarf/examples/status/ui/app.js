// The status page: the pages, every model with the state of its lazy
// loader, every collection with the model the store recorded for it, and
// the Tagger's embedding cache. It reloads itself while a model loads.
(() => {
  const { $, el, api, toast, busy, secs, bytes } = window.ui;
  let timer;

  const when = (iso) => new Date(iso).toLocaleString();
  const duration = (ms) => {
    const s = Math.floor(ms / 1000);
    if (s < 90) return s + ' s';
    if (s < 5400) return Math.round(s / 60) + ' min';
    if (s < 172800) return Math.round(s / 3600) + ' h';
    return Math.round(s / 86400) + ' days';
  };
  const facts = (rows) => {
    const f = el('div', { class: 'facts' });
    for (const [k, v, cls] of rows) if (v !== null && v !== undefined && v !== '') f.append(el('span', {}, k), el('b', { class: cls || '' }, String(v)));
    return f;
  };

  // the state of a lazy loader, as a pill
  const statePill = (m) => {
    const kind = { loaded: 'ok', loading: 'busy', failed: 'bad', closed: 'warn' }[m.state] || '';
    return el('span', { class: 'pill ' + (m.lazy ? kind : 'ok') }, m.state);
  };

  function modelRow(m) {
    const btn = m.lazy
      ? el('button', { class: 'button small', title: 'call Load on the lazy loader: start the load, or wait for the one that runs' }, m.state === 'loaded' ? 'loaded' : 'load now')
      : null;
    if (btn) {
      btn.disabled = m.state !== 'not started' && m.state !== 'loading';
      btn.addEventListener('click', () => busy(btn, async () => {
        const load = api(`api/models/${encodeURIComponent(m.key)}/load`, { method: 'POST' });
        setTimeout(refresh, 300); // show "loading" meanwhile
        const st = await load;
        toast(st.state === 'loaded' ? `${m.name}: loaded${st.loadMs ? ' in ' + secs(st.loadMs) : ''}` : `${m.name}: ${st.state}`);
        await refresh();
      }));
    }
    return el('div', { class: 'row' + (m.state === 'failed' ? ' bad' : '') },
      el('div', { class: 'title' }, m.name, statePill(m), el('small', {}, m.lazy ? 'lazy' : 'not lazy')),
      el('div', { class: 'act' }, btn),
      facts([
        ['model', m.id, 'mono'],
        ['dimensions', m.dimensions || null],
        ['runtime', m.runtime],
        ['runs on', m.runsOn],
        ['used by', m.usedBy],
        ['loaded in', m.loadMs ? secs(m.loadMs) + ' (asked from this page)' : null],
        ['why', m.why],
      ]),
      m.error ? el('div', { class: 'problem' }, m.error) : null,
      m.recovery ? el('div', { class: 'recovery' }, m.recovery) : null);
  }

  function collectionRow(c) {
    const btn = c.canReset ? el('button', { class: 'button small danger', title: 'empty the collection: every vector in it goes, and the page embeds its files again on its next scan' }, 'reset') : null;
    if (btn) {
      btn.addEventListener('click', () => {
        if (!confirm(`Reset "${c.name}"? Its ${c.docs} documents and their vectors are deleted; the page's next scan embeds everything again.`)) return;
        busy(btn, async () => {
          const st = await api(`api/collections/${encodeURIComponent(c.key)}/reset`, { method: 'POST' });
          toast(`${c.name}: emptied, ${st.docs} documents left; rescan on its page to build it again`);
          await refresh();
        });
      });
    }
    return el('div', { class: 'row' + (c.mismatch ? ' warn' : '') },
      el('div', { class: 'title' }, el('a', { href: c.page, style: 'color:inherit' }, c.name),
        el('span', { class: 'pill' + (c.docs ? ' ok' : '') }, `${c.docs} documents`),
        c.mismatch ? el('span', { class: 'pill warn' }, 'another model') : null),
      el('div', { class: 'act' }, btn),
      facts([
        ['store', c.backend],
        ['coordinate', `index ${c.index} · partition ${c.partition} · collection ${c.collection}`],
        ['recorded model', c.recorded || '(none: the collection is empty)', 'mono'],
        ['the page runs', c.model, 'mono'],
      ]),
      c.mismatch ? el('div', { class: 'recovery' }, 'The store holds this collection under another model than the page runs: writes and searches fail with vector.ErrModelMismatch. Reset it, then rescan on its page.') : null,
      c.error ? el('div', { class: 'problem' }, c.error) : null);
  }

  function cacheRow(c) {
    const btn = c.canPrune ? el('button', { class: 'button small', title: 'run Tagger.Prune now, with the age the schedule uses' }, 'prune now') : null;
    if (btn) {
      btn.addEventListener('click', () => busy(btn, async () => {
        const st = await api('api/cache/prune', { method: 'POST' });
        toast(st.lastPrune ? `pruned: ${st.lastPrune.removed} embeddings removed` : 'pruned');
        await refresh();
      }));
    }
    const p = c.lastPrune;
    return el('div', { class: 'row' },
      el('div', { class: 'title' }, c.name),
      el('div', { class: 'act' }, btn),
      facts([
        ['store', c.backend],
        ['size', c.bytes >= 0 ? bytes(c.bytes) + ' (the database that holds it; the cache has no count of its own)' : (c.sizeError || 'unknown')],
        ['last prune', p ? `${when(p.at)}: ${p.removed} embeddings removed, of those unused for ${duration(p.unusedForMs)}` : 'none yet'],
      ]),
      p && p.error ? el('div', { class: 'problem' }, p.error) : null);
  }

  async function refresh() {
    clearTimeout(timer);
    const d = await api('api/status');
    $('#info').replaceChildren(`up ${duration(d.uptimeMs)}, since ${when(d.started)}`,
      ...(d.setup || []).flatMap((f) => [' · ', f.name + ' ', el('b', {}, f.value)]));
    $('#pages').replaceChildren(...(d.pages || []).map((p) => el('a', { class: 'page', href: p.path }, el('b', {}, p.name), el('span', {}, p.shows))));
    $('#models').replaceChildren(el('div', { class: 'rows' }, ...d.models.map(modelRow)));
    $('#collections').replaceChildren(d.collections.length
      ? el('div', { class: 'rows' }, ...d.collections.map(collectionRow))
      : el('div', { class: 'none' }, 'no collections'));
    $('#cachesection').hidden = !d.cache;
    if (d.cache) $('#cache').replaceChildren(el('div', { class: 'rows' }, cacheRow(d.cache)));
    // a load in progress ends on its own: look again
    if (d.models.some((m) => m.state === 'loading')) timer = setTimeout(() => refresh().catch(() => {}), 1500);
  }

  $('#refresh').addEventListener('click', (e) => busy(e.target, refresh));
  refresh().catch((e) => toast(e.message));
})();
