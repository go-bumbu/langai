// The overview page, in tabs: the features, every model with the state of
// its lazy loader, every collection with the model the store recorded for
// it, and the Tagger's embedding cache. The open tab is the URL's hash, so a
// link can point at one. The page reloads itself while a model loads.
(() => {
  const { $, el, api, toast, busy, secs, bytes } = window.ui;
  let timer;

  // the tabs
  const tabs = [...document.querySelectorAll('.tab')];
  const show = (name, focus) => {
    const tab = tabs.find((t) => t.dataset.tab === name && !t.hidden) || tabs[0];
    for (const t of tabs) {
      const on = t === tab;
      t.setAttribute('aria-selected', String(on));
      t.tabIndex = on ? 0 : -1;
      $('#' + t.dataset.tab).hidden = !on;
    }
    if (focus) tab.focus();
    return tab.dataset.tab;
  };
  const fromHash = () => show(location.hash.slice(1));
  for (const t of tabs) {
    t.addEventListener('click', () => history.replaceState(null, '', '#' + show(t.dataset.tab)));
    // left and right move between the tabs, as a tablist does
    t.addEventListener('keydown', (e) => {
      if (e.key !== 'ArrowRight' && e.key !== 'ArrowLeft') return;
      const shown = tabs.filter((x) => !x.hidden);
      const next = shown[(shown.indexOf(t) + (e.key === 'ArrowRight' ? 1 : -1) + shown.length) % shown.length];
      history.replaceState(null, '', '#' + show(next.dataset.tab, true));
    });
  }
  window.addEventListener('hashchange', fromHash);
  fromHash();

  // a tab's count: the text, and ok, warn or bad
  const count = (id, text, kind) => {
    const c = $(id);
    c.textContent = text;
    c.className = 'count' + (kind ? ' ' + kind : '');
  };

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
    for (const [k, v, cls] of rows) {
      if (v === null || v === undefined || v === '') continue;
      f.append(el('span', {}, k), v instanceof Node ? v : el('b', { class: cls || '' }, String(v)));
    }
    return f;
  };
  // a model's identity is its name and then the spec's parameters
  // ("google/siglip2-base-patch16-224;output=pooler_output;…"): the name
  // stands out, the parameters follow, quieter
  const modelID = (id) => {
    if (!id) return null;
    const [name, ...params] = id.split(';');
    return el('b', { class: 'mono' }, name, params.length ? el('span', { class: 'spec' }, params.join(' · ')) : null);
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
        ['model', modelID(m.id)],
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
    return el('div', { class: 'row' + (c.mismatch || c.error ? ' warn' : '') },
      el('div', { class: 'title' }, el('a', { href: c.page, style: 'color:inherit' }, c.name),
        el('span', { class: 'pill' + (c.docs ? ' ok' : '') }, `${c.docs} documents`),
        c.mismatch ? el('span', { class: 'pill warn' }, 'another model') : null,
        c.legacy ? el('span', { class: 'pill warn' }, 'old schema') : null),
      el('div', { class: 'act' }, btn),
      facts([
        ['store', c.backend],
        ['coordinate', `index ${c.index} · partition ${c.partition} · collection ${c.collection}`],
        ['recorded model', c.recorded ? modelID(c.recorded) : '(none: the collection is empty)'],
        ['the page runs', modelID(c.model)],
      ]),
      c.mismatch ? el('div', { class: 'recovery' }, 'The store holds this collection under another model than the page runs: writes and searches fail with vector.ErrModelMismatch. Reset it, then rescan on its page.') : null,
      c.legacy ? el('div', { class: 'recovery' }, 'The store holds this index in the tables of an older version: every read and write fails with vector.ErrLegacySchema. Reset drops those tables (Store.DeleteIndex); then rescan on its page.') : null,
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

  // the size of a model and its download, as one line
  const specLine = (o) => [o.engine, o.dims ? o.dims + ' dims' : '', o.bytes ? bytes(o.bytes) : ''].filter(Boolean).join(' · ');

  // the model a feature runs now: its name, engine, size and what it runs
  // on, the state of its loader, and what it is good at
  const runningOption = (o, m) => el('div', { class: 'running' },
    el('div', { class: 'name' }, el('span', { class: 'id' }, o.name),
      el('span', { class: 'pill ok' }, 'in use'), m ? statePill(m) : null),
    el('div', { class: 'spec' }, specLine(o), m && m.runsOn ? ' · runs on ' + m.runsOn.split(' (')[0] : ''),
    o.does ? el('div', { class: 'does' }, o.does) : null);

  // a model it could run instead: one line, with how to switch to it
  const otherOption = (o) => el('li', {},
    el('div', { class: 'name' }, el('span', { class: 'id' }, o.name), o.select ? el('code', { class: 'select' }, o.select) : null),
    specLine(o) ? el('div', { class: 'spec' }, specLine(o)) : null,
    o.does ? el('div', { class: 'does' }, o.does) : null);

  // one role of a feature: the model in use, then the others
  const featureChoice = (c, models) => {
    const running = c.options.filter((o) => o.key);
    const others = c.options.filter((o) => !o.key);
    return el('div', { class: 'choice' },
      el('div', { class: 'role' }, c.role),
      ...running.map((o) => runningOption(o, models.get(o.key))),
      others.length ? el('div', { class: 'alt' }, running.length ? 'also possible' : 'options') : null,
      others.length ? el('ul', { class: 'others' }, ...others.map(otherOption)) : null,
      c.note ? el('div', { class: 'note' }, c.note) : null);
  };

  // a feature: a row with what it does, the models it runs and what to try
  // on it; the whole row opens its page
  const pageCard = (p, models) => el('a', { class: 'page', href: p.path },
    el('div', { class: 'about' },
      el('b', {}, p.name, el('span', { class: 'open', 'aria-hidden': 'true' }, '→')),
      el('span', { class: 'shows' }, p.shows),
      p.uses && p.uses.length ? el('div', { class: 'uses' }, ...p.uses.map((u) => el('code', {}, u))) : null),
    el('div', { class: 'col models' }, el('h3', {}, 'Models'),
      ...(p.choices || []).map((c) => featureChoice(c, models))),
    p.try && p.try.length ? el('div', { class: 'col try' }, el('h3', {}, 'Try'), el('ul', {}, ...p.try.map((t) => el('li', {}, t)))) : null);

  async function refresh() {
    clearTimeout(timer);
    const d = await api('api/status');
    $('#info').textContent = `up ${duration(d.uptimeMs)}, since ${when(d.started)}`;
    $('#setup').replaceChildren(...(d.setup || []).map((f) => el('div', {}, el('dt', {}, f.name), el('dd', {}, f.value))));
    const models = new Map(d.models.map((m) => [m.key, m]));
    $('#pages').replaceChildren(...(d.pages || []).map((p) => pageCard(p, models)));

    $('#model-rows').replaceChildren(el('div', { class: 'rows' }, ...d.models.map(modelRow)));
    const lazy = d.models.filter((m) => m.lazy);
    const loaded = lazy.filter((m) => m.state === 'loaded').length;
    const failed = d.models.some((m) => m.state === 'failed');
    count('#models-count', lazy.length ? `${loaded}/${lazy.length} loaded` : String(d.models.length),
      failed ? 'bad' : d.models.some((m) => m.state === 'loading') ? '' : loaded ? 'ok' : '');

    $('#collection-rows').replaceChildren(d.collections.length
      ? el('div', { class: 'rows' }, ...d.collections.map(collectionRow))
      : el('div', { class: 'none' }, 'no collections'));
    const trouble = d.collections.filter((c) => c.mismatch || c.error).length;
    count('#collections-count', trouble ? `${trouble} to fix` : String(d.collections.length), trouble ? 'warn' : '');

    // the cache tab exists only when the service has a cache
    $('#tab-cache').hidden = !d.cache;
    if (d.cache) $('#cache-rows').replaceChildren(el('div', { class: 'rows' }, cacheRow(d.cache)));
    fromHash();

    // a load in progress ends on its own: look again
    if (d.models.some((m) => m.state === 'loading')) timer = setTimeout(() => refresh().catch(() => {}), 1500);
  }

  $('#refresh').addEventListener('click', (e) => busy(e.target, refresh));
  refresh().catch((e) => toast(e.message));
})();
