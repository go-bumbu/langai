// The auto tagging page: the notebook's tags on the left, its notes on the
// right. Every note shows the tags put on it by hand and the ones the Tagger
// picks among the notebook's tags; "recommend" has the chat model write new
// tags for it, each with the score the embedder gives it.
(() => {
  const $ = (sel) => document.querySelector(sel);
  const infoEl = $('#info');
  const taglist = $('#taglist');
  const notelist = $('#notelist');
  let book = { tags: [], notes: [], suggestMs: 0 };
  let info = {};
  let filter = null;           // a tag id, or null
  const recs = new Map();      // note id → recommendation, or 'loading', or an Error
  const open = new Set();      // note ids whose text is expanded
  const showScores = new Set(); // note ids whose scores are shown

  const enc = encodeURIComponent;
  const s = (n) => (n / 1000).toFixed(2) + ' s';
  const el = (tag, attrs = {}, ...children) => {
    const e = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (k === 'class') e.className = v;
      else if (k.startsWith('on')) e.addEventListener(k.slice(2), v);
      else if (v !== null && v !== undefined && v !== false) e.setAttribute(k, v);
    }
    for (const c of children) if (c !== null && c !== undefined && c !== false) e.append(c);
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

  const api = async (url, opts = {}) => {
    if (opts.json !== undefined) {
      opts = { ...opts, body: JSON.stringify(opts.json), headers: { 'Content-Type': 'application/json' } };
    }
    const r = await fetch(url, opts);
    const body = r.status === 204 ? {} : await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(body.error || r.statusText);
    return body;
  };
  // busy disables a button while fn runs and reports a failure
  const busy = async (btn, fn) => {
    if (btn) btn.disabled = true;
    try { await fn(); } catch (e) { toast(e.message); } finally { if (btn) btn.disabled = false; }
  };

  async function loadInfo() {
    const d = await api('api/info');
    info = d.info;
    const chat = d.chat ? ` (${d.chat})` : '';
    const embedder = d.embedder ? ` (${d.embedder})` : '';
    const chunks = info.chunkBytes > 0 ? ` · notes over ${info.chunkBytes} bytes read in parts` : '';
    infoEl.innerHTML = `existing tags: <b>${info.embedder}</b>${embedder} · name ≥ ${info.nameMinScore}, examples ≥ ${info.memberMinScore}${chunks}<br>` +
      `new tags: <b>${info.completer}</b>${chat} scored by the embedder · cutoff ${info.minScore} · ${d.runsOn || info.accel}`;
    $('#samples').hidden = !info.samples;
    $('#samples-tagged').hidden = !info.samples;
  }

  async function load() {
    book = await api('api/notebook');
    render();
  }

  const tagName = (id) => (book.tags.find((t) => t.id === id) || { name: id }).name;

  function render() {
    if (filter && !book.tags.some((t) => t.id === filter)) filter = null;
    taglist.replaceChildren(...book.tags.map(tagRow));
    const f = $('#filter');
    f.hidden = !filter;
    if (filter) f.replaceChildren(`showing notes tagged ${tagName(filter)} · `, el('a', { onclick: () => { filter = null; render(); } }, 'show all'));
    const notes = filter ? book.notes.filter((n) => n.tags.includes(filter) || n.auto.includes(filter)) : book.notes;
    notelist.replaceChildren(...notes.map(noteCard));
    const tagged = book.notes.filter((n) => n.auto.length).length;
    $('#summary').textContent = book.notes.length
      ? `${notes.length} of ${book.notes.length} notes · ${tagged} with an automatic tag · ${book.tags.length} tags · suggestions for every note took ${s(book.suggestMs)}`
      : 'No notes yet: add one above' + (info.samples ? ', or load the samples.' : '.');
  }

  function tagRow(t) {
    return el('li', { class: filter === t.id ? 'active' : '', title: t.name, onclick: () => { filter = filter === t.id ? null : t.id; render(); } },
      el('span', { class: 'name' }, t.name, t.incomplete
        ? el('span', { class: 'incomplete', title: 'built without one of the notes it was put on by hand: the embedder could not take that note\'s text (Scope.Incomplete). It suggests on its name and its other notes until the notebook next changes.' }, ' ⚠ incomplete')
        : null),
      el('span', { class: 'counts', title: 'notes tagged by hand · notes the Tagger picks it for' }, `${t.hand} · ${t.auto}`),
      el('button', { class: 'icon', title: 'rename', onclick: (e) => { e.stopPropagation(); renameTag(t); } }, '✎'),
      el('button', { class: 'icon del', title: 'delete', onclick: (e) => { e.stopPropagation(); deleteTag(t); } }, '×'));
  }

  function noteCard(n) {
    const scoreOf = new Map(n.scores.map((x) => [x.tagId, x]));
    const scoreTag = (id) => {
      const x = scoreOf.get(id);
      return x ? el('span', { class: 'num' }, x.score.toFixed(2)) : null;
    };
    const hand = n.tags.map((id) => el('span', {
      class: 'chip hand' + (n.auto.includes(id) ? ' agree' : ''),
      title: n.auto.includes(id) ? 'by hand, and the Tagger picks it too' : 'by hand; the Tagger does not pick it',
    }, tagName(id), scoreTag(id), el('button', { title: 'take it off', onclick: () => untag(n.id, id) }, '×')));
    const auto = n.auto.filter((id) => !n.tags.includes(id)).map((id) => el('span', {
      class: 'chip auto', title: `the Tagger's pick #${n.auto.indexOf(id) + 1}; + puts it on by hand, which teaches the tag`,
    }, tagName(id), scoreTag(id), el('button', { title: 'accept: tag by hand', onclick: () => tagNote(n.id, id) }, '+')));
    const have = new Set([...n.tags, ...n.auto].map((id) => tagName(id).toLowerCase()));
    const expected = (n.expected || []).map((name) => el('span', {
      class: 'chip expected ' + (have.has(name.toLowerCase()) ? 'hit' : 'miss'),
      title: have.has(name.toLowerCase()) ? 'expected, and the note has it' : 'expected, but the note has not got it',
    }, name));
    const text = el('div', { class: 'text' + (open.has(n.id) ? ' open' : ''), title: 'click to expand', onclick: (e) => {
      open.has(n.id) ? open.delete(n.id) : open.add(n.id);
      e.currentTarget.classList.toggle('open');
    } }, n.text);
    const recBtn = el('button', { class: 'button small', title: 'have the chat model write tags for this note; the first time loads it' }, recs.has(n.id) ? 'recommend again' : 'recommend new tags');
    recBtn.addEventListener('click', () => recommend(n.id, recBtn));
    const scoresBtn = el('button', { class: 'button small', title: "the note's score against every tag, and the cutoff each tag picks at" }, showScores.has(n.id) ? 'hide scores' : 'scores');
    scoresBtn.addEventListener('click', () => { showScores.has(n.id) ? showScores.delete(n.id) : showScores.add(n.id); render(); });
    const size = `${(n.bytes / 1024).toFixed(1)} KB` + (n.parts > 1 ? ` · read in ${n.parts} parts` : '');
    return el('div', { class: 'note' },
      el('div', { class: 'head' }, el('span', { class: 'title' }, n.title || n.text.split('\n')[0]),
        el('span', { class: 'id', title: n.parts > 1 ? 'longer than the embedder reads at once: each part is embedded and each tag scores by the two best parts' : 'what the Tagger reads' }, size),
        el('span', { class: 'id' }, n.id)),
      n.title ? text : null,
      el('div', { class: 'line' }, el('span', { class: 'label' }, 'by hand'), ...(hand.length ? hand : [el('span', { class: 'none' }, 'none')])),
      el('div', { class: 'line' }, el('span', { class: 'label' }, 'auto'), ...(auto.length ? auto : [el('span', { class: 'none' }, n.auto.length ? 'only the hand tags' : 'nothing clears its cutoff')])),
      expected.length ? el('div', { class: 'line' }, el('span', { class: 'label' }, 'expected'), ...expected) : null,
      n.error ? el('div', { class: 'error' }, n.error) : null,
      el('div', { class: 'foot' }, scoresBtn, recBtn, el('span', { class: 'spacer' }),
        el('button', { class: 'icon del', title: 'delete the note', onclick: () => deleteNote(n) }, 'delete')),
      showScores.has(n.id) ? scoresView(n) : null,
      recView(n));
  }

  // scoresView lists the note's score against every tag, each bar marked
  // where that tag's cutoff is
  function scoresView(n) {
    const rows = n.scores.map((x) => {
      const picked = x.score >= x.cutoff && !x.shadowedBy;
      const title = picked ? 'reaches the cutoff: picked'
        : x.shadowedBy ? `reaches the cutoff, but ${tagName(x.shadowedBy)}, on the same path, fits the note better, so this one is not picked`
          : `${(x.cutoff - x.score).toFixed(3)} short of the cutoff`;
      return el('div', { class: 'rec' + (picked ? '' : ' below'), title },
        el('span', {}, tagName(x.tagId), n.tags.includes(x.tagId) ? el('span', { class: 'existing' }, ' · by hand') : null,
          x.shadowedBy ? el('span', { class: 'existing' }, ` · ${tagName(x.shadowedBy)} fits better`) : null),
        el('span', { class: 'bar' }, el('i', { style: `width:${Math.max(0, Math.min(1, x.score)) * 100}%` }), el('span', { class: 'cut', style: `left:${Math.min(1, x.cutoff) * 100}%` })),
        el('span', { class: 'score' }, x.score.toFixed(3)),
        el('span', { class: 'existing' }, `cutoff ${x.cutoff >= 2 ? 'never' : x.cutoff.toFixed(3)}`));
    });
    return el('div', { class: 'recs' }, ...(rows.length ? rows : [el('div', { class: 'none' }, 'no tags to score against')]),
      el('div', { class: 'meta' }, 'a tag with no hand tags picks at the name cutoff; one with examples at a cutoff mixed from both. Along a path (dev, dev / go) only the tag cleared by more is picked. See the README'));
  }

  function recView(n) {
    const r = recs.get(n.id);
    if (!r) return null;
    if (r === 'loading') return el('div', { class: 'recs' }, el('div', { class: 'meta' }, 'the chat model is writing tags… the first call loads it (a 2.5 GB download the very first time)'));
    if (r instanceof Error) return el('div', { class: 'recs' }, el('div', { class: 'error' }, r.message));
    const tree = r.options && r.options.tree;
    const rows = r.tags.map((t) => {
      const onNote = t.tagId && n.tags.includes(t.tagId);
      const create = tree ? 'create the missing levels of the path and put it on the note' : 'create the tag and put it on the note';
      const btn = onNote
        ? el('span', { class: 'existing' }, 'on the note')
        : el('button', { class: 'button small', title: t.tagId ? `tag the note with ${tagName(t.tagId)}` : create }, t.tagId ? '+ tag' : '+ create');
      if (!onNote) btn.addEventListener('click', () => busy(btn, () => accept(n.id, t, tree)));
      return el('div', { class: 'rec' + (t.above ? '' : ' below'), title: t.above ? 'clears the cutoff' : 'under the cutoff: Generate would drop it' },
        el('span', {}, tree ? pathView(t) : t.name, t.tagId && !tree ? el('span', { class: 'existing' }, ` = ${tagName(t.tagId)}`) : null),
        el('span', { class: 'bar' }, el('i', { style: `width:${Math.max(0, Math.min(1, t.score)) * 100}%` }), el('span', { class: 'cut', style: `left:${r.minScore * 100}%` })),
        el('span', { class: 'score' }, t.score.toFixed(3)),
        btn);
    });
    const kept = r.tags.filter((t) => t.above).length;
    const how = `${r.options.language ? 'in ' + r.options.language : "in the note's language"}${tree ? ', as a tree (grey levels are your tags already)' : ''}`;
    // a long note the model read only in part: its tags are good, some may be missing
    const partial = r.error ? el('div', { class: 'error', title: 'recommend again to retry the windows that failed' }, r.error) : null;
    return el('div', { class: 'recs' }, ...(rows.length ? rows : [el('div', { class: 'none' }, 'the model wrote no usable tag')]),
      partial,
      el('div', { class: 'meta' }, `${r.tags.length} written ${how}, ${kept} clear the cutoff ${r.minScore} · ${s(r.ms)}`));
  }

  // pathView shows a tree tag with the levels that are tags already greyed
  // and the new ones in bold
  function pathView(t) {
    const levels = t.name.split('/').map((l) => l.trim()).filter(Boolean);
    const parts = [];
    levels.forEach((l, i) => {
      if (i > 0) parts.push(' / ');
      parts.push(el('span', { class: i < t.known ? 'known' : 'new' }, l));
    });
    return el('span', { class: 'path' }, ...parts);
  }

  async function recommend(id, btn) {
    recs.set(id, 'loading');
    render();
    btn.disabled = true;
    const q = new URLSearchParams({ lang: $('#lang').value, tree: $('#tree').checked ? '1' : '0' });
    try {
      recs.set(id, await api(`api/notes/${enc(id)}/recommend?${q}`));
    } catch (e) {
      recs.set(id, e);
    }
    render();
    loadInfo().catch(() => {});
  }

  async function accept(noteID, t, tree) {
    let id = t.tagId;
    if (!id) id = (await api('api/tags', { method: 'POST', json: { name: t.name, parents: !!tree } })).id;
    await api(`api/notes/${enc(noteID)}/tags/${enc(id)}`, { method: 'PUT' });
    const r = recs.get(noteID);
    if (r && r.tags) for (const x of r.tags) if (x.name === t.name) x.tagId = id;
    await load();
  }

  const tagNote = (noteID, tagID) => busy(null, async () => {
    await api(`api/notes/${enc(noteID)}/tags/${enc(tagID)}`, { method: 'PUT' });
    await load();
  });
  const untag = (noteID, tagID) => busy(null, async () => {
    await api(`api/notes/${enc(noteID)}/tags/${enc(tagID)}`, { method: 'DELETE' });
    await load();
  });
  const renameTag = (t) => {
    const name = prompt('Rename the tag (its name is what it tags by until it has examples):', t.name);
    if (!name || name === t.name) return;
    busy(null, async () => { await api(`api/tags/${enc(t.id)}`, { method: 'PUT', json: { name } }); await load(); });
  };
  const deleteTag = (t) => {
    if (!confirm(`Delete the tag "${t.name}"? It comes off ${t.hand} notes.`)) return;
    busy(null, async () => { await api(`api/tags/${enc(t.id)}`, { method: 'DELETE' }); await load(); });
  };
  const deleteNote = (n) => {
    if (!confirm(`Delete the note "${n.title || n.id}"?`)) return;
    busy(null, async () => { await api(`api/notes/${enc(n.id)}`, { method: 'DELETE' }); recs.delete(n.id); await load(); });
  };

  $('#addtag').addEventListener('submit', (e) => {
    e.preventDefault();
    const input = $('#tagname');
    busy(e.submitter, async () => {
      await api('api/tags', { method: 'POST', json: { name: input.value } });
      input.value = '';
      await load();
    });
  });
  $('#addnote').addEventListener('submit', (e) => {
    e.preventDefault();
    busy(e.submitter, async () => {
      await api('api/notes', { method: 'POST', json: { title: $('#notetitle').value, text: $('#notetext').value } });
      $('#notetitle').value = '';
      $('#notetext').value = '';
      await load();
    });
  });
  for (const btn of [$('#samples'), $('#samples-tagged')]) {
    btn.addEventListener('click', () => busy(btn, async () => {
      toast('importing the samples: every new document is embedded once…');
      const st = await api(`api/samples?hand=${btn.dataset.hand}`, { method: 'POST' });
      toast(`imported ${st.tags} tags and ${st.notes} notes, put ${st.handTags} tags on by hand, in ${s(st.ms)}`);
      await load();
    }));
  }

  $('#warm').addEventListener('click', (e) => busy(e.target, async () => {
    const st = await api('api/warm', { method: 'POST' });
    toast(`Tagger warm at revision ${st.revision}: ${st.notes} notes and ${st.tags} tags in ${s(st.ms)}` +
      (st.incomplete.length ? `; ${st.incomplete.length} tags built without one of their notes` : '; what the cache held was not embedded again'));
    await Promise.all([loadInfo(), load()]);
  }));

  // the recommendation settings survive a reload, in this browser only
  const remember = (sel, prop) => {
    const el = $(sel);
    try {
      const v = localStorage.getItem('autotag' + sel);
      if (v !== null) el[prop] = prop === 'checked' ? v === '1' : v;
    } catch (e) { /* storage blocked: keep the defaults */ }
    el.addEventListener('change', () => {
      try { localStorage.setItem('autotag' + sel, prop === 'checked' ? (el.checked ? '1' : '0') : el.value); } catch (e) { /* ignore */ }
    });
  };
  remember('#lang', 'value');
  remember('#tree', 'checked');

  loadInfo().then(load).catch((e) => toast(e.message));
})();
