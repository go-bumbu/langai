// The duplicates page: on the left the groups of pictures whose raw hashes
// are within the threshold of each other, with the Hamming distance between
// every two members; on the right the copies of one picture as a MediaIndex
// over the same hash finds them, with the cosine next to the distance it
// stands for.
(() => {
  const { $, el, api, toast, busy, encPath, secs, bytes, bar } = window.ui;
  const detail = $('#detail');
  let kinds = [];
  let failed = {};     // the pictures that could not be hashed, with the reason
  let active = null;   // the catalog picture whose copies are shown
  let lastUpload = null;

  const kindName = () => `${$('#algo').value}-${$('#bits').value}`;
  const kind = () => kinds.find((k) => k.name === kindName()) || { bits: 64, threshold: 4 };
  const threshold = () => Math.max(0, Math.min(kind().bits, parseInt($('#threshold').value, 10) || 0));
  const params = () => `kind=${encodeURIComponent(kindName())}&threshold=${threshold()}`;
  const took = (us) => us < 1000 ? us + ' µs' : us < 1e6 ? (us / 1000).toFixed(1) + ' ms' : (us / 1e6).toFixed(2) + ' s';
  const file = (id) => 'file/' + encPath(id);
  // a 256-bit hash is 64 digits: show its head
  const short = (hash) => hash.length > 16 ? hash.slice(0, 16) + '…' : hash;

  function showFloor() {
    const k = kind();
    $('#threshold').max = k.bits;
    $('#floor').textContent = `= cosine ≥ ${(1 - 2 * threshold() / k.bits).toFixed(4)}`;
  }

  async function loadInfo() {
    const d = await api('api/info');
    kinds = d.kinds;
    failed = d.failed || {};
    const s = d.lastScan;
    const scan = s && s.at && !s.at.startsWith('0001')
      ? ` · scan ${secs(s.elapsedMs)}: ${s.hashed} hashed (${took(s.hashUs)} decoding and hashing, four hashes each), ${s.unchanged} unchanged, ${s.failed} failed, ${s.removed} removed`
      : (d.scanning ? ' · scanning…' : '');
    $('#info').replaceChildren(el('b', {}, 'provider/phash'), ' · no model · ', el('b', {}, kindName()), ` · ${kind().bits} dims, one per bit`, el('br'),
      el('b', {}, String(d.pictures)), ` pictures in ${d.dir}${scan}`);
    showFloor();
    return d;
  }

  function member(m, i) {
    const node = el('div', { class: 'member' + (active === m.id ? ' active' : ''), title: `${m.id}\nclick: find its copies through the MediaIndex`, onclick: () => similar(m.id) },
      el('img', { src: file(m.id), loading: 'lazy', alt: m.name }),
      el('span', { class: 'n' }, String(i + 1)),
      el('div', { class: 'name' }, m.name),
      el('div', { class: 'meta' }, `${m.width}×${m.height} · ${bytes(m.size)}`),
      el('div', { class: 'hash mono', title: m.hash }, short(m.hash)));
    node.dataset.id = m.id;
    return node;
  }

  // matrix is the Hamming distance between every two members of a group
  function matrix(g, t) {
    const n = g.members.length;
    const head = el('tr', {}, el('th', {}, ''), ...g.members.map((_, i) => el('th', {}, String(i + 1))));
    const rows = g.distances.map((row, i) => el('tr', {}, el('th', {}, String(i + 1)),
      ...row.map((d, j) => el('td', { class: i === j ? 'self' : (d > t ? 'over' : ''), title: i === j ? '' : `cosine ${(1 - 2 * d / kind().bits).toFixed(4)}` }, i === j ? '·' : String(d)))));
    return el('table', { class: 'matrix' }, el('caption', {}, `Hamming distance between the ${n} members, in bits`), head, ...rows);
  }

  async function loadGroups() {
    const g = await api('api/groups?' + params());
    const k = g.kind;
    const near = g.nearest
      ? [' · closest pair outside a group: ', el('b', {}, `${g.nearest.distance} bits`), ` (${g.nearest.a} ↔ ${g.nearest.b})`]
      : [];
    $('#summary').replaceChildren(
      el('b', {}, String(g.pictures)), ` pictures, ${g.pairs} pairs compared in ${took(g.elapsedUs)} with Hash.Distance · `,
      el('b', {}, String(g.groups.length)), ' groups holding ', el('b', {}, String(g.duplicates)), ' pictures · ',
      `threshold ${g.threshold} of ${k.bits} bits = cosine ≥ ${g.minScore.toFixed(4)}`, ...near,
      ...Object.entries(failed).map(([id, why]) => el('div', { class: 'failure' }, `not hashed: ${id}: ${why}`)));
    const list = $('#grouplist');
    if (!g.groups.length) {
      list.replaceChildren(el('p', { class: 'hint' }, g.pictures ? 'no two pictures are within the threshold: raise it, or run make prepare for sample copies' : 'no pictures: run make prepare, or drop some into the directory and rescan'));
      return;
    }
    list.replaceChildren(...g.groups.map((gr) => el('div', { class: 'group' },
      el('div', { class: 'head' },
        el('span', { class: 'pill ' + (gr.exact ? 'ok' : 'busy') }, gr.exact ? 'exact duplicates' : 'near-exact duplicates'),
        gr.exact ? 'the same hash: one picture to the algorithm' : `up to ${gr.max} bits apart` + (gr.max > g.threshold ? ' (linked through a closer member)' : '')),
      el('div', { class: 'members' }, ...gr.members.map(member)),
      matrix(gr, g.threshold))));
  }

  function mark(id) {
    active = id;
    for (const m of document.querySelectorAll('.member')) m.classList.toggle('active', m.dataset.id === id);
  }

  // hits renders a Result: each hit's cosine with the distance it stands for
  function hits(res) {
    const bits = res.kind.bits;
    const list = el('div', {},
      el('h3', {}, 'copies, by the MediaIndex'),
      el('div', { class: 'summary' }, (res.stored
        ? `stored vector reused, nothing hashed, ${res.hits.length} hits in ${took(res.elapsedUs)}`
        : `picture hashed and searched, ${res.hits.length} hits in ${took(res.elapsedUs)}`) +
        ` · ${res.copies} within ${res.threshold} bits (cosine ≥ ${res.minScore.toFixed(4)})`));
    if (!res.hits.length) list.append(el('p', { class: 'hint' }, 'the index is empty'));
    for (const h of res.hits) {
      const agrees = h.distance < 0 || h.distance === h.hamming;
      list.append(el('div', { class: 'hit' + (h.within ? '' : ' outside'), title: h.within ? 'within the threshold: a copy' : 'further than the threshold: a floor would drop it', onclick: () => similar(h.picture.id) },
        el('img', { src: file(h.picture.id), loading: 'lazy', alt: '' }),
        el('div', { class: 'meta' },
          el('div', { class: 'title' }, el('span', { title: h.picture.id }, h.picture.name), el('span', { class: 'num' }, h.score.toFixed(4))),
          bar(h.score, res.minScore),
          el('div', { class: 'relation' }, 'cosine ', el('b', {}, h.score.toFixed(4)), ` = 1 − 2·${h.hamming}/${bits} → `, el('b', {}, `${h.hamming} bits`),
            h.distance < 0 ? null : el('span', { class: agrees ? '' : 'differs', title: agrees ? 'the distance of the two raw hashes, computed without the index' : 'the index turns a JPEG upright by its EXIF orientation before hashing; Hash takes the pixels as decoded' },
              ` · Hash.Distance ${h.distance}${agrees ? ' ✓' : ' ≠'}`)))));
    }
    return list;
  }

  function facts(rows) {
    const f = el('div', { class: 'facts' });
    for (const [k, v, cls] of rows) if (v) f.append(el('span', {}, k), el('b', { class: cls || '' }, v));
    return f;
  }

  // similar looks a catalog picture's copies up with its stored vector
  async function similar(id) {
    mark(id);
    lastUpload = null;
    const preview = el('div', { class: 'preview' }, el('img', { src: file(id), alt: '' }));
    detail.replaceChildren(preview, el('p', { class: 'hint' }, 'searching…'));
    try {
      const res = await api('api/similar/' + encPath(id) + '?' + params());
      const p = res.query;
      detail.replaceChildren(preview,
        facts([['path', p.id], ['size', `${p.width}×${p.height} · ${bytes(p.size)}`], [res.kind.name, res.queryHash, 'mono']]),
        hits(res));
    } catch (e) {
      detail.replaceChildren(preview, el('p', { class: 'error' }, e.message));
    }
  }

  async function upload(f) {
    mark(null);
    lastUpload = f;
    const preview = el('div', { class: 'preview' }, el('img', { src: URL.createObjectURL(f), alt: '' }));
    detail.replaceChildren(preview, el('p', { class: 'hint' }, `hashing and searching ${f.name}…`));
    const form = new FormData();
    form.append('file', f);
    try {
      const res = await api('api/search?' + params(), { method: 'POST', body: form });
      detail.replaceChildren(preview,
        facts([['upload', f.name], ['size', bytes(f.size)], [res.kind.name, res.queryHash, 'mono']]),
        hits(res));
    } catch (e) {
      detail.replaceChildren(preview, el('p', { class: 'error' }, e.message));
    }
  }

  // a change of hash or threshold runs everything shown again
  async function refresh() {
    showFloor();
    await Promise.all([loadInfo(), loadGroups()]);
    if (active) await similar(active);
    else if (lastUpload) await upload(lastUpload);
  }

  for (const sel of ['#algo', '#bits']) {
    $(sel).addEventListener('change', () => {
      $('#threshold').value = kind().threshold; // each hash starts at its own default
      refresh().catch((e) => toast(e.message));
    });
  }
  $('#threshold').addEventListener('change', () => refresh().catch((e) => toast(e.message)));
  $('#upload').addEventListener('change', (e) => { if (e.target.files[0]) upload(e.target.files[0]); e.target.value = ''; });
  $('#rescan').addEventListener('click', (e) => busy(e.target, async () => {
    const s = await api('api/rescan', { method: 'POST' });
    toast(`rescan: ${s.hashed} hashed, ${s.unchanged} unchanged, ${s.failed} failed, ${s.removed} removed in ${secs(s.elapsedMs)}`);
    await refresh();
  }));

  // while the startup scan runs, look again until it is done
  async function start(first) {
    const d = await loadInfo();
    if (first) $('#threshold').value = kind().threshold;
    showFloor();
    await loadGroups();
    if (d.scanning) setTimeout(() => start(false).catch(() => {}), 1000);
  }
  start(true).catch((e) => toast(e.message));
})();
