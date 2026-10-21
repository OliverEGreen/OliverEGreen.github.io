// Ready: the story page. Specimen controls, the playable keyboard and Ready's overlay scrollbars.
(() => {
  const $ = (s, r = document) => r.querySelector(s);

  /* ---------- 2 · the font ---------- */
  const spec = $('#spec'), inp = $('#spec-in'), show = $('#spec-show'), seg = $('#spec-seg'), ind = $('.segind', seg);
  const track = $('#spec-track'), thumb = $('.thumb', track), fill = $('.fill', track), sval = $('#spec-val');
  const MIN = 24, MAX = 120;
  let size = innerWidth < 600 ? 48 : 72;

  function setText() {
    const t = inp.value.trim() ? inp.value : 'Less, but better.';
    show.textContent = t;
    spec.querySelectorAll('.wv').forEach((w) => { w.textContent = t; });
  }
  inp.addEventListener('input', setText);

  function setSize(v) {
    size = Math.round(Math.min(MAX, Math.max(MIN, v)));
    const p = (size - MIN) / (MAX - MIN) * 100;
    thumb.style.left = p + '%'; fill.style.width = p + '%';
    show.style.fontSize = size + 'px';
    sval.textContent = size + ' px';
    thumb.setAttribute('aria-valuenow', size); thumb.setAttribute('aria-valuetext', size + ' pixels');
  }
  const fromX = (x) => { const r = track.getBoundingClientRect(); return MIN + (MAX - MIN) * Math.min(1, Math.max(0, (x - r.left) / r.width)); };
  track.addEventListener('pointerdown', (e) => {
    if (e.button !== 0) return;
    e.preventDefault(); thumb.focus({ preventScroll: true });
    track.classList.add('drag'); setSize(fromX(e.clientX));
    try { track.setPointerCapture(e.pointerId); } catch (err) { /* optional */ }
    const mv = (ev) => setSize(fromX(ev.clientX));
    const up = () => { track.classList.remove('drag'); track.removeEventListener('pointermove', mv); track.removeEventListener('pointerup', up); track.removeEventListener('pointercancel', up); };
    track.addEventListener('pointermove', mv); track.addEventListener('pointerup', up); track.addEventListener('pointercancel', up);
  });
  thumb.addEventListener('keydown', (e) => {
    const big = e.shiftKey ? 10 : 1;
    const map = { ArrowRight: size + big, ArrowUp: size + big, ArrowLeft: size - big, ArrowDown: size - big, PageUp: size + 10, PageDown: size - 10, Home: MIN, End: MAX };
    if (!(e.key in map)) return;
    e.preventDefault(); setSize(map[e.key]);
  });

  // segmented control: the chosen segment is a raised key that slides between the two
  function placeInd(animate) {
    const on = $('.sg.on', seg); if (!on) return;
    if (!animate) ind.style.transition = 'none';
    ind.style.width = on.offsetWidth + 'px';
    ind.style.transform = 'translateX(' + on.offsetLeft + 'px)';
    ind.style.opacity = '1';
    if (!animate) { void ind.offsetWidth; ind.style.transition = ''; }
  }
  function setFace(face) {
    seg.querySelectorAll('.sg').forEach((b) => { const on = b.dataset.face === face; b.classList.toggle('on', on); b.setAttribute('aria-pressed', on); });
    const cond = face === 'cond';
    show.classList.toggle('cond', cond); spec.classList.toggle('cond-on', cond);
    spec.querySelectorAll('.fv[data-ready]').forEach((f) => { f.textContent = cond ? f.dataset.cond : f.dataset.ready; });
    placeInd(true);
  }
  seg.addEventListener('click', (e) => { const b = e.target.closest('.sg'); if (b && !b.classList.contains('on')) setFace(b.dataset.face); });
  seg.addEventListener('keydown', (e) => {
    if (e.key !== 'ArrowLeft' && e.key !== 'ArrowRight') return;
    const bs = [...seg.querySelectorAll('.sg')], i = bs.indexOf(document.activeElement); if (i < 0) return;
    e.preventDefault(); const n = bs[(i + (e.key === 'ArrowRight' ? 1 : bs.length - 1)) % bs.length]; n.focus(); setFace(n.dataset.face);
  });
  new ResizeObserver(() => placeInd(false)).observe(seg);
  if (document.fonts) document.fonts.ready.then(() => placeInd(false));
  setText(); setSize(size); placeInd(false);

  /* ---------- 3 · our keyboard: built and played by /fun/ready/assets/keyboard.js ---------- */
  // your own keys only play it once you have clicked into it, so the rest of the page scrolls and types as normal
  if (window.ReadyKeyboard) window.ReadyKeyboard.mount($('#kbwrap'), { scope: $('#kbpanel') });

  /* ---------- overlay scrollbars, as in the Ready app ---------- */
  (function installOverlayScrollbars() {
    if (window.__rsb) return; window.__rsb = 1;
    const doc = document, root = doc.scrollingElement || doc.documentElement, map = new Map(), PAD = 2, TH = 10, MIN = 32;
    const layer = doc.createElement('div'); layer.className = 'rsb-layer'; doc.body.appendChild(layer);
    const rectOf = (el) => (el === root ? { top: 0, left: 0, right: innerWidth, bottom: innerHeight, width: innerWidth, height: innerHeight } : el.getBoundingClientRect());
    const can = (el, ax) => {
      if (el === root) return ax === 'y' ? root.scrollHeight > innerHeight + 1 : root.scrollWidth > innerWidth + 1;
      if (!(el instanceof Element) || el === doc.documentElement || el === doc.body) return false;
      const cs = getComputedStyle(el), o = ax === 'y' ? cs.overflowY : cs.overflowX;
      if (o !== 'auto' && o !== 'scroll') return false;
      return ax === 'y' ? el.scrollHeight > el.clientHeight + 1 : el.scrollWidth > el.clientWidth + 1;
    };
    const dims = (el) => (el === root ? { ch: innerHeight, sh: root.scrollHeight, cw: innerWidth, sw: root.scrollWidth } : { ch: el.clientHeight, sh: el.scrollHeight, cw: el.clientWidth, sw: el.scrollWidth });
    const mk = (el, ax) => { const t = doc.createElement('i'); t.className = 'rsb rsb-' + ax; layer.appendChild(t); t.addEventListener('pointerdown', (e) => drag(e, el, ax, t)); return t; };
    const get = (el) => { let s = map.get(el); if (!s) { s = { el, y: mk(el, 'y'), x: mk(el, 'x') }; map.set(el, s); } return s; };
    const place = (s) => {
      const el = s.el;
      if (el !== root && !el.isConnected) { s.y.remove(); s.x.remove(); map.delete(el); return; }
      const r = rectOf(el), d = dims(el);
      if (can(el, 'y')) {
        const tr = r.height - PAD * 2, len = Math.max(MIN, tr * d.ch / d.sh), max = d.sh - d.ch;
        const pos = max > 0 ? (tr - len) * el.scrollTop / max : 0;
        s.y.style.display = ''; s.y.style.left = (r.right - PAD - TH) + 'px'; s.y.style.top = (r.top + PAD + pos) + 'px'; s.y.style.height = len + 'px';
      } else s.y.style.display = 'none';
      if (can(el, 'x')) {
        const tr = r.width - PAD * 2, len = Math.max(MIN, tr * d.cw / d.sw), max = d.sw - d.cw;
        const pos = max > 0 ? (tr - len) * el.scrollLeft / max : 0;
        s.x.style.display = ''; s.x.style.top = (r.bottom - PAD - TH) + 'px'; s.x.style.left = (r.left + PAD + pos) + 'px'; s.x.style.width = len + 'px';
      } else s.x.style.display = 'none';
    };
    const showBar = (s) => { place(s); s.y.classList.add('on'); s.x.classList.add('on'); };
    const drag = (e, el, ax, t) => {
      e.preventDefault(); t.classList.add('drag');
      try { t.setPointerCapture(e.pointerId); } catch (err) { /* optional */ }
      const r = rectOf(el), d = dims(el), tr = (ax === 'y' ? r.height : r.width) - PAD * 2;
      const len = parseFloat(ax === 'y' ? t.style.height : t.style.width) || MIN;
      const max = ax === 'y' ? d.sh - d.ch : d.sw - d.cw;
      const k = max / Math.max(1, tr - len), start = ax === 'y' ? e.clientY : e.clientX, from = ax === 'y' ? el.scrollTop : el.scrollLeft;
      const mv = (ev) => { const dd = (ax === 'y' ? ev.clientY : ev.clientX) - start; if (ax === 'y') el.scrollTop = from + dd * k; else el.scrollLeft = from + dd * k; };
      const up = () => { t.removeEventListener('pointermove', mv); t.removeEventListener('pointerup', up); t.removeEventListener('pointercancel', up); t.classList.remove('drag'); };
      t.addEventListener('pointermove', mv); t.addEventListener('pointerup', up); t.addEventListener('pointercancel', up);
    };
    doc.addEventListener('scroll', (e) => { const el = e.target === doc ? root : e.target; if ((el === root || el instanceof Element) && (can(el, 'y') || can(el, 'x'))) showBar(get(el)); }, true);
    const scan = () => {
      if (can(root, 'y') || can(root, 'x')) showBar(get(root));
      doc.querySelectorAll('body *').forEach((el) => { if (can(el, 'y') || can(el, 'x')) showBar(get(el)); });
      map.forEach(place);
    };
    let timer = 0; const later = () => { clearTimeout(timer); timer = setTimeout(scan, 120); };
    new MutationObserver((recs) => { if (recs.some((r) => !layer.contains(r.target))) later(); })
      .observe(doc.body, { childList: true, subtree: true, attributes: true, attributeFilter: ['class', 'style'] });
    window.addEventListener('resize', later);
    const tick = () => { map.forEach(place); requestAnimationFrame(tick); };
    scan(); if (doc.fonts) doc.fonts.ready.then(scan); requestAnimationFrame(tick);
  })();
})();
