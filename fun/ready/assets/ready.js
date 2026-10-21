/* Ready — the shared helpers for the Fun pieces (olliegreen.info/fun/ready/), taken from the Catalogue web app. Plain JS, no build step.
   Load it at the end of <body>, before the piece's own script; it exposes one global, Ready. */
(function () {
  'use strict';
  const HERE = (document.currentScript && document.currentScript.src) || location.href;
  const BASE = new URL('.', HERE).href;   // the assets folder

  /* ---------------------------------------------------------------- helpers */
  const $ = (s, el = document) => el.querySelector(s);
  const $$ = (s, el = document) => Array.from(el.querySelectorAll(s));
  const esc = (s) => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
  const svg = (d, { fill = false, size = 16, sw = 1.4, style = '' } = {}) => fill
    ? `<svg aria-hidden="true" width="${size}" height="${size}" viewBox="0 0 16 16" fill="currentColor"${style ? ` style="${style}"` : ''}>${d}</svg>`
    : `<svg aria-hidden="true" width="${size}" height="${size}" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="${sw}" stroke-linecap="round" stroke-linejoin="round"${style ? ` style="${style}"` : ''}>${d}</svg>`;
  const CHEV = '<path d="M4 6.5l4 4 4-4"></path>';
  const I = {
    chev: `<span class="chev">${svg(CHEV)}</span>`,
    chevPlain: svg(CHEV),
    left: svg(CHEV, { style: 'transform: rotate(90deg)' }),
    right: svg(CHEV, { style: 'transform: rotate(-90deg)' }),
    bigLeft: svg(CHEV, { size: 24, sw: 1.5, style: 'transform: rotate(90deg)' }),
    bigRight: svg(CHEV, { size: 24, sw: 1.5, style: 'transform: rotate(-90deg)' }),
    back: svg('<path d="M10 3.5L5.5 8l4.5 4.5"></path>'),
    gchev: svg('<path d="M6 4l4 4-4 4"></path>'),
    search: svg('<circle cx="7" cy="7" r="4.5"></circle><path d="M10.5 10.5L14 14"></path>'),
    x: svg('<path d="M4.5 4.5l7 7M11.5 4.5l-7 7"></path>'),
    xs: svg('<path d="M5 5l6 6M11 5l-6 6"></path>', { size: 14 }),
    dl: svg('<path d="M8 2.5v8M4.5 7L8 10.5 11.5 7M3 13.5h10"></path>'),
    up: svg('<path d="M8 12V3M4 7l4-4 4 4M3 14h10"></path>'),
    kebab: svg('<circle cx="8" cy="3.5" r="1.3"></circle><circle cx="8" cy="8" r="1.3"></circle><circle cx="8" cy="12.5" r="1.3"></circle>', { fill: true }),
    arrow: svg('<path d="M8 3.5v9M4.5 9L8 12.5 11.5 9"></path>'),
    sortUp: svg('<path d="M8 12.5v-9M4.5 7L8 3.5 11.5 7"></path>'),
    filter: svg('<path d="M2.5 3.5h11l-4.25 5v4l-2.5 1v-5z"></path>'),
    group: svg('<path d="M2.5 4h11M5 8h8.5M5 12h8.5M2.5 8v4"></path>'),
    width: svg('<path d="M2.5 8h11M5 5.5L2.5 8 5 10.5M11 5.5l2.5 2.5L11 10.5"></path>'),
    restore: svg('<path d="M3 8a5 5 0 1 0 1.5-3.5"></path><path d="M3 2.5V5h2.5"></path>'),
    pin: svg('<path d="M6 2.5h4M7 2.5v4l-2.5 3h7L9 6.5v-4M8 9.5v4"></path>'),
    trash: svg('<path d="M3 4.5h10M6.5 4.5V3h3v1.5M4.5 4.5l.6 8.5h5.8l.6-8.5"></path>'),
    collapse: svg('<path d="M4.5 2.5L8 6l3.5-3.5M4.5 13.5L8 10l3.5 3.5"></path>'),
    expand: svg('<path d="M4.5 6L8 2.5 11.5 6M4.5 10L8 13.5 11.5 10"></path>'),
    tick: svg('<path d="M3.5 8.5l3 3 6-7"></path>', { sw: 1.8 }),
    ctick: svg('<path d="M3.5 8.5l3 3 6-7"></path>', { size: 12, sw: 2.1 }),
    bigTick: svg('<path d="M3.5 8.5l3 3 6-7"></path>', { size: 24, sw: 1.6 }),
    device: svg('<rect x="2.5" y="3" width="11" height="7.5" rx="1"></rect><path d="M6 13.5h4M8 10.5v3"></path>'),
    folder: svg('<path d="M2.5 4.5a1 1 0 0 1 1-1h3l1.5 1.5h4.5a1 1 0 0 1 1 1v6a1 1 0 0 1-1 1h-9a1 1 0 0 1-1-1z"></path>'),
    folderOpen: svg('<path d="M2.5 12V4.5a1 1 0 0 1 1-1h3l1.5 1.5h4a1 1 0 0 1 1 1V7"></path><path d="M2.5 12l1.6-4.2a1 1 0 0 1 .9-.6h8.6a.6.6 0 0 1 .6.8L12.6 12a1 1 0 0 1-.9.6H3.1a.6.6 0 0 1-.6-.6z"></path>'),
    file: svg('<path d="M4 2.5h5l3 3v8a.5.5 0 0 1-.5.5h-7.5a.5.5 0 0 1-.5-.5v-10.5a.5.5 0 0 1 .5-.5z"></path><path d="M9 2.5v3h3"></path>'),
    openx: svg('<path d="M9.5 3.5h3v3M12.5 3.5L7 9M11 9.5v3a.5.5 0 0 1-.5.5h-7a.5.5 0 0 1-.5-.5v-7a.5.5 0 0 1 .5-.5h3"></path>'),
    rename: svg('<path d="M3 13h3l7-7-3-3-7 7z"></path><path d="M8.5 4.5l3 3"></path>'),
    move: svg('<path d="M2.5 5a1 1 0 0 1 1-1h3l1.5 1.5h4.5a1 1 0 0 1 1 1V12a1 1 0 0 1-1 1h-9a1 1 0 0 1-1-1z"></path><path d="M7 9h4M9.5 7.5L11 9l-1.5 1.5"></path>'),
    history: svg('<path d="M3 8a5 5 0 1 0 1.5-3.5"></path><path d="M3 2.5V5h2.5M8 5.5V8l2 1.5"></path>'),
    overview: svg('<rect x="2.5" y="2.5" width="4.5" height="4.5" rx=".8"></rect><rect x="9" y="2.5" width="4.5" height="4.5" rx=".8"></rect><rect x="2.5" y="9" width="4.5" height="4.5" rx=".8"></rect><rect x="9" y="9" width="4.5" height="4.5" rx=".8"></rect>'),
    projects: svg('<rect x="2.5" y="3" width="11" height="10" rx="1"></rect><path d="M2.5 6.5h11"></path>'),
    plus: svg('<path d="M8 3.5v9M3.5 8h9"></path>'),
    minus: svg('<path d="M3.5 8h9"></path>'),
    cal: svg('<rect x="2.5" y="3.5" width="11" height="10" rx="1"></rect><path d="M2.5 6.5h11M5.5 2v3M10.5 2v3"></path>'),
    share: svg('<path d="M8 10V2.5M5 5.5l3-3 3 3M3.5 9v3.5a1 1 0 0 0 1 1h7a1 1 0 0 0 1-1V9"></path>'),
    picks: svg('<path d="M2.5 4.5l1.5 1.5 2.5-3M2.5 11l1.5 1.5 2.5-3M8.5 4.5h5M8.5 11h5"></path>'),
    empty: svg('<path d="M2.5 9.5l2-6h7l2 6M2.5 9.5v3a1 1 0 0 0 1 1h9a1 1 0 0 0 1-1v-3M2.5 9.5h3l1 1.5h3l1-1.5h3"></path>', { size: 24, sw: 1.3 }),
    nosearch: svg('<circle cx="7" cy="7" r="4.5"></circle><path d="M10.5 10.5L14 14M5.5 5.5l3 3M8.5 5.5l-3 3"></path>', { size: 24, sw: 1.3 }),
  };
  const cboxHtml = (state, attrs = '') => `<span class="cbox ${state}" ${attrs}>${I.ctick}<i></i></span>`;
  /* search matches: solid orange, ink text */
  const hl = (text, q) => {
    const s = String(text); q = (q || '').trim(); if (!q) return esc(s);
    const lo = s.toLowerCase(), qq = q.toLowerCase(); let i = 0, out = '';
    for (;;) { const j = lo.indexOf(qq, i); if (j < 0) { out += esc(s.slice(i)); break; } out += esc(s.slice(i, j)) + '<span class="hit">' + esc(s.slice(j, j + qq.length)) + '</span>'; i = j + qq.length; }
    return out;
  };
  const reducedMotion = () => !!(window.matchMedia && matchMedia('(prefers-reduced-motion: reduce)').matches);
  /* restart a .swap fade on an element whose contents just changed */
  const swap = (el) => { if (!el) return; el.classList.remove('swap'); void el.offsetWidth; el.classList.add('swap'); };

  /* ---------------------------------------------------------------- dates: the catalogue's "now" is fixed */
  const NOW = new Date(2026, 9, 20, 12, 47);
  const DAYNAMES = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];
  const MONTHS = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'];
  const MON = MONTHS.map((m) => m.slice(0, 3));
  const pad = (n) => String(n).padStart(2, '0');
  function dayLabel(d) {
    const a = new Date(d); a.setHours(0, 0, 0, 0);
    const b = new Date(NOW); b.setHours(0, 0, 0, 0);
    const diff = Math.round((b - a) / 864e5);
    return diff === 0 ? 'Today' : diff === 1 ? 'Yesterday' : `${DAYNAMES[d.getDay()]} ${d.getDate()} ${MON[d.getMonth()]}`;
  }
  const savedLabel = (d) => `${dayLabel(d)}, ${pad(d.getHours())}:${pad(d.getMinutes())}`;

  /* ---------------------------------------------------------------- photos of the designs (Wikimedia Commons), served as files */
  const CC = { 'CC BY-SA 2.0': 'https://creativecommons.org/licenses/by-sa/2.0/', 'CC BY-SA 3.0': 'https://creativecommons.org/licenses/by-sa/3.0/',
    'CC BY-SA 4.0': 'https://creativecommons.org/licenses/by-sa/4.0/', CC0: 'https://creativecommons.org/publicdomain/zero/1.0/' };
  const PHOTOS = {};
  [['sk4', 'SK 4', 'Braun SK 4.1.jpg', 'With Associates', 'CC BY-SA 2.0', 800, 600],
    ['t52', 'T 52', 'Braun T 52 radio.jpg', 'Kaldari', 'CC0', 800, 533],
    ['et66', 'ET 66', '1987 Braun calculator ET66 by Dieter Rams (13964223413).jpg', 'Pinot Dita', 'CC BY-SA 2.0', 609, 900],
    ['tp1', 'TP 1', 'DieterRamsTP1Perspective.jpg', 'PeterAjtony', 'CC BY-SA 4.0', 747, 900],
    ['606', '606', '606-Universal-Shelving-System.jpg', 'Vitsoe', 'CC BY-SA 3.0', 800, 587],
    ['t1000', 'T 1000', 'Braun T1000 - Design Museum, Kensington - London - DSC01598.jpg', 'Daderot', 'CC0', 800, 632],
  ].forEach(([k, name, file, author, licence, w, h]) => {
    PHOTOS[k] = { key: k, name, file, author, licence, w, h, src: BASE + 'photos/' + k + '.jpg', licenceUrl: CC[licence],
      commons: 'https://commons.wikimedia.org/wiki/File:' + file.replace(/ /g, '_'), credit: `Photo: ${author}, ${licence}, via Wikimedia Commons` };
  });
  const POS = { et66: 'center 28%', tp1: 'center 40%' };
  /* an <img> of a design, lazy-loaded: cls "cimg" (card top), "hthumb" (row thumbnail) or "" (fills an .ocard) */
  function photo(k, label, cls = 'cimg') {
    const p = PHOTOS[k]; if (!p) return '';
    return `<img class="${cls ? cls + ' ' : ''}ph" src="${p.src}" width="${p.w}" height="${p.h}" alt="${esc(label || p.name)} photo" title="${esc(p.credit)}" loading="lazy" decoding="async" style="object-position: ${POS[k] || 'center'}">`;
  }
  /* the visible credits line: each photo's author, licence and Commons page */
  function creditsHtml(keys) {
    const parts = keys.map((k) => { const p = PHOTOS[k]; return `<a class="plain" href="${p.commons}">${esc(p.name)}</a> by ${esc(p.author)}, <a class="plain" href="${p.licenceUrl}" rel="license">${esc(p.licence)}</a>`; });
    return `<p class="credits hint">Photos from Wikimedia Commons, cropped and resized: ${parts.join('; ')}.</p>`;
  }
  const statusHtml = (st, text) => (st === 'live' ? `<span class="status"><span class="dot"></span>${text}</span>` : `<span class="status ${st || ''}">${text}</span>`);

  /* ---------------------------------------------------------------- in-place state changes: the control stays in the page, so its transition runs */
  function setCbox(el, state) {
    if (!el) return;
    el.classList.toggle('on', state === 'on'); el.classList.toggle('mixed', state === 'mixed');
    if (el.getAttribute('role') === 'checkbox') el.setAttribute('aria-checked', state === 'on' ? 'true' : state === 'mixed' ? 'mixed' : 'false');
  }
  function setToggle(el, on) {
    el.classList.toggle('on', on); el.setAttribute('aria-checked', on);
    const t = el.nextElementSibling; if (t && t.classList.contains('tgtext')) { t.classList.toggle('on', on); t.textContent = on ? 'On' : 'Off'; }
  }
  /* a segmented option or a radio tile: the chosen one rises, its siblings settle; a sliding segment's indicator follows */
  function setChoice(el) {
    $$(':scope > button', el.parentNode).forEach((b) => { const on = b === el; b.classList.toggle('on', on); b.setAttribute(b.getAttribute('role') === 'tab' ? 'aria-selected' : 'aria-checked', on); if (b.getAttribute('role') === 'tab') b.tabIndex = on ? 0 : -1; const r = $('.rbox', b); r && r.classList.toggle('on', on); });
    if (el.parentNode.classList.contains('slide')) syncTabs(el.parentNode.parentNode);
  }
  const toggleHtml = (on, attrs = '') => `<span class="tgrow"><button class="tg${on ? ' on' : ''}" type="button" role="switch" aria-checked="${on}" ${attrs}><span class="knob"><span class="kdot"></span></span></button><span class="tgtext${on ? ' on' : ''}" aria-hidden="true">${on ? 'On' : 'Off'}</span></span>`;
  /* a search field: magnifier until there is text, then the clear key in its place */
  const searchHtml = ({ id, label, placeholder, value = '' }) =>
    `<label class="field search${value ? ' has' : ''}"><span class="sr">${esc(label)}</span><input type="text" id="${id}" placeholder="${esc(placeholder || label)}" value="${esc(value)}" spellcheck="false" autocomplete="off"><button class="fclear" type="button" aria-label="Clear search" data-tip="Clear search" data-act="clearSearch">${I.x}</button><span class="ico">${I.search}</span></label>`;

  /* ---------------------------------------------------------------- toasts */
  function toasts() { let b = $('#toasts'); if (!b) { b = document.createElement('div'); b.className = 'toasts'; b.id = 'toasts'; b.setAttribute('aria-live', 'polite'); document.body.appendChild(b); } return b; }
  function toast(msg, action) {
    const t = document.createElement('div');
    t.className = 'toast'; t.setAttribute('role', 'status');
    t.innerHTML = `<span>${esc(msg)}</span>${action ? `<button class="k" type="button">${esc(action.label)}</button>` : ''}<button class="tx" type="button" aria-label="Dismiss">${I.xs}</button>`;
    const kill = () => { t.classList.add('out'); setTimeout(() => t.remove(), 160); };
    if (action) $('.k', t).onclick = () => { action.run(); kill(); };
    $('.tx', t).onclick = kill;
    toasts().appendChild(t);
    setTimeout(kill, action ? 5000 : 3200);
    return t;
  }

  /* ---------------------------------------------------------------- menus: the only lifted surface besides dialogs */
  let MENU = null;
  function closeMenu(refocus) {
    if (!MENU) return;
    const { veil, el, anchor, onClose } = MENU;
    veil.remove(); el.remove(); anchor && anchor.classList.remove('open');
    if (anchor) anchor.setAttribute('aria-expanded', 'false');
    MENU = null;
    if (onClose) onClose();
    if (refocus && anchor && document.contains(anchor)) anchor.focus();
  }
  function placeFloating(el, anchor, align, atPoint) {
    const r = anchor.getBoundingClientRect(), w = el.offsetWidth, h = el.offsetHeight;
    let x = atPoint ? atPoint.x : (align === 'right' ? r.right - w : r.left);
    let y = atPoint ? atPoint.y : r.bottom + 4;
    if (y + h > innerHeight - 8) y = atPoint ? innerHeight - 8 - h : Math.max(8, r.top - h - 4);
    x = Math.max(8, Math.min(x, innerWidth - w - 8));
    el.style.left = x + 'px'; el.style.top = y + 'px';
  }
  /* items: { label, icon, danger, tick, check, count, keep, disabled, run } | { group } | { gap: true }
     opts: { align: 'right', width, refresh: () => items (for kept menus), onClose, at: { x, y } } */
  function openMenu(anchor, items, opts = {}) {
    closeMenu();
    const veil = document.createElement('div'); veil.className = 'veil';
    const el = document.createElement('div'); el.className = 'menu'; el.setAttribute('role', 'menu'); el.tabIndex = -1;
    if (opts.width) el.style.width = Math.min(opts.width, innerWidth - 16) + 'px';
    el.innerHTML = items.map((it, i) => {
      if (it.gap) return '<div class="mgap"></div>';
      if (it.group) return `<div class="mgroup">${esc(it.group)}</div>`;
      const pre = it.check !== undefined ? cboxHtml(it.check, 'aria-hidden="true"') : (it.icon ? `<span class="mico">${it.icon}</span>` : '');
      const end = it.tick ? `<span class="mend">${I.tick}</span>` : (it.count !== undefined ? `<span class="mcount">${it.count}</span>` : '');
      const role = it.check !== undefined ? 'menuitemcheckbox' : (it.tick !== undefined ? 'menuitemradio' : 'menuitem');
      return `<button class="mrow${it.danger ? ' danger' : ''}" type="button" role="${role}" data-i="${i}"${it.disabled ? ' disabled' : ''}${it.check !== undefined ? ` aria-checked="${it.check === 'on'}"` : ''}${it.tick !== undefined ? ` aria-checked="${!!it.tick}"` : ''}>${pre}<span class="ell">${esc(it.label)}</span>${end}</button>`;
    }).join('');
    document.body.append(veil, el);
    placeFloating(el, anchor, opts.align, opts.at);
    anchor.classList.add('open'); anchor.setAttribute('aria-expanded', 'true');
    MENU = { veil, el, anchor, onClose: opts.onClose };
    veil.addEventListener('click', () => closeMenu());
    veil.addEventListener('contextmenu', (e) => { e.preventDefault(); closeMenu(); });
    el.addEventListener('click', (e) => {
      const b = e.target.closest('.mrow'); if (!b) return;
      const it = items[+b.dataset.i];
      if (it.run) it.run();
      if (it.keep) {
        if (opts.refresh) items = opts.refresh();
        $$('.mrow[data-i]', el).forEach((row) => { const x = items[+row.dataset.i]; if (x && x.check !== undefined) { setCbox($('.cbox', row), x.check); row.setAttribute('aria-checked', x.check === 'on'); } });
        b.focus();
      } else closeMenu(true);
    });
    el.addEventListener('keydown', (e) => {
      if (e.key === 'Escape') { e.preventDefault(); e.stopPropagation(); closeMenu(true); return; }
      if (e.key === 'Tab') { e.preventDefault(); closeMenu(true); return; }
      if (e.key === 'ArrowDown' || e.key === 'ArrowUp' || e.key === 'Home' || e.key === 'End') {
        e.preventDefault();
        const rows = $$('.mrow:not([disabled])', el), i = rows.indexOf(document.activeElement);
        const n = e.key === 'Home' ? rows[0] : e.key === 'End' ? rows[rows.length - 1] : rows[(i + (e.key === 'ArrowDown' ? 1 : -1) + rows.length) % rows.length];
        n && n.focus();
      }
    });
    const first = $('.mrow:not([disabled])', el); (first || el).focus();
    return el;
  }
  /* a lifted surface holding any markup (a calendar, a small form); bind(el) wires it up */
  function openPopover(anchor, html, bind, opts = {}) {
    closeMenu();
    const veil = document.createElement('div'); veil.className = 'veil';
    const el = document.createElement('div'); el.className = 'menu'; el.tabIndex = -1;
    el.innerHTML = html;
    document.body.append(veil, el);
    placeFloating(el, anchor, opts.align, opts.at);
    anchor.classList.add('open');
    MENU = { veil, el, anchor, onClose: opts.onClose };
    veil.addEventListener('click', () => closeMenu());
    el.addEventListener('keydown', (e) => { if (e.key === 'Escape') { e.preventDefault(); e.stopPropagation(); closeMenu(true); } });
    bind && bind(el);
    el.focus();
    return el;
  }

  /* ---------------------------------------------------------------- dialogs: a scrim, a lifted panel; Escape or the scrim closes it, focus stays inside and returns after */
  let DLG = null;
  function openDialog({ id = 'dialog', labelledby = 'dlg-title', onClose, width } = {}) {
    closeDialog(); closeMenu();
    const scrim = document.createElement('div'); scrim.className = 'scrim'; scrim.id = 'scrim';
    const dlg = document.createElement('div'); dlg.className = 'dialog'; dlg.id = id; dlg.setAttribute('role', 'dialog'); dlg.setAttribute('aria-modal', 'true'); dlg.setAttribute('aria-labelledby', labelledby); dlg.tabIndex = -1;
    if (width) dlg.style.width = `min(${width}px, calc(100vw - 48px))`;
    document.body.append(scrim, dlg);
    DLG = { scrim, dlg, prev: document.activeElement, onClose };
    scrim.addEventListener('click', () => closeDialog());
    dlg.addEventListener('keydown', (e) => {
      if (e.key !== 'Tab') return;
      const f = $$('button:not([disabled]), [href], input, textarea, select, [tabindex="0"]', dlg).filter((x) => x.offsetParent !== null);
      if (!f.length) return;
      const a = f[0], z = f[f.length - 1];
      if (e.shiftKey && (document.activeElement === a || document.activeElement === dlg)) { e.preventDefault(); z.focus(); }
      else if (!e.shiftKey && document.activeElement === z) { e.preventDefault(); a.focus(); }
    });
    requestAnimationFrame(() => { if (DLG && DLG.dlg === dlg && !dlg.contains(document.activeElement)) dlg.focus(); });
    return dlg;
  }
  function closeDialog() {
    if (!DLG) return;
    const { scrim, dlg, prev, onClose } = DLG;
    DLG = null; dlg.remove(); scrim.remove();
    if (onClose) onClose();
    if (prev && document.contains(prev)) prev.focus();
  }

  /* ---------------------------------------------------------------- folder tabs and sliding segments: the chosen key is one indicator that slides */
  function syncTabs(root = document) {
    const bars = root.matches && root.matches('.fxbar, .seg.slide') ? [root] : $$('.fxbar, .seg.slide', root);
    bars.forEach((bar) => {
      const on = $('.fxt.on, .sg.on', bar), ind = $('.fxind, .segind', bar);
      if (!on || !ind) return;
      const x = on.offsetLeft, w = on.offsetWidth;
      if (!w || (ind._x === x && ind._w === w)) return;
      const first = ind._x === undefined;
      if (first) ind.style.transition = 'none';
      ind.style.transform = `translateX(${x}px)`; ind.style.width = w + 'px'; ind.style.opacity = '1';
      ind._x = x; ind._w = w;
      if (first) requestAnimationFrame(() => requestAnimationFrame(() => { ind.style.transition = ''; }));
    });
  }
  /* id names the tab list (and comes back in the ready:tab event); panel is the id of the tabpanel it controls */
  function tabsHtml(id, tabs, cur, full = false, panel = '') {
    const slug = String(id).toLowerCase().replace(/[^a-z0-9]+/g, '-');
    return `<div class="fxbar${full ? ' full' : ''}"><i class="fxind" aria-hidden="true"></i><div class="fxtabs" role="tablist" aria-label="${esc(id)}">` +
      tabs.map((t, i) => `<button class="fxt${i === cur ? ' on' : ''}" type="button" role="tab" id="${slug}-tab-${i}" aria-selected="${i === cur}" tabindex="${i === cur ? 0 : -1}"${panel ? ` aria-controls="${panel}"` : ''} data-act="tab" data-tabs="${esc(id)}" data-i="${i}">${esc(t)}</button>`).join('') +
      '</div></div>';
  }
  function updateTabs(btn, i) {
    const bar = btn.closest('.fxbar');
    $$('.fxt', bar).forEach((t, j) => { t.classList.toggle('on', j === i); t.setAttribute('aria-selected', j === i); t.tabIndex = j === i ? 0 : -1; });
    syncTabs(bar);
  }

  /* ---------------------------------------------------------------- actions: one delegated click handler; pieces add theirs with Ready.on({ name(el, e) {} }) */
  const ACTS = {
    /* folder tab: slide to it, then tell the piece (document 'ready:tab' event, detail { tabs, i }) */
    tab(el) {
      if (el.classList.contains('on')) return;
      const i = +el.dataset.i; updateTabs(el, i);
      el.dispatchEvent(new CustomEvent('ready:tab', { bubbles: true, detail: { tabs: el.dataset.tabs, i, el } }));
    },
    /* a search field's clear key: empties the field and sends it an input event, so the piece's own input handler runs */
    clearSearch(el) {
      const f = el.closest('.field'), q = f && $('input', f); if (!q) return;
      q.value = ''; f.classList.remove('has'); q.dispatchEvent(new Event('input', { bubbles: true })); q.focus();
    },
    toastMsg(el) { toast(el.dataset.msg); },
    closeDialog() { closeDialog(); },
  };
  const on = (map) => Object.assign(ACTS, map);
  document.addEventListener('click', (e) => {
    const el = e.target.closest('[data-act]');
    if (!el || el.disabled) return;
    const fn = ACTS[el.dataset.act];
    if (fn) { if (el.tagName === 'A' && el.getAttribute('href') === '#') e.preventDefault(); fn(el, e); }
  });
  document.addEventListener('input', (e) => {
    const f = e.target.closest && e.target.closest('.field.search'); if (f) f.classList.toggle('has', !!e.target.value);
  });
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && DLG && !MENU) { e.preventDefault(); closeDialog(); return; }
    const t = e.target;
    if (!t.classList) return;
    // Escape in a search field with text clears it
    if (e.key === 'Escape' && t.tagName === 'INPUT' && t.value && t.closest('.field.search')) { e.preventDefault(); ACTS.clearSearch(t); return; }
    // folder tabs and tab-like segments: arrow keys move along the strip, Home and End jump to its ends
    const strip = (t.classList.contains('fxt') && t.closest('.fxbar')) || (t.classList.contains('sg') && t.getAttribute('role') === 'tab' && t.closest('.seg'));
    if (strip && ['ArrowRight', 'ArrowLeft', 'Home', 'End'].includes(e.key)) {
      e.preventDefault();
      const tabs = $$(t.classList.contains('fxt') ? '.fxt' : '.sg', strip), i = tabs.indexOf(t);
      const n = e.key === 'Home' ? tabs[0] : e.key === 'End' ? tabs[tabs.length - 1] : tabs[(i + (e.key === 'ArrowRight' ? 1 : -1) + tabs.length) % tabs.length];
      n.focus(); n.click();
    }
  });
  window.addEventListener('resize', () => syncTabs());
  if (document.fonts) document.fonts.ready.then(() => syncTabs());

  /* ---------------------------------------------------------------- overlay scrollbars: a warm stone thumb floats over whatever can scroll and stays while it can; no track, no space taken */
  function installOverlayScrollbars() {
    if (window.__rsb) return; window.__rsb = 1;
    const doc = document, root = doc.scrollingElement || doc.documentElement, map = new Map(), PAD = 2, TH = 10, MIN = 32;
    const layer = doc.createElement('div'); layer.className = 'rsb-layer'; layer.setAttribute('aria-hidden', 'true'); doc.body.appendChild(layer);
    const rectOf = (el) => (el === root ? { top: 0, left: 0, right: innerWidth, bottom: innerHeight, width: innerWidth, height: innerHeight } : el.getBoundingClientRect());
    const can = (el, ax) => {
      if (el === root) return ax === 'y' ? root.scrollHeight > innerHeight + 1 : root.scrollWidth > innerWidth + 1;
      if (!(el instanceof Element) || el === doc.documentElement || el === doc.body || el.closest('.strip, .fxtabs')) return false;
      const cs = getComputedStyle(el), o = ax === 'y' ? cs.overflowY : cs.overflowX;
      if (o !== 'auto' && o !== 'scroll') return false;
      return ax === 'y' ? el.scrollHeight > el.clientHeight + 1 : el.scrollWidth > el.clientWidth + 1;
    };
    const dims = (el) => (el === root ? { ch: innerHeight, sh: root.scrollHeight, cw: innerWidth, sw: root.scrollWidth } : { ch: el.clientHeight, sh: el.scrollHeight, cw: el.clientWidth, sw: el.scrollWidth });
    const mk = (el, ax) => {
      const t = doc.createElement('i'); t.className = 'rsb rsb-' + ax; layer.appendChild(t);
      t.addEventListener('pointerdown', (e) => drag(e, el, ax, t));
      return t;
    };
    const get = (el) => { let s = map.get(el); if (!s) { s = { el, y: mk(el, 'y'), x: mk(el, 'x'), drag: 0 }; map.set(el, s); } return s; };
    const place = (s) => {
      const el = s.el;
      if (el !== root && !el.isConnected) { s.y.remove(); s.x.remove(); map.delete(el); return; }
      const r = rectOf(el), d = dims(el);
      if (can(el, 'y')) {
        const track = r.height - PAD * 2, len = Math.max(MIN, track * d.ch / d.sh), max = d.sh - d.ch;
        const pos = max > 0 ? (track - len) * (el === root ? root.scrollTop : el.scrollTop) / max : 0;
        s.y.style.display = ''; s.y.style.left = (r.right - PAD - TH) + 'px'; s.y.style.top = (r.top + PAD + pos) + 'px'; s.y.style.height = len + 'px';
      } else s.y.style.display = 'none';
      if (can(el, 'x')) {
        const track = r.width - PAD * 2, len = Math.max(MIN, track * d.cw / d.sw), max = d.sw - d.cw;
        const pos = max > 0 ? (track - len) * el.scrollLeft / max : 0;
        s.x.style.display = ''; s.x.style.top = (r.bottom - PAD - TH) + 'px'; s.x.style.left = (r.left + PAD + pos) + 'px'; s.x.style.width = len + 'px';
      } else s.x.style.display = 'none';
    };
    const show = (s) => { place(s); s.y.classList.add('on'); s.x.classList.add('on'); };
    const drag = (e, el, ax, t) => {
      e.preventDefault(); const s = get(el); s.drag = 1; t.classList.add('drag');
      try { t.setPointerCapture(e.pointerId); } catch (err) { /* capture is optional */ }
      const r = rectOf(el), d = dims(el), track = (ax === 'y' ? r.height : r.width) - PAD * 2;
      const len = parseFloat(ax === 'y' ? t.style.height : t.style.width) || MIN;
      const max = ax === 'y' ? d.sh - d.ch : d.sw - d.cw;
      const k = max / Math.max(1, track - len), start = ax === 'y' ? e.clientY : e.clientX, from = ax === 'y' ? el.scrollTop : el.scrollLeft;
      const mv = (ev) => { const dd = (ax === 'y' ? ev.clientY : ev.clientX) - start; if (ax === 'y') el.scrollTop = from + dd * k; else el.scrollLeft = from + dd * k; };
      const up = () => { t.removeEventListener('pointermove', mv); t.removeEventListener('pointerup', up); t.removeEventListener('pointercancel', up); s.drag = 0; t.classList.remove('drag'); };
      t.addEventListener('pointermove', mv); t.addEventListener('pointerup', up); t.addEventListener('pointercancel', up);
    };
    doc.addEventListener('scroll', (e) => {
      const el = e.target === doc ? root : e.target;
      if ((el === root || el instanceof Element) && (can(el, 'y') || can(el, 'x'))) show(get(el));
    }, true);
    // find everything that can scroll, now and whenever the page changes; each keeps its bar for as long as it can scroll
    const scan = () => {
      if (can(root, 'y') || can(root, 'x')) show(get(root));
      doc.querySelectorAll('body *').forEach((el) => { if (can(el, 'y') || can(el, 'x')) show(get(el)); });
      map.forEach(place);
    };
    let timer = 0; const later = () => { clearTimeout(timer); timer = setTimeout(scan, 120); };
    new MutationObserver((recs) => { if (recs.some((r) => !layer.contains(r.target))) later(); })
      .observe(doc.body, { childList: true, subtree: true, attributes: true, attributeFilter: ['class', 'style'] });
    window.addEventListener('resize', later);
    window.addEventListener('load', later);
    const tick = () => { map.forEach(place); requestAnimationFrame(tick); };
    scan(); if (doc.fonts) doc.fonts.ready.then(scan); requestAnimationFrame(tick);
  }
  if (document.body) installOverlayScrollbars(); else document.addEventListener('DOMContentLoaded', installOverlayScrollbars);

  window.Ready = {
    // utilities
    $, $$, esc, svg, I, hl, cboxHtml, pad, reducedMotion, swap,
    // dates
    NOW, DAYNAMES, MONTHS, MON, dayLabel, savedLabel,
    // photos
    PHOTOS, POS, photo, creditsHtml, statusHtml,
    // controls, changed in place
    setCbox, setToggle, setChoice, toggleHtml, searchHtml,
    // surfaces
    toast, openMenu, closeMenu, openPopover, placeFloating, menuOpen: () => !!MENU,
    openDialog, closeDialog, dialogOpen: () => !!DLG,
    // tabs and sliding segments
    tabsHtml, syncTabs, updateTabs,
    // actions
    on, actions: ACTS,
    installOverlayScrollbars,
  };
})();
