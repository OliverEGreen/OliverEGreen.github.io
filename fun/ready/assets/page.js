// Ready UI Design System page: the Desktop / Mobile switch. Its light key slides to the chosen view; the choice is remembered.
(() => {
  const seg = document.getElementById('rd-view'), shots = document.getElementById('rd-shots');
  if (!seg || !shots) return;
  const btns = [...seg.querySelectorAll('.sg')], ind = seg.querySelector('.ind');
  let placed = false;
  function place() {
    const on = btns.find((b) => b.classList.contains('on')); if (!on || !on.offsetWidth) return;
    if (!placed) ind.style.transition = 'none';
    ind.style.transform = 'translateX(' + on.offsetLeft + 'px)'; ind.style.width = on.offsetWidth + 'px'; ind.style.opacity = '1';
    if (!placed) { placed = true; requestAnimationFrame(() => requestAnimationFrame(() => { ind.style.transition = ''; })); }
  }
  function show(view, keep) {
    btns.forEach((b) => { const on = b.dataset.view === view; b.classList.toggle('on', on); b.setAttribute('aria-checked', on); b.tabIndex = on ? 0 : -1; });
    shots.classList.toggle('mobile', view === 'mobile'); shots.classList.toggle('desktop', view !== 'mobile');
    place();
    if (keep) { try { localStorage.setItem('ready-view', view); } catch (e) { /* private window */ } }
  }
  seg.addEventListener('click', (e) => { const b = e.target.closest('.sg'); if (b) show(b.dataset.view, true); });
  seg.addEventListener('keydown', (e) => {
    if (!/^Arrow(Left|Right|Up|Down)$/.test(e.key)) return;
    e.preventDefault();
    const next = btns[(btns.findIndex((b) => b.classList.contains('on')) + 1) % btns.length];
    show(next.dataset.view, true); next.focus();
  });
  let saved = null; try { saved = localStorage.getItem('ready-view'); } catch (e) { /* private window */ }
  show(saved === 'mobile' ? 'mobile' : 'desktop', false);
  new ResizeObserver(place).observe(seg);
  if (document.fonts) document.fonts.ready.then(place);
})();

// Ready UI Design System page: each component runs live in a frame, which grows to the component's height
(() => {
  const frames = [...document.querySelectorAll('.rd-live iframe')];
  if (!frames.length) return;
  addEventListener('message', (e) => {
    if (e.origin !== location.origin || !e.data || typeof e.data.readyFrameHeight !== 'number') return;
    const f = frames.find((x) => x.contentWindow === e.source);
    if (f) f.style.height = Math.max(120, e.data.readyFrameHeight) + 'px';
  });
})();
