// Ready keyboard: builds the keys into a case and plays them.
// ReadyKeyboard.mount(wrap, { scope }) — with a scope element, your own keys only play it after a click or focus inside that element;
// without one they always do. Click or tap a key to press it, or slide across the keys.
(() => {
  // [code, legend, shifted legend, width in units, kind]
  const ROWS = [
    [['Backquote','`','~'],['Digit1','1','!'],['Digit2','2','@'],['Digit3','3','#'],['Digit4','4','$'],['Digit5','5','%'],['Digit6','6','^'],['Digit7','7','&'],['Digit8','8','*'],['Digit9','9','('],['Digit0','0',')'],['Minus','-','_'],['Equal','=','+'],['Backspace','DELETE',null,2,'mod']],
    [['Tab','TAB',null,1.5,'mod'],['KeyQ','Q'],['KeyW','W'],['KeyE','E'],['KeyR','R'],['KeyT','T'],['KeyY','Y'],['KeyU','U'],['KeyI','I'],['KeyO','O'],['KeyP','P'],['BracketLeft','[','{'],['BracketRight',']','}'],['Backslash','\\','|',1.5]],
    [['CapsLock','CAPS|LOCK',null,1.75,'mod'],['KeyA','A'],['KeyS','S'],['KeyD','D'],['KeyF','F'],['KeyG','G'],['KeyH','H'],['KeyJ','J'],['KeyK','K'],['KeyL','L'],['Semicolon',';',':'],['Quote',"'",'"'],['Enter','RETURN',null,2.25,'go']],
    [['ShiftLeft','SHIFT',null,2.25,'mod'],['KeyZ','Z'],['KeyX','X'],['KeyC','C'],['KeyV','V'],['KeyB','B'],['KeyN','N'],['KeyM','M'],['Comma',',','<'],['Period','.','>'],['Slash','/','?'],['ShiftRight','SHIFT',null,2.75,'mod']],
    [['ControlLeft','CTRL',null,1.5,'mod'],['AltLeft','ALT',null,1.25,'mod'],['MetaLeft','CMD',null,1.5,'mod'],['Space','',null,'space'],['MetaRight','CMD',null,1.5,'mod'],['AltRight','ALT',null,1.25,'mod'],['ControlRight','CTRL',null,1.5,'mod']]
  ];
  const esc = (s) => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  const NAMES = { Backspace: 'Delete', Tab: 'Tab', CapsLock: 'Caps lock', Enter: 'Return', ShiftLeft: 'Shift', ShiftRight: 'Shift', ControlLeft: 'Control', ControlRight: 'Control',
    AltLeft: 'Alt', AltRight: 'Alt', MetaLeft: 'Command', MetaRight: 'Command', Space: 'Space' };

  function mount(wrap, opts = {}) {
    const doc = document, scope = opts.scope || null, MAX = opts.max || 1.15;
    const kcase = doc.createElement('div'), plate = doc.createElement('div'), kb = doc.createElement('div');
    kcase.className = 'kcase'; plate.className = 'kplate'; kb.className = 'kb';
    kb.tabIndex = 0; kb.setAttribute('role', 'group'); kb.setAttribute('aria-label', 'Keyboard. Click the keys, or type, to press them');
    plate.appendChild(kb); kcase.appendChild(plate); wrap.appendChild(kcase); wrap.classList.add('unfit');
    const keys = {};

    // build: the space bar takes whatever width its row leaves
    ROWS.forEach((row) => {
      const r = doc.createElement('div'); r.className = 'krow';
      const fixed = row.reduce((s, k) => s + (k[3] === 'space' ? 0 : (k[3] || 1)), 0);
      row.forEach(([code, label, shifted, w, kind]) => {
        const b = doc.createElement('button'); b.type = 'button'; b.tabIndex = -1; b.className = 'k'; b.dataset.code = code;
        let width = w || 1;
        if (w === 'space') { width = 15 - fixed; b.classList.add('space'); kind = null; }
        if (kind) b.classList.add(kind);
        b.style.setProperty('--kw', width);
        let face = shifted ? '<span class="two"><span class="up">' + esc(shifted) + '</span><span class="dn">' + esc(label) + '</span></span>'
          : label.includes('|') ? '<span class="lg stack">' + label.split('|').map((w) => '<span>' + esc(w) + '</span>').join('') + '</span>'
          : (label ? '<span class="lg">' + esc(label) + '</span>' : '');
        if (code === 'CapsLock' || code.startsWith('Shift')) face = '<i class="dot" aria-hidden="true"></i>' + face;
        b.innerHTML = '<span class="kbody"><span class="kt">' + face + '</span></span>';
        b.setAttribute('aria-label', NAMES[code] || (shifted ? label + ' or ' + shifted : label));
        keys[code] = b;
        r.appendChild(b);
      });
      kb.appendChild(r);
    });

    // scale the case to fit, leaving room for its shadow
    const SIDE = 22, BELOW = 64, ABOVE = 4;
    function fit() {
      const W = wrap.clientWidth, cw = kcase.offsetWidth, ch = kcase.offsetHeight; if (!W || !cw) return;
      const s = Math.min(MAX, W / (cw + SIDE * 2));
      kcase.style.transform = 'translate(' + ((W - cw * s) / 2).toFixed(2) + 'px,' + ABOVE + 'px) scale(' + s.toFixed(4) + ')';
      wrap.style.height = Math.ceil(ABOVE + (ch + BELOW) * s) + 'px';
      wrap.classList.remove('unfit');
    }
    new ResizeObserver(fit).observe(wrap); fit();
    if (doc.fonts) doc.fonts.ready.then(fit);

    // state: caps lock, a latched shift (clicked) and shifts held on your own keyboard
    let caps = false, latched = false, active = !scope; const held = new Set();
    const isMod = (code) => /^(Shift|Alt|Meta|Control|CapsLock)/.test(code);
    function refresh() {
      const sh = latched || held.has('ShiftLeft') || held.has('ShiftRight');
      kb.classList.toggle('shifted', sh);
      keys.CapsLock.classList.toggle('lit', caps);
      ['ShiftLeft', 'ShiftRight'].forEach((c) => { keys[c].classList.toggle('lit', sh); keys[c].classList.toggle('down', latched || held.has(c)); });
    }
    const unlatch = (code) => { if (latched && !isMod(code)) { latched = false; refresh(); } };
    function releaseAll() { held.clear(); Object.values(keys).forEach((k) => k.classList.remove('down')); refresh(); }

    // pointer: press on down, let go on up; sliding across the keys presses each in turn
    function down(code) {
      if (code.startsWith('Shift')) { latched = !latched; refresh(); return; }
      if (code === 'CapsLock') { caps = !caps; refresh(); }
      keys[code].classList.add('down');
    }
    function up(code) {
      if (code.startsWith('Shift')) return;
      keys[code].classList.remove('down'); unlatch(code);
    }
    let ptr = null, cur = null;
    kb.addEventListener('pointerdown', (e) => {
      const b = e.target.closest('.k'); if (!b || ptr !== null) return;
      e.preventDefault();
      ptr = e.pointerId; cur = b.dataset.code; down(cur);
    });
    doc.addEventListener('pointermove', (e) => {
      if (e.pointerId !== ptr) return;
      const over = doc.elementFromPoint(e.clientX, e.clientY), b = over && over.closest ? over.closest('.k') : null;
      let code = b && kb.contains(b) ? b.dataset.code : null;
      if (code && /^(Shift|CapsLock)/.test(code)) code = null;   // sliding never toggles shift or caps lock
      if (code === cur) return;
      if (cur) up(cur); cur = code; if (cur) down(cur);
    });
    const end = (e) => { if (e.pointerId !== ptr) return; if (cur) up(cur); ptr = null; cur = null; };
    doc.addEventListener('pointerup', end); doc.addEventListener('pointercancel', end);

    // your own keyboard presses the same keys; shortcuts with command or control are left alone
    if (scope) {
      const set = (on) => { if (active === on) return; active = on; if (!on) releaseAll(); };
      doc.addEventListener('pointerdown', (e) => set(scope.contains(e.target)), true);
      doc.addEventListener('focusin', (e) => set(scope.contains(e.target)));
    }
    doc.addEventListener('keydown', (e) => {
      if (!active) return;
      const k = keys[e.code]; if (!k) return;
      if ((e.metaKey || e.ctrlKey) && !/^(Meta|Control)/.test(e.code)) return;
      if (e.target !== kb && e.target.closest && e.target.closest('input, textarea, select, [contenteditable]')) return;
      if (e.code === 'Tab') { k.classList.add('down'); return; }   // focus still moves on
      e.preventDefault();
      if (e.code.startsWith('Shift')) { held.add(e.code); refresh(); return; }
      if (e.code === 'CapsLock') { caps = e.getModifierState ? e.getModifierState('CapsLock') : !caps; refresh(); }
      k.classList.add('down');
      if (!e.repeat) unlatch(e.code);
    });
    doc.addEventListener('keyup', (e) => {
      const k = keys[e.code]; if (!k) return;
      if (e.code.startsWith('Shift')) { held.delete(e.code); refresh(); return; }
      if (e.code === 'CapsLock' && e.getModifierState) { caps = e.getModifierState('CapsLock'); refresh(); }
      k.classList.remove('down');
      // a Mac sends no key-ups for keys let go while command is held
      if (/^(Meta|Control)/.test(e.code)) Object.entries(keys).forEach(([c, el]) => { if (!isMod(c)) el.classList.remove('down'); });
    });
    window.addEventListener('blur', releaseAll);
    refresh();
    return { el: kcase, fit };
  }
  window.ReadyKeyboard = { mount };
})();
