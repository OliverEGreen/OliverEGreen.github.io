// olliegreen.info redesign — assets/js/site.js
(function () {
  // Shared warm-neutral palette: creams, beiges, olives and sands.
  // solid = flat fill (footer, blockquotes), line = darker companion (borders).
  var PALETTE = [
    { solid: '#E6E8D6', line: '#C6CCA9' },
    { solid: '#E9E4DB', line: '#D5CEBD' },
    { solid: '#EDE6D0', line: '#D8CCA8' },
    { solid: '#E4E8C9', line: '#C4CC97' },
    { solid: '#EAE0CC', line: '#D2C3A3' },
    { solid: '#E8E9DE', line: '#CBCFB8' },
    { solid: '#F0E8D8', line: '#DCCCA9' },
    { solid: '#E2E4D0', line: '#C2C7A5' },
    { solid: '#EDE3D3', line: '#D6C3AB' },
    { solid: '#E7E7CF', line: '#CBCBA0' }
  ];
  var alphaOf = function (hex, a) {
    var n = parseInt(hex.slice(1), 16);
    return 'rgba(' + (n >> 16) + ', ' + ((n >> 8) & 255) + ', ' + (n & 255) + ', ' + a + ')';
  };
  // Dracula pops, shared by the gutter blobs and paragraph dots
  var BLOBS = ['#8be9fd', '#50fa7b', '#ffb86c', '#ff79c6', '#bd93f9', '#ff5555', '#f1fa8c'];

  // Consecutive picks are always different, shared across header and quotes
  var lastTint = -1;
  var pickTint = function () {
    var i;
    do { i = Math.floor(Math.random() * PALETTE.length); } while (i === lastTint);
    lastTint = i;
    return PALETTE[i];
  };

  // 1) Header darkens once it overlaps content
  var header = document.querySelector('.site-header');
  if (header) {
    // Each time the header enters its scrolled state, pick a fresh tint
    // from the palette; the footer follows the same pick.
    var themeMeta = document.querySelector('meta[name="theme-color"]');
    var currentSolid = '#FAF9F7';
    var lastChrome = '';
    var applyTint = function () {
      var pick = pickTint();
      var rootStyle = document.documentElement.style;
      rootStyle.setProperty('--scroll-tint', alphaOf(pick.solid, 0.96));
      rootStyle.setProperty('--scroll-line', alphaOf(pick.line, 0.5));
      rootStyle.setProperty('--footer-live', pick.solid);
      currentSolid = pick.solid;
    };
    applyTint();
    var wasScrolled = false;
    var onScroll = function () {
      var sc = window.scrollY > 24;
      if (sc && !wasScrolled) applyTint();
      wasScrolled = sc;
      header.classList.toggle('scrolled', sc);
      // iOS 26 Safari ignores theme-color and samples body's background-color
      // instead; older browsers still honour the meta. Feed both, once per
      // state change.
      var chromeColor = sc ? currentSolid : '#FAF9F7';
      if (chromeColor !== lastChrome) {
        lastChrome = chromeColor;
        document.body.style.backgroundColor = chromeColor;
        if (themeMeta) themeMeta.setAttribute('content', chromeColor);
      }
    };
    window.addEventListener('scroll', onScroll, { passive: true });
    onScroll();
  }

  // 1a) Writing pages: a random Dracula dot follows each section heading.
  //     On scroll, each dot's lightness/chroma breathes on a sine wave
  //     (hue never changes, so the scheme holds). Per-dot phase offsets
  //     make the page shimmer rather than pulse in unison.
  var BLOB_LCH = [
    [0.883, 0.093, 212.8], // #8be9fd
    [0.871, 0.220, 148.0], // #50fa7b
    [0.834, 0.124, 66.6],  // #ffb86c
    [0.755, 0.183, 346.8], // #ff79c6
    [0.742, 0.149, 301.9], // #bd93f9
    [0.682, 0.206, 24.4],  // #ff5555
    [0.955, 0.134, 112.8]  // #f1fa8c
  ];
  var lastDot = -1;
  var headDots = [];
  document.querySelectorAll('article.writing-post > h1, article.writing-post > h2').forEach(function (h, k) {
    var i;
    do { i = Math.floor(Math.random() * BLOBS.length); } while (i === lastDot);
    lastDot = i;
    h._base = i;
    h._lch = BLOB_LCH[i];
    h._phase = k * 2.399; // golden-angle offsets: organic shimmer, no unison
    h.style.setProperty('--head-dot', BLOBS[i]);
    headDots.push(h);
  });
  if (headDots.length) {
    // 'breathe': sine-wave lightness/chroma shimmer. 'step': dots walk the
    // palette in lockstep, one step per DOT_PERIOD px, cross-fading between.
    var DOT_MODE = 'step';
    var DOT_WAVELEN = 1400; // px of scroll per full sine cycle (breathe)
    var DOT_PERIOD = 260;   // px of scroll per palette step (step)
    var dotRaf = 0, lastStepPhase = 0;
    var updateDots = function () {
      dotRaf = 0;
      if (DOT_MODE === 'step') {
        var phase = Math.floor(window.scrollY / DOT_PERIOD);
        if (phase === lastStepPhase) return;
        lastStepPhase = phase;
        headDots.forEach(function (h) {
          h.style.setProperty('--head-dot', BLOBS[(h._base + phase) % BLOBS.length]);
        });
      } else {
        var theta = (window.scrollY / DOT_WAVELEN) * Math.PI * 2;
        headDots.forEach(function (h) {
          var s = Math.sin(theta + h._phase);
          var L = Math.min(0.97, h._lch[0] * (1 + 0.10 * s));
          var C = h._lch[1] * (1 + 0.20 * s);
          h.style.setProperty('--head-dot', 'oklch(' + L.toFixed(4) + ' ' + C.toFixed(4) + ' ' + h._lch[2] + ')');
        });
      }
    };
    document.documentElement.style.setProperty('--dot-fade', DOT_MODE === 'step' ? '600ms' : '120ms');
    window.addEventListener('scroll', function () {
      if (!dotRaf) dotRaf = requestAnimationFrame(updateDots);
    }, { passive: true });
    updateDots();
  }

  // 1a2) Collapsible contents: tall TOC cards start cut off with a fade
  var toc = document.getElementById('markdown-toc');
  if (toc && toc.scrollHeight > 320) {
    toc.classList.add('toc-collapsible', 'toc-collapsed');
    var tocBtn = document.createElement('button');
    tocBtn.type = 'button';
    tocBtn.className = 'toc-toggle';
    tocBtn.textContent = 'Show more';
    toc.appendChild(tocBtn);
    tocBtn.addEventListener('click', function () {
      var collapsed = toc.classList.toggle('toc-collapsed');
      tocBtn.textContent = collapsed ? 'Show more' : 'Show less';
    });
  }

  // 1a3) Article end: asterism full stop, centred escape hatch, boxed footnotes
  var endArt = document.querySelector('article.writing-post');
  var endNav = document.querySelector('.post-nav');
  if (endArt && endNav) {
    var fns = endArt.querySelector('.footnotes');
    var endMark = document.createElement('div');
    endMark.className = 'article-end';
    var endCols = [];
    for (var e = 0; e < 3; e++) {
      var ci;
      do { ci = Math.floor(Math.random() * BLOBS.length); } while (endCols.indexOf(ci) !== -1);
      endCols.push(ci);
      var endDot = document.createElement('i');
      endDot.style.background = BLOBS[ci];
      endMark.appendChild(endDot);
    }
    if (fns) endArt.insertBefore(endMark, fns); else endArt.appendChild(endMark);
    endNav.classList.add('article-end-nav');
    if (fns) endArt.insertBefore(endNav, fns); else endArt.appendChild(endNav);
    var endA = endNav.querySelector('a');
    if (endA) {
      var endTxt = endA.textContent.replace(/^\u2190\s*/, '');
      endA.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M19 12H5"/><path d="M11 18l-6-6 6-6"/></svg><span></span>';
      endA.querySelector('span').textContent = endTxt;
    }
    if (fns) {
      var fnTitle = document.createElement('p');
      fnTitle.className = 'fn-title';
      fnTitle.textContent = 'Footnotes';
      endArt.insertBefore(fnTitle, fns);
      if (fns.scrollHeight > 220) {
        fns.classList.add('fn-collapsible', 'fn-collapsed');
        var fnBtn = document.createElement('button');
        fnBtn.type = 'button';
        fnBtn.className = 'toc-toggle';
        fnBtn.textContent = 'Show all';
        fns.appendChild(fnBtn);
        fnBtn.addEventListener('click', function () {
          var c = fns.classList.toggle('fn-collapsed');
          fnBtn.textContent = c ? 'Show all' : 'Show less';
        });
        document.querySelectorAll('sup a.footnote').forEach(function (m) {
          m.addEventListener('click', function () {
            if (fns.classList.contains('fn-collapsed')) fnBtn.click();
          });
        });
      }
    }
  }

  // 1b) Blockquotes: each quote takes its own random tint from the palette
  document.querySelectorAll('article.post blockquote').forEach(function (bq) {
    var pick = pickTint();
    bq.style.background = pick.solid;
    bq.style.borderLeftColor = pick.line;
  });

  // 2) Random Dracula colour on nav/logo rollover (changes every time).
  //    Hover-capable devices only — taps on touch screens fake mouseenter
  //    and would leave the colours (and dark header) stuck on.
  var canHover = window.matchMedia('(hover: hover) and (pointer: fine)').matches;
  var COLS = ['#FF79C6', '#BD93F9', '#50FA7B', '#FF5555', '#8BE9FD', '#FFB86C'];
  // Remember the last pick so consecutive colours are always different
  var lastCol = null;
  var pickCol = function () {
    var pool = COLS.filter(function (c) { return c !== lastCol; });
    lastCol = pool[Math.floor(Math.random() * pool.length)];
    return lastCol;
  };
  if (canHover) {
    document.querySelectorAll('.site-nav a, .nav-menu a').forEach(function (el) {
      el.addEventListener('mouseenter', function () {
        el.style.setProperty('color', pickCol(), 'important');
      });
      el.addEventListener('mouseleave', function () { el.style.removeProperty('color'); });
    });
  }

  // 2b) OG logo: new random dark-mode colour each time the header goes dark
  if (header && canHover) {
    header.addEventListener('mouseenter', function () {
      header.style.setProperty('--og-dark', pickCol());
    });
    header.addEventListener('mouseleave', function () {
      header.style.removeProperty('--og-dark');
    });
  }

  // 2c) Olive spins once on click, 2s cooldown
  var olive = document.querySelector('.logo-olive');
  var oliveBusy = false;
  if (olive) olive.addEventListener('click', function () {
    if (oliveBusy) return;
    oliveBusy = true;
    olive.classList.add('spinning');
    setTimeout(function () { olive.classList.remove('spinning'); }, 1100);
    setTimeout(function () { oliveBusy = false; }, 2000);
  });

  // 3) Mobile menu
  var btn = document.querySelector('.menu-btn');
  var panel = document.querySelector('.mobile-panel');
  if (btn && panel) btn.addEventListener('click', function () { panel.classList.toggle('open'); });

  // 4) Writing filter toggle (All / Technical / Non-technical).
  //    A data-limit on the list (homepage) caps it at the N most recent matches.
  //    #technical / #non-technical in the URL applies the filter on landing
  //    and whenever the hash changes (e.g. via the topnav dropdown).
  var toggle = document.querySelector('.filter-toggle');
  var list = document.querySelector('.post-list');
  if (toggle && list) {
    var limit = parseInt(list.getAttribute('data-limit'), 10) || Infinity;
    var applyFilter = function (f) {
      toggle.querySelectorAll('button').forEach(function (x) {
        x.classList.toggle('active', x.getAttribute('data-filter') === f);
      });
      var shown = 0;
      list.querySelectorAll('li').forEach(function (li) {
        var match = f === 'all' || li.getAttribute('data-kind') === f;
        var show = match && shown < limit;
        if (show) shown++;
        li.classList.toggle('hidden', !show);
      });
    };
    toggle.querySelectorAll('button').forEach(function (b) {
      b.addEventListener('click', function () { applyFilter(b.getAttribute('data-filter')); });
    });
    var applyHash = function () {
      var h = window.location.hash.replace('#', '');
      if (h === 'technical' || h === 'non-technical') applyFilter(h);
    };
    window.addEventListener('hashchange', applyHash);
    applyHash();
  }

  // 5) Post-nav back link: randomise the escape text per page load
  var backLink = document.querySelector('.post-nav a.escape-hatch');
  if (backLink) {
    var EXITS = [
      'Quick! Let’s escape',
      'Take me back',
      'Who let you in here? Get out!',
      'You can leave, now',
      'Exit through the gift shop'
    ];
    var exitPhrase = EXITS[Math.floor(Math.random() * EXITS.length)];
    var exitSpan = backLink.querySelector('span');
    if (exitSpan) exitSpan.textContent = exitPhrase;
    else backLink.textContent = '← ' + exitPhrase;
  }

  // 6) Contact form: validates, then POSTs to the form backend so the
  //    destination address never appears anywhere client-side.
  var cform = document.querySelector('.contact-form');
  if (cform) {
    var endpoint = cform.getAttribute('data-endpoint');
    var pending = document.getElementById('contact-pending');
    if (endpoint) {
      cform.hidden = false;
      if (pending) pending.hidden = true;
      var cerr = cform.querySelector('.contact-error');
      cform.addEventListener('submit', function (e) {
        e.preventDefault();
        var email = cform.querySelector('[name="email"]').value;
        var subject = cform.querySelector('[name="_subject"]').value;
        var message = cform.querySelector('[name="message"]').value;
        var problem = null;
        if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email)) problem = 'That’s not an email and you know it.';
        else if (subject.trim().length < 3) problem = 'Give it a subject.';
        else if (message.trim().length < 10) problem = 'The message is rather the point.';
        if (problem) { cerr.textContent = problem; cerr.hidden = false; return; }
        cerr.hidden = true;
        var btn = cform.querySelector('button[type="submit"]');
        btn.disabled = true; btn.textContent = 'Sending…';
        fetch(endpoint, { method: 'POST', headers: { 'Accept': 'application/json' }, body: new FormData(cform) })
          .then(function (r) {
            if (!r.ok) throw new Error('send failed');
            cform.outerHTML = '<div class="sub-done">Message sent. I read everything, eventually.</div>';
          })
          .catch(function () {
            btn.disabled = false; btn.innerHTML = 'Send it <span>&rarr;</span>';
            cerr.textContent = 'Something broke. Try again in a minute.'; cerr.hidden = false;
          });
      });
    }
  }

  // 7) Subscribe form: random CEO placeholder + validation. Valid submissions
  //    POST to Buttondown via fetch so the visitor never leaves the site;
  //    Buttondown then sends its confirmation email. (No-JS fallback: the
  //    native form POST still works, landing on Buttondown's page.)
  var PHS = ['sjobs@apple.com', 'billg@microsoft.com', 'jeff@amazon.com', 'zuck@fb.com', 'sundar@google.com', 'jack@twitter.com', 'elon@x.com', 'sam@openai.com', 'satyan@microsoft.com', 'patrick@stripe.com'];
  document.querySelectorAll('.sub-form').forEach(function (form) {
    var input = form.querySelector('input[type="email"]');
    var err = form.parentElement.querySelector('.sub-error');
    if (input) input.placeholder = PHS[Math.floor(Math.random() * PHS.length)];
    form.addEventListener('submit', function (e) {
      var ok = /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(input.value);
      if (!ok) {
        e.preventDefault();
        if (err) { err.textContent = 'That’s not an email and you know it.'; err.hidden = false; }
        return;
      }
      if (err) err.hidden = true;
      e.preventDefault();
      var btn = form.querySelector('button');
      if (btn) btn.disabled = true;
      fetch(form.action, { method: 'POST', mode: 'no-cors', body: new FormData(form) })
        .then(function () {
          form.outerHTML = '<div class="sub-done">Bold move. Check your inbox to confirm.</div>';
        })
        .catch(function () {
          if (btn) btn.disabled = false;
          if (err) { err.textContent = 'Something broke. Try again in a minute.'; err.hidden = false; }
        });
    });
    if (input && err) input.addEventListener('input', function () { err.hidden = true; });
  });

  // 8) Lightbox: click a project image to see it whole, scaled to fit.
  //    Native <dialog> supplies Esc-to-close and the backdrop for free.
  var lbTargets = document.querySelectorAll('article.post.project-page figure img, article.post .project figure img');
  if (lbTargets.length && window.HTMLDialogElement) {
    var dlg = document.createElement('dialog');
    dlg.className = 'lightbox';
    dlg.innerHTML = '<figure><img alt=""><figcaption class="lightbox-caption"></figcaption></figure><button class="lightbox-close" aria-label="Close" type="button">\u00d7</button>';
    document.body.appendChild(dlg);
    var dlgImg = dlg.querySelector('img');
    dlg.querySelector('.lightbox-close').addEventListener('click', function () { dlg.close(); });
    dlg.addEventListener('click', function (e) { if (e.target === dlg) dlg.close(); });
    lbTargets.forEach(function (img) {
      img.style.cursor = 'zoom-in';
      img.addEventListener('click', function () {
        dlgImg.src = img.currentSrc || img.src;
        dlgImg.alt = img.alt || '';
        var cap = img.closest('figure') ? img.closest('figure').querySelector('figcaption') : null;
        dlg.querySelector('.lightbox-caption').textContent = cap ? cap.textContent : '';
        dlg.showModal();
      });
    });
  }

  // 9) Project cards: seeded random corners (concentric frame) + gutter blobs.
  //    Each card's shape is stable per load; hovering re-rolls only that card.
  var grids = Array.prototype.slice.call(document.querySelectorAll('.pgrid'));
  if (grids.length) {
    var CHAOS = 1, FRAME = 8, MINR = 3, MAXR = 3 + 45 * CHAOS;
    var xorshift = function (seed) {
      var s = seed >>> 0 || 1;
      return function () { s ^= s << 13; s ^= s >>> 17; s ^= s << 5; s >>>= 0; return s / 4294967296; };
    };
    var rollCorners = function (rand) {
      var c = [0, 0, 0, 0].map(function () { return rand() < 0.3 ? 0 : Math.round(MINR + rand() * (MAXR - MINR)); });
      if (c.every(function (v) { return v === 0; })) c[Math.floor(rand() * 4)] = Math.round(MAXR * 0.6);
      return c;
    };
    var applyShape = function (card) {
      var c = rollCorners(xorshift(card._seed));
      card.style.borderRadius = c.map(function (v) { return v + 'px'; }).join(' ');
      var inner = card.querySelector('.pcard-inner');
      if (inner) inner.style.borderRadius = c.map(function (v) { return (v === 0 ? 0 : Math.max(2, v - FRAME)) + 'px'; }).join(' ');
    };
    var cardIndex = 0;
    grids.forEach(function (g) {
      g.querySelectorAll('.pcard').forEach(function (card) {
        card._seed = ((cardIndex + 1) * 2654435761) >>> 0;
        cardIndex++;
        applyShape(card);
        if (canHover) card.addEventListener('mouseenter', function () {
          card._seed = (Math.imul(card._seed, 1103515245) + 12345) >>> 0;
          applyShape(card);
          // Cycle the colour of blobs sitting against this card
          var grid = card.closest('.pgrid');
          if (grid) {
            var idx = Array.prototype.indexOf.call(grid.querySelectorAll('.pcard'), card);
            grid.querySelectorAll('.blob').forEach(function (b) {
              var pair = (b.dataset.cards || '').split(',');
              if (pair.indexOf(String(idx)) !== -1) {
                var cur = parseInt(b.dataset.col, 10) || 0;
                var next = (cur + 1 + Math.floor(Math.random() * (BLOBS.length - 1))) % BLOBS.length;
                b.dataset.col = next;
                b.style.background = BLOBS[next];
              }
            });
          }
        });
      });
    });

    // Gutter blobs: one Dracula dot per gap, measured from the DOM.
    // Existing dots are moved rather than recreated so resize tracks smoothly
    // and hover-cycled colours survive a reflow.
    var layBlobs = function () {
      var brand = xorshift(482634);
      grids.forEach(function (g) {
        var gRect = g.getBoundingClientRect();
        var rects = Array.prototype.map.call(g.querySelectorAll('.pcard'), function (c) {
          var r = c.getBoundingClientRect();
          return { l: r.left - gRect.left, t: r.top - gRect.top, r: r.right - gRect.left, b: r.bottom - gRect.top, w: r.width, h: r.height };
        });
        var spots = [];
        for (var i = 0; i < rects.length; i++) {
          for (var j = 0; j < rects.length; j++) {
            if (i === j) continue;
            var a = rects[i], b = rects[j];
            if (Math.abs(a.t - b.t) < 2 && b.l > a.r && b.l - a.r < 60) spots.push({ x: (a.r + b.l) / 2, y: a.t + a.h / 2, cards: [i, j] });
            if (Math.abs(a.l - b.l) < 2 && b.t > a.b && b.t - a.b < 60) spots.push({ x: a.l + a.w / 2, y: (a.b + b.t) / 2, cards: [i, j] });
          }
        }
        var prev = -1;
        var existing = g.querySelectorAll('.blob');
        spots.forEach(function (s, k) {
          var pick = Math.floor(brand() * BLOBS.length);
          if (pick === prev) pick = (pick + 1) % BLOBS.length;
          prev = pick;
          var d = existing[k];
          if (!d) {
            d = document.createElement('div');
            d.className = 'blob';
            d.style.background = BLOBS[pick];
            d.dataset.col = pick;
            g.appendChild(d);
          }
          d.style.left = (s.x - 4) + 'px';
          d.style.top = (s.y - 4) + 'px';
          d.dataset.cards = s.cards.join(',');
        });
        for (var k = spots.length; k < existing.length; k++) existing[k].remove();
      });
    };
    // Home grid: show however many cards fill exactly two rows at this width
    var homeGrid = document.querySelector('.section-projects .pgrid');
    var trimRows = function () {
      if (!homeGrid) return;
      var cards = homeGrid.querySelectorAll('.pcard');
      var cols = getComputedStyle(homeGrid).gridTemplateColumns.split(' ').length;
      var max = Math.max(cols * 2, 4);
      cards.forEach(function (c, i) { c.style.display = i < max ? '' : 'none'; });
    };
    trimRows();
    layBlobs();
    var blobRaf = 0;
    var queueBlobs = function () {
      if (blobRaf) return;
      blobRaf = requestAnimationFrame(function () { blobRaf = 0; trimRows(); layBlobs(); });
    };
    window.addEventListener('resize', queueBlobs);
    if (window.ResizeObserver) {
      var ro = new ResizeObserver(queueBlobs);
      grids.forEach(function (g) { ro.observe(g); });
    }
    if (document.fonts && document.fonts.ready) document.fonts.ready.then(layBlobs);
  }
})();
