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
  // Copy via the async clipboard API, falling back to a hidden textarea
  // if the API is missing or refuses (permissions, odd embedders)
  var copyFallback = function (t) {
    return new Promise(function (resolve, reject) {
      var ta = document.createElement('textarea');
      ta.value = t;
      ta.style.position = 'fixed';
      ta.style.opacity = '0';
      document.body.appendChild(ta);
      ta.select();
      try { document.execCommand('copy') ? resolve() : reject(new Error('copy refused')); } finally { ta.remove(); }
    });
  };
  var copyText = function (t) {
    if (navigator.clipboard && navigator.clipboard.writeText) {
      return navigator.clipboard.writeText(t).catch(function () { return copyFallback(t); });
    }
    return copyFallback(t);
  };
  // All article pages get the dots: writing posts, CV, About, projects.
  // Raw-HTML pages lack kramdown ids, so headings are slugged on the fly
  // to make their section permalinks real.
  document.querySelectorAll('article.post > h1, article.post > h2').forEach(function (h, k) {
    if (!h.id) {
      var slug = h.textContent.trim().toLowerCase()
        .replace(/[^a-z0-9\s-]/g, '').replace(/\s+/g, '-');
      while (slug && document.getElementById(slug)) slug += '-2';
      if (slug) h.id = slug;
    }
    var i;
    do { i = Math.floor(Math.random() * BLOBS.length); } while (i === lastDot);
    lastDot = i;
    h._base = i;
    h._lch = BLOB_LCH[i];
    h._phase = k * 2.399; // golden-angle offsets: organic shimmer, no unison
    h.style.setProperty('--head-dot', BLOBS[i]);
    headDots.push(h);
    // The dot itself is a secret section-permalink button. It inherits
    // --head-dot from the heading, so the scroll animation drives it too.
    var url = location.origin + location.pathname + (h.id ? '#' + h.id : '');
    var tip = h.id ? 'Link to this section' : 'Link to this post';
    // Freeze the heading's accessible name before the button joins it,
    // so heading navigation doesn't announce the button label 16 times
    var name = h.textContent.trim();
    h.setAttribute('aria-label', name);
    var btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'head-dot';
    btn.setAttribute('aria-label', h.id ? 'Copy link to section: ' + name : 'Copy link to this post');
    btn.dataset.tip = tip;
    h.appendChild(btn);
    btn.addEventListener('click', function () {
      copyText(url).then(function () {
        btn.dataset.tip = 'Copied';
        btn.classList.add('copied');
        if (srLive) srLive.textContent = 'Link copied';
        clearTimeout(btn._tipT);
        btn._tipT = setTimeout(function () {
          btn.classList.remove('copied');
          if (srLive) srLive.textContent = '';
          // restore the label once the tooltip has faded out
          btn._tipT = setTimeout(function () { btn.dataset.tip = tip; }, 250);
        }, 1300);
      }).catch(function () {});
    });
  });
  // 1a1) Tear-off separators: a subtle dashed line before each section
  //      heading — but strictly between big blocks of text. Never before
  //      the Contents block or the first real section, never after a
  //      heading, never trailing the article.
  var tearFirstSkipped = false;
  var tearLines = [];
  headDots.forEach(function (h) {
    if (h.id === 'contents') return;
    if (!tearFirstSkipped) { tearFirstSkipped = true; return; }
    var prev = h.previousElementSibling;
    if (!prev || /^H[1-6]$/.test(prev.tagName)) return;
    var tear = document.createElement('div');
    tear.className = 'tear-line';
    tear.setAttribute('aria-hidden', 'true');
    h.parentNode.insertBefore(tear, h);
    tearLines.push(tear);
  });
  if (tearLines.length) {
    // Equal perforation pitch: pick the dot period nearest 10px that
    // divides the full-bleed width exactly, and phase the pattern so dot
    // centres land on multiples of it — the end dots coincide with (and
    // hide under) the larger edge semicircles, so every visible gap
    // matches, edge to edge.
    var sizeTears = function () {
      // True visible width — 100vw would overshoot by the scrollbar and
      // push the edge semicircles off screen
      var W = document.documentElement.clientWidth;
      var artLeft = tearLines[0].parentNode.getBoundingClientRect().left;
      if (!W) return;
      var p = W / Math.max(4, Math.round(W / 10));
      tearLines.forEach(function (t) {
        t.style.width = W + 'px';
        t.style.marginLeft = (-artLeft) + 'px';
        t.style.backgroundSize = p + 'px 6px';
        t.style.backgroundPositionX = (p / 2) + 'px';
      });
    };
    sizeTears();
    var tearRaf = 0;
    window.addEventListener('resize', function () {
      if (!tearRaf) tearRaf = requestAnimationFrame(function () { tearRaf = 0; sizeTears(); });
    }, { passive: true });
  }

  // Polite live region so screen readers hear the copy succeed
  var srLive = null;
  if (headDots.length) {
    srLive = document.createElement('div');
    srLive.className = 'sr-live';
    srLive.setAttribute('role', 'status');
    document.body.appendChild(srLive);
  }
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

  // Shared toggle-pill label: text plus a drawn up/down arrow
  var setToggle = function (btn, label, dir) {
    btn.innerHTML = '<span></span><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">' +
      (dir === 'down' ? '<path d="M12 5v14"/><path d="M6 13l6 6 6-6"/>' : '<path d="M12 19V5"/><path d="M6 11l6-6 6 6"/>') + '</svg>';
    btn.querySelector('span').textContent = label;
  };

  // 1a2) Collapsible contents: tall TOC cards start cut off with a fade
  var toc = document.getElementById('markdown-toc');
  if (toc && toc.scrollHeight > 320) {
    toc.classList.add('toc-collapsible', 'toc-collapsed');
    var tocBtn = document.createElement('button');
    tocBtn.type = 'button';
    tocBtn.className = 'toc-toggle';
    setToggle(tocBtn, 'Show more', 'down');
    toc.appendChild(tocBtn);
    tocBtn.addEventListener('click', function () {
      var collapsed = toc.classList.toggle('toc-collapsed');
      if (collapsed) setToggle(tocBtn, 'Show more', 'down');
      else setToggle(tocBtn, 'Show less', 'up');
    });
  }

  // 1a3) Article end: asterism full stop, centred escape hatch, boxed footnotes
  var endArt = document.querySelector('article.writing-post');
  var endNav = document.querySelector('.post-nav');
  if (endArt && endNav) {
    var fns = endArt.querySelector('.footnotes');
    // Squiggle recipe: amp 6.5, pure sine, 2.5px, nacre 45%. The wave count
    // is responsive: the recipe's 22 waves were tuned on the 672px desktop
    // column (~30.5px per wave), so narrower screens fit fewer whole waves
    // at the same wavelength. Rounding keeps the count integer, which keeps
    // the centred cosine ending crest-to-crest symmetric.
    var SQ_WAVELEN = 672 / 22, SQ_AMP = 6.5, SQ_THICK = 2.5, SQ_NACRE = 0.45;
    var sqFreq = function (W) { return Math.max(4, Math.round(W / SQ_WAVELEN)); };
    var SQ_STOPS = [['0%','#F2D9C8'],['20%','#F4D8E3'],['40%','#E3DCF4'],['60%','#CFE7EF'],['78%','#D8ECD9'],['100%','#F3EBC9']];
    var SVG_NS = 'http://www.w3.org/2000/svg';
    var squiggles = [];
    var makeSquiggle = function (flip) {
      var svg = document.createElementNS(SVG_NS, 'svg');
      svg.setAttribute('class', 'article-end' + (flip === -1 ? ' below' : ''));
      svg.setAttribute('aria-hidden', 'true');
      svg._flip = flip;
      squiggles.push(svg);
      return svg;
    };
    var drawSquiggles = function () {
      var W = endArt.clientWidth;
      if (!W) return;
      var H = 2 * SQ_AMP + SQ_THICK + 4, mid = H / 2;
      squiggles.forEach(function (svg, si) {
        svg.setAttribute('viewBox', '0 0 ' + W + ' ' + H);
        svg.style.height = H + 'px';
        var d = '';
        for (var x = 0; x <= W; x += 1.5) {
          var y = mid + svg._flip * SQ_AMP * Math.cos(((x - W / 2) / W) * Math.PI * 2 * sqFreq(W));
          d += (x === 0 ? 'M' : 'L') + x.toFixed(1) + ' ' + y.toFixed(2);
        }
        var stops = SQ_STOPS.map(function (st) { return '<stop offset="' + st[0] + '" stop-color="' + st[1] + '"/>'; }).join('');
        svg.innerHTML = '<defs><linearGradient id="sqnac' + si + '" x1="0" y1="0" x2="1" y2="0">' + stops + '</linearGradient></defs>' +
          '<path d="' + d + '" fill="none" stroke="#D5CEBD" stroke-width="' + SQ_THICK + '" stroke-linecap="round"/>' +
          '<path d="' + d + '" fill="none" stroke="url(#sqnac' + si + ')" stroke-width="' + SQ_THICK + '" stroke-linecap="round" opacity="' + SQ_NACRE + '"/>';
      });
    };
    var sqAbove = makeSquiggle(1);
    if (fns) endArt.insertBefore(sqAbove, fns); else endArt.appendChild(sqAbove);
    endNav.classList.add('article-end-nav');
    if (fns) endArt.insertBefore(endNav, fns); else endArt.appendChild(endNav);
    // Bottom squiggle only when something follows it (the footnotes card);
    // with nothing beneath, the back button is the full stop, centred in
    // the space between the squiggle and the footer
    if (fns) endArt.insertBefore(makeSquiggle(-1), fns);
    else endNav.classList.add('article-end-solo');
    drawSquiggles();
    var sqRaf = 0;
    window.addEventListener('resize', function () {
      if (!sqRaf) sqRaf = requestAnimationFrame(function () { sqRaf = 0; drawSquiggles(); });
    }, { passive: true });
    var endA = endNav.querySelector('a');
    if (endA) {
      var endTxt = endA.textContent.replace(/^\u2190\s*/, '');
      endA.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M19 12H5"/><path d="M11 18l-6-6 6-6"/></svg><span></span>';
      endA.querySelector('span').textContent = endTxt;
    }
    // Back-to-top button, sharing the row with the back button (label random,
    // up arrow after the text to match the site's pill convention)
    var TOP_LABELS = ['Head up', 'Up top', 'Rewind', 'Again!'];
    var toTop = document.createElement('button');
    toTop.type = 'button';
    toTop.className = 'to-top';
    var topLabel = TOP_LABELS[Math.floor(Math.random() * TOP_LABELS.length)];
    toTop.setAttribute('aria-label', topLabel + ' \u2014 back to top');
    toTop.innerHTML = '<span></span><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M12 19V5"/><path d="M6 11l6-6 6 6"/></svg>';
    toTop.querySelector('span').textContent = topLabel;
    toTop.addEventListener('click', function () {
      window.scrollTo({ top: 0, behavior: 'smooth' });
    });
    endNav.appendChild(toTop);
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
        setToggle(fnBtn, 'Show all', 'down');
        fns.appendChild(fnBtn);
        fnBtn.addEventListener('click', function () {
          var c = fns.classList.toggle('fn-collapsed');
          if (c) setToggle(fnBtn, 'Show all', 'down');
          else setToggle(fnBtn, 'Show less', 'up');
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
  // 2/3) The pixel olive spins on click; on mobile it doubles as the menu
  //      toggle (the hamburger is retired — the olive IS the button).
  var oliveWrap = document.querySelector('.olive-wrap');
  var olive = document.querySelector('.logo-olive');
  var panel = document.querySelector('.mobile-panel');
  var mobileNav = window.matchMedia('(max-width: 470px)');
  var oliveBusy = false;
  var spinOlive = function () {
    if (oliveBusy || !olive) return;
    oliveBusy = true;
    olive.classList.add('spinning');
    setTimeout(function () { olive.classList.remove('spinning'); }, 1100);
    setTimeout(function () { oliveBusy = false; }, 2000);
  };
  var setOliveLabel = function () {
    if (oliveWrap) oliveWrap.setAttribute('aria-label', mobileNav.matches ? 'Menu' : 'Spin the olive');
  };
  setOliveLabel();
  if (oliveWrap) oliveWrap.addEventListener('click', function () {
    if (mobileNav.matches && panel) {
      var open = panel.classList.toggle('open');
      oliveWrap.setAttribute('aria-expanded', open ? 'true' : 'false');
    }
    spinOlive();
  });
  // returning to desktop width closes the panel and resets state
  mobileNav.addEventListener('change', function (e) {
    setOliveLabel();
    if (!e.matches && panel) {
      panel.classList.remove('open');
      if (oliveWrap) oliveWrap.setAttribute('aria-expanded', 'false');
    }
  });

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
