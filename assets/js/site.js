// olliegreen.info redesign — assets/js/site.js
(function () {
  // 1) Header darkens once it overlaps content
  var header = document.querySelector('.site-header');
  if (header) {
    var onScroll = function () { header.classList.toggle('scrolled', window.scrollY > 24); };
    window.addEventListener('scroll', onScroll, { passive: true });
    onScroll();
  }

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
  var backLink = document.querySelector('.post-nav a');
  if (backLink) {
    var EXITS = [
      'Quick! Let’s escape',
      'Take me back',
      'Who let you in here? Get out!',
      'You can leave, now',
      'Exit through the gift shop'
    ];
    backLink.textContent = '← ' + EXITS[Math.floor(Math.random() * EXITS.length)];
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
})();
