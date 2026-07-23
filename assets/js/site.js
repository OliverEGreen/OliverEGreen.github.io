// olliegreen.info redesign — assets/js/site.js
(function () {
  // 1) Header darkens once it overlaps content
  var header = document.querySelector('.site-header');
  if (header) {
    var onScroll = function () { header.classList.toggle('scrolled', window.scrollY > 24); };
    window.addEventListener('scroll', onScroll, { passive: true });
    onScroll();
  }

  // 2) Random Dracula colour on nav/logo rollover (changes every time)
  var COLS = ['#FF79C6', '#BD93F9', '#50FA7B', '#FF5555', '#8BE9FD', '#FFB86C'];
  document.querySelectorAll('.site-nav a, .nav-menu a, .site-logo').forEach(function (el) {
    el.addEventListener('mouseenter', function () {
      el.style.setProperty('color', COLS[Math.floor(Math.random() * COLS.length)], 'important');
    });
    el.addEventListener('mouseleave', function () { el.style.removeProperty('color'); });
  });

  // 3) Mobile menu
  var btn = document.querySelector('.menu-btn');
  var panel = document.querySelector('.mobile-panel');
  if (btn && panel) btn.addEventListener('click', function () { panel.classList.toggle('open'); });

  // 4) Writing filter toggle (All / Technical / Non-technical)
  var toggle = document.querySelector('.filter-toggle');
  if (toggle) {
    toggle.querySelectorAll('button').forEach(function (b) {
      b.addEventListener('click', function () {
        toggle.querySelectorAll('button').forEach(function (x) { x.classList.remove('active'); });
        b.classList.add('active');
        var f = b.getAttribute('data-filter');
        document.querySelectorAll('.post-list li').forEach(function (li) {
          li.classList.toggle('hidden', f !== 'all' && li.getAttribute('data-kind') !== f);
        });
      });
    });
  }

  // 5) Subscribe form: random CEO placeholder + mock submit.
  //    To make it real, point the form at Buttondown/Mailchimp etc. (see README).
  var PHS = ['sjobs@apple.com', 'billg@microsoft.com', 'jeff@amazon.com', 'zuck@fb.com', 'sundar@google.com', 'jack@twitter.com', 'elon@x.com', 'sam@openai.com', 'satyan@microsoft.com', 'patrick@stripe.com'];
  document.querySelectorAll('.sub-form').forEach(function (form) {
    var input = form.querySelector('input[type="email"]');
    var err = form.parentElement.querySelector('.sub-error');
    if (input) input.placeholder = PHS[Math.floor(Math.random() * PHS.length)];
    form.addEventListener('submit', function (e) {
      e.preventDefault();
      var ok = /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(input.value);
      if (!ok) { if (err) err.hidden = false; return; }
      if (err) err.hidden = true;
      form.outerHTML = '<div class="sub-done">Bold move. You\u2019re in.</div>';
      // TODO: actually submit somewhere.
    });
    if (input && err) input.addEventListener('input', function () { err.hidden = true; });
  });
})();
