// Ready UI components live inside the Ready UI Design System page, each in its own frame.
// Opened on their own, they go to that page. Inside it, the page title and back link are hidden,
// an optional starting state is set (?state=grouped-day), and the frame is told how tall the component is.
(() => {
  if (window.top === window.self) { location.replace('/fun/ready/design-system/'); return; }
  document.documentElement.classList.add('embed');
  const state = new URLSearchParams(location.search).get('state');
  const wait = (ms) => new Promise((r) => setTimeout(r, ms));

  async function setState() {
    if (state !== 'grouped-day') return;
    // set up quietly: nothing takes the keyboard focus, so the page around the frame never scrolls to it
    const focus = HTMLElement.prototype.focus;
    HTMLElement.prototype.focus = function () {};
    try {
      const head = document.querySelector('[data-act="colMenu"][data-k="saved"]'); if (!head) return;
      head.click(); await wait(60);
      const row = [...document.querySelectorAll('.mrow')].find((r) => r.textContent.trim().startsWith('Group by day'));
      if (row) { row.click(); await wait(60); }
      document.querySelectorAll('*').forEach((e) => { if (e.scrollLeft) e.scrollLeft = 0; });
    } finally {
      HTMLElement.prototype.focus = focus;
    }
  }

  function report() {
    const page = document.querySelector('.page'); if (!page) return;
    const send = () => {
      const h = Math.ceil(page.getBoundingClientRect().bottom + parseFloat(getComputedStyle(document.body).paddingBottom || 0));
      window.parent.postMessage({ readyFrameHeight: h }, location.origin);
    };
    new ResizeObserver(send).observe(page);
    if (document.fonts) document.fonts.ready.then(send);
    send();
  }

  addEventListener('DOMContentLoaded', async () => { await setState(); report(); });
})();
