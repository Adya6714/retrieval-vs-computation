/* C6 content renderers — applications, position, Track T cards, footer cite. */
(function () {
  const GH = 'https://github.com/Adya6714/retrieval-vs-computation/blob/main';

  function esc(s) {
    return String(s ?? '')
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;');
  }

  function renderApplications() {
    const root = document.getElementById('app-list');
    const filters = document.getElementById('app-filters');
    if (!root || !filters || !DATA.applications) return;
    let active = 'all';
    const areas = ['all', ...DATA.applications.map((a) => a.area)];
    function paint() {
      filters.innerHTML = '';
      areas.forEach((a) => {
        const b = document.createElement('button');
        b.type = 'button';
        b.textContent = a === 'all' ? 'All' : a;
        b.className = a === active ? 'active' : '';
        b.addEventListener('click', () => {
          active = a;
          paint();
        });
        filters.appendChild(b);
      });
      root.innerHTML = '';
      DATA.applications
        .filter((row) => active === 'all' || row.area === active)
        .forEach((row) => {
          const d = document.createElement('div');
          d.className = 'app-row';
          d.innerHTML = `<div>
              <h3 class="app-h">${esc(row.area)}</h3>
              <p>${esc(row.adds)}</p>
              <div class="dir">${esc(row.direction)}</div>
            </div>
            <span class="badge ${esc(row.status)}">${esc(row.status)}</span>`;
          root.appendChild(d);
        });
    }
    paint();
  }

  function renderPosition() {
    const root = document.getElementById('pos-toggles');
    if (!root || !DATA.positionToggles) return;
    root.innerHTML = '';
    DATA.positionToggles.forEach((t) => {
      const b = document.createElement('button');
      b.type = 'button';
      b.className = 'pos-toggle';
      b.innerHTML = `<div class="pt-head"><span class="pt-id">${esc(t.id)}</span><span>${esc(t.title)}</span></div>
        <div class="pt-body">
          <p style="margin:0;max-width:none"><strong>Finding.</strong> ${esc(t.finding)}</p>
          <p style="margin:.45rem 0 0;max-width:none"><strong>Human analogue.</strong> ${esc(t.analogue)}</p>
          <p class="verify">to be verified in full text</p>
        </div>`;
      b.addEventListener('click', () => b.classList.toggle('open'));
      root.appendChild(b);
    });
  }

  function renderTrackNext() {
    const root = document.getElementById('next-grid');
    const tiers = document.getElementById('method-tiers');
    if (tiers) tiers.hidden = true;
    if (!root || !DATA.trackT) return;
    root.innerHTML = '';
    DATA.trackT.forEach((n) => {
      const d = document.createElement('div');
      d.className = 'next-card' + (n.priority ? ' priority' : '');
      d.innerHTML = `<div class="tier">${esc(n.id)}</div>
        <h3>${esc(n.title)}</h3>
        <p class="q">${esc(n.hypothesis)}</p>
        <div class="meta">
          <span class="status-pill">${esc(n.status)}</span>
          <span class="status-pill">${esc(n.cost)}</span>
        </div>
        <div class="body"><dl>
          <div><dt>Primary contrast</dt><dd>${esc(n.contrast)}</dd></div>
          <div><dt>Kill / decision criterion</dt><dd>${esc(n.kill)}</dd></div>
          <div><dt>Serves</dt><dd>${esc(n.serves)}</dd></div>
          ${n.link ? `<div><dt>Note</dt><dd><a href="${GH}/${esc(n.link)}" target="_blank" rel="noopener">${esc(n.linkLabel || n.link)}</a></dd></div>` : ''}
        </dl></div>`;
      d.onclick = () => d.classList.toggle('open');
      root.appendChild(d);
    });
  }

  function wireFooter() {
    const btn = document.getElementById('bib-copy');
    const pre = document.getElementById('bib-text');
    if (btn && pre) {
      btn.addEventListener('click', async () => {
        try {
          await navigator.clipboard.writeText(pre.textContent);
          btn.textContent = 'Copied';
          btn.classList.add('done');
          setTimeout(() => {
            btn.textContent = 'Copy';
            btn.classList.remove('done');
          }, 1600);
        } catch (_) {
          btn.textContent = 'Select text to copy';
        }
      });
    }
    const built = document.getElementById('built-date');
    if (built && !built.dataset.locked) {
      built.textContent = 'Last built: 2026-09-25';
    }
  }

  function boot() {
    renderApplications();
    renderPosition();
    renderTrackNext();
    wireFooter();
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', boot);
  else boot();
})();
