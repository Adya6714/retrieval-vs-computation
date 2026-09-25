/* Pipeline Explorer — site/js/pipeline.js */
(function () {
  const STAGES = [
    { id: 1, label: 'Source' },
    { id: 2, label: 'Transform' },
    { id: 3, label: 'Verify' },
    { id: 4, label: 'Probe 1' },
    { id: 5, label: 'Probe 2' },
    { id: 6, label: 'Probe 3' },
    { id: 7, label: 'What we can say' },
    { id: 8, label: 'Where it goes' },
  ];

  const reduceMotion = matchMedia('(prefers-reduced-motion: reduce)').matches;
  const PLAY_MS = reduceMotion ? 0 : 1600;

  let data = null;
  let itemIdx = 0;
  let stage = 1;
  let variantKey = 'canonical';
  let playing = false;
  let playTimer = null;
  let phaseSel = null;

  const $ = (sel, root) => (root || document).querySelector(sel);

  function tokenize(text) {
    return String(text || '').match(/\S+|\s+/g) || [];
  }

  function isNumberToken(t) {
    return /^-?\d+(\.\d+)?%?$/.test(String(t).replace(/[,$]/g, ''));
  }

  /** Word-level LCS diff. Returns tokens with op: equal|del|ins */
  function wordDiff(aText, bText) {
    const a = tokenize(aText).filter((t) => t.trim() !== '');
    const b = tokenize(bText).filter((t) => t.trim() !== '');
    const n = a.length;
    const m = b.length;
    const dp = Array.from({ length: n + 1 }, () => new Array(m + 1).fill(0));
    for (let i = 1; i <= n; i++) {
      for (let j = 1; j <= m; j++) {
        dp[i][j] =
          a[i - 1] === b[j - 1]
            ? dp[i - 1][j - 1] + 1
            : Math.max(dp[i - 1][j], dp[i][j - 1]);
      }
    }
    const ops = [];
    let i = n;
    let j = m;
    while (i > 0 || j > 0) {
      if (i > 0 && j > 0 && a[i - 1] === b[j - 1]) {
        ops.push({ op: 'equal', text: a[i - 1] });
        i--;
        j--;
      } else if (j > 0 && (i === 0 || dp[i][j - 1] >= dp[i - 1][j])) {
        ops.push({ op: 'ins', text: b[j - 1] });
        j--;
      } else {
        ops.push({ op: 'del', text: a[i - 1] });
        i--;
      }
    }
    ops.reverse();
    return ops;
  }

  function renderDiffHtml(canonical, current) {
    if (canonical === current) {
      return escapeHtml(current);
    }
    const ops = wordDiff(canonical, current);
    return ops
      .map((o) => {
        const t = escapeHtml(o.text);
        if (o.op === 'del') return `<span class="diff-del">${t}</span>`;
        if (o.op === 'ins') {
          const cls = isNumberToken(o.text) ? 'diff-ins diff-num pulse' : 'diff-ins';
          return `<span class="${cls}">${t}</span>`;
        }
        if (isNumberToken(o.text)) return `<span class="diff-num pulse">${t}</span>`;
        return t;
      })
      .join(' ');
  }

  function escapeHtml(s) {
    return String(s)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;');
  }

  function currentItem() {
    return data.items[itemIdx];
  }

  function variantOf(item, key) {
    return (item.variants || []).find((v) => v.variant === key) || item.variants[0];
  }

  function canonicalOf(item) {
    return variantOf(item, 'canonical') || item.variants[0];
  }

  function announce(msg) {
    const live = $('#pipe-live');
    if (live) live.textContent = msg;
  }

  function setMarker() {
    const marker = $('#pipe-rail-marker');
    const btn = document.querySelector(`.pipe-stage-btn[data-stage="${stage}"]`);
    if (!marker || !btn) return;
    marker.style.transform = `translateY(${btn.offsetTop}px)`;
    marker.style.height = `${btn.offsetHeight}px`;
  }

  function findingsFor(item) {
    const fam = String(item.family || '').toLowerCase();
    const vars = new Set((item.variants || []).map((v) => v.variant));
    const out = [];
    if (vars.has('W3') || vars.has('W6')) out.push('F1');
    if (vars.has('W4')) out.push('F2');
    if (fam.includes('planning') || fam.includes('block') || vars.has('W5')) out.push('F3');
    out.push('F4', 'F5');
    if (item.probe2_cci && item.probe2_cci.length) out.push('F6');
    out.push('F7');
    return [...new Set(out)];
  }

  function phaseStatusClass(status) {
    const s = String(status || '').toLowerCase();
    if (s.includes('complete') || s === 'passed') return 'complete';
    if (s.includes('ready') || s.includes('in_progress') || s.includes('next')) return 'ready';
    if (s.includes('block')) return 'blocked';
    if (s.includes('design')) return 'designed';
    return 'planned';
  }

  function mappingPairs(mapping) {
    if (!mapping) return [];
    const inner = mapping.mapping && typeof mapping.mapping === 'object' ? mapping.mapping : mapping;
    if (typeof inner !== 'object') return [];
    return Object.entries(inner).filter(([k]) => k !== 'domain');
  }

  function renderStage() {
    const item = currentItem();
    const canvas = $('#pipe-canvas');
    const can = canonicalOf(item);
    const cur = variantOf(item, variantKey) || can;
    document.querySelectorAll('.pipe-stage-btn').forEach((b) => {
      b.classList.toggle('active', +b.dataset.stage === stage);
    });
    setMarker();
    announce(`Stage ${stage}: ${STAGES[stage - 1].label}. Item ${item.problem_id}.`);

    if (stage === 1) {
      canvas.innerHTML = `
        <div class="pipe-meta">
          <span><b>Family</b> ${escapeHtml(item.family)}</span>
          <span><b>Subtype</b> ${escapeHtml(item.subtype || '—')}</span>
          <span><b>Difficulty</b> ${escapeHtml(item.difficulty || '—')}</span>
          <span><b>Source</b> ${escapeHtml(item.source || '—')}</span>
        </div>
        <div class="pipe-serif">${escapeHtml(can.text)}</div>
        <p class="pipe-caption">This is the item as it appears in the public benchmark.</p>`;
      return;
    }

    if (stage === 2) {
      const vars = item.variants || [];
      const chips = vars
        .map(
          (v) =>
            `<button type="button" class="pipe-var${v.variant === variantKey ? ' active' : ''}" data-var="${escapeHtml(v.variant)}">${escapeHtml(v.variant)}</button>`
        )
        .join('');
      const goldFixed = !['W5', 'W6'].includes(cur.variant);
      const pairs = cur.variant === 'W3' ? mappingPairs(cur.mapping) : [];
      const mapHtml = pairs.length
        ? `<table class="pipe-map"><thead><tr><th>Canonical term</th><th>New term</th></tr></thead><tbody>${pairs
            .map(([a, b]) => `<tr><td>${escapeHtml(a)}</td><td>${escapeHtml(b)}</td></tr>`)
            .join('')}</tbody></table>`
        : '';
      const w3cap =
        cur.variant === 'W3'
          ? `<p class="pipe-caption">As built, W3 moves the problem into a new real-word domain (a cover-story isomorph). A true nonce rename (W3a) is being added so the two can be compared.</p>`
          : `<p class="pipe-caption">${escapeHtml(cur.meaning || '')}</p>`;
      canvas.innerHTML = `
        <div class="pipe-vars">${chips}</div>
        <div class="pipe-gold ${goldFixed ? 'fixed' : 'rederived'}">${goldFixed ? 'Gold fixed' : 'Gold re-derived'}</div>
        <div class="pipe-serif" id="pipe-diff">${renderDiffHtml(can.text, cur.text)}</div>
        <p class="pipe-caption"><b>Gold.</b> <span class="num">${escapeHtml(String(cur.gold || ''))}</span> · ${escapeHtml(cur.gold_rule || '')}</p>
        ${mapHtml}
        ${w3cap}`;
      canvas.querySelectorAll('.pipe-var').forEach((b) => {
        b.addEventListener('click', () => {
          variantKey = b.dataset.var;
          renderStage();
        });
      });
      return;
    }

    if (stage === 3) {
      const vf = cur.verifier_function || can.verifier_function || '—';
      const excl = [];
      (item.variants || []).forEach((v) => {
        (v.probe1 || []).forEach((p) => {
          if (p.included === false && p.exclusion_reason) {
            excl.push(`${p.model}: ${p.exclusion_reason}`);
          }
        });
      });
      const uniq = [...new Set(excl)].slice(0, 6);
      canvas.innerHTML = `
        <div class="pipe-meta">
          <span><b>Verifier kind</b> ${escapeHtml(item.verifier_kind || '—')}</span>
          <span><b>Function</b> <code>${escapeHtml(vf)}</code></span>
        </div>
        <div class="pipe-verify-row">
          <span class="pipe-verify-step" id="pv-s1">Gold answer</span>
          <span aria-hidden="true">→</span>
          <span class="pipe-verify-step" id="pv-s2">Own verifier</span>
          <span class="pipe-check" id="pv-ok" aria-hidden="true">✓</span>
        </div>
        ${uniq.length ? `<div class="pipe-excl">Excluded, never scored — ${uniq.map(escapeHtml).join('; ')}</div>` : ''}
        <p class="pipe-caption">No model is queried until the gold answer passes its own verifier.</p>`;
      const s1 = $('#pv-s1');
      const s2 = $('#pv-s2');
      const ok = $('#pv-ok');
      const t = reduceMotion ? 0 : 350;
      setTimeout(() => s1 && s1.classList.add('on'), t);
      setTimeout(() => {
        if (s2) {
          s2.classList.add('on');
          s2.classList.add('ok');
        }
        if (ok) ok.classList.add('show');
      }, t * 2);
      return;
    }

    if (stage === 4) {
      const models = [];
      const variants = (item.variants || []).map((v) => v.variant);
      const cell = {};
      (item.variants || []).forEach((v) => {
        (v.probe1 || []).forEach((p) => {
          if (!models.includes(p.model)) models.push(p.model);
          cell[`${p.model}||${v.variant}`] = p;
        });
      });
      let head = '<tr><th></th>' + variants.map((v) => `<th>${escapeHtml(v)}</th>`).join('') + '</tr>';
      let body = models
        .map((m) => {
          const tds = variants
            .map((v) => {
              const p = cell[`${m}||${v}`];
              let cls = 'na';
              if (p && p.included === false) cls = 'na';
              else if (p && p.correct === true) cls = 'ok';
              else if (p && p.correct === false) cls = 'bad';
              return `<td><div class="p1-cell ${cls}" data-col="${v}"></div></td>`;
            })
            .join('');
          return `<tr><th class="row-h">${escapeHtml(m)}</th>${tds}</tr>`;
        })
        .join('');
      const ret = models
        .map((m) => {
          const c = cell[`${m}||canonical`];
          const w = cell[`${m}||W3`];
          const cOk = c && c.included !== false && c.correct === true;
          const wOk = w && w.included !== false && w.correct === true;
          return `<div class="ret-item"><span>${escapeHtml(m)}</span>
            <span class="ret-dot ${cOk ? 'on' : 'off'}" title="canonical"></span>
            <span aria-hidden="true">→</span>
            <span class="ret-dot ${wOk ? 'on' : 'off'}" title="W3"></span>
            <b>${cOk ? '1' : '0'}→${wOk ? '1' : '0'}</b></div>`;
        })
        .join('');
      canvas.innerHTML = `
        <div class="p1-grid-wrap"><table class="p1-grid"><thead>${head}</thead><tbody>${body}</tbody></table></div>
        <div class="ret-row">${ret}</div>
        <p class="pipe-caption">n = 1 item. Aggregate retention: <a href="../results/derived/probe1_per_model_variant.csv" target="_blank" rel="noopener">probe1_per_model_variant.csv</a>.</p>`;
      const cols = variants.slice();
      cols.forEach((v, ci) => {
        const delay = reduceMotion ? 0 : ci * 120;
        setTimeout(() => {
          canvas.querySelectorAll(`.p1-cell[data-col="${CSS.escape(v)}"]`).forEach((el) => el.classList.add('show'));
        }, delay);
      });
      return;
    }

    if (stage === 5) {
      const rows = item.probe2_cci || [];
      if (!rows.length) {
        canvas.innerHTML = `<p class="pipe-caption" style="margin:0">Probe 2 was not run on this item.</p>`;
        return;
      }
      const list = rows
        .map((r) => {
          const cci = r.cci != null ? Number(r.cci).toFixed(3) : '—';
          return `<div class="cci-row"><span>${escapeHtml(r.model || r.model_id || 'model')}</span><b>CCI ${cci}</b></div>`;
        })
        .join('');
      canvas.innerHTML = `
        <div class="cci-list">${list}</div>
        <button type="button" class="pipe-ctrl primary" id="pipe-replay-p2">Replay how this is measured</button>
        <p class="pipe-caption">Scrolls to the staged Probe 2 player in Methodology and starts it.</p>`;
      $('#pipe-replay-p2')?.addEventListener('click', () => {
        if (typeof window.startProbe2Player === 'function') window.startProbe2Player();
      });
      return;
    }

    if (stage === 6) {
      const p3 = item.probe3_proximity || {};
      canvas.innerHTML = `
        <svg class="strip" id="pipe-strip" role="img" aria-label="Contamination strip plot"></svg>
        <p class="pipe-caption">Within-family exposure proxy from Infini-gram over public corpora. Not membership evidence for any closed model.
        Score <span class="num">${p3.contamination_score != null ? Number(p3.contamination_score).toFixed(4) : '—'}</span>,
        family percentile <span class="num">${p3.family_percentile != null ? Number(p3.family_percentile).toFixed(3) : '—'}</span>.</p>`;
      drawStrip(p3);
      return;
    }

    if (stage === 7) {
      const tags = findingsFor(item)
        .map((f) => `<span class="f-tag">${f}</span>`)
        .join('');
      canvas.innerHTML = `
        <div class="verdict">Probe-level evidence recorded. No per-instance retrieval or computation label is issued: the labelling rule is not calibrated until gate G1.</div>
        <div class="f-tags">${tags}</div>
        <p class="pipe-caption">Findings this item can contribute to, given its family and variants.</p>`;
      return;
    }

    if (stage === 8) {
      const phases = data.phase_routing || [];
      if (phaseSel == null) phaseSel = phases[0]?.phase ?? null;
      const track = phases
        .map((p) => {
          const cls = phaseStatusClass(p.status);
          const active = String(p.phase) === String(phaseSel) ? ' active' : '';
          return `<button type="button" class="pipe-phase ${cls}${active}" data-phase="${escapeHtml(String(p.phase))}">${escapeHtml(String(p.phase))}</button>`;
        })
        .join('');
      const sel = phases.find((p) => String(p.phase) === String(phaseSel)) || phases[0];
      const ids = (sel?.claims || [])
        .map((id) => `<button type="button" data-pid="${escapeHtml(id)}">${escapeHtml(id)}</button>`)
        .join('');
      canvas.innerHTML = `
        <div class="phase-track">${track}</div>
        <div class="phase-detail">
          <div><b>Phase ${escapeHtml(String(sel?.phase ?? ''))}</b> · ${escapeHtml(sel?.name || '')} · ${escapeHtml(sel?.status || '')}</div>
          <p style="margin:.45rem 0 0;max-width:none">${escapeHtml(sel?.uses_item || '')}</p>
          <div class="ids">${ids}</div>
        </div>`;
      canvas.querySelectorAll('.pipe-phase').forEach((b) => {
        b.addEventListener('click', () => {
          phaseSel = b.dataset.phase;
          renderStage();
        });
      });
      canvas.querySelectorAll('.phase-detail button[data-pid]').forEach((b) => {
        b.addEventListener('click', () => {
          const id = b.dataset.pid;
          document.getElementById('programme')?.scrollIntoView({
            behavior: reduceMotion ? 'auto' : 'smooth',
          });
          const tab = document.querySelector(
            id.startsWith('H') || id.startsWith('h')
              ? '#prog-tabs [data-view="hypotheses"]'
              : '#prog-tabs [data-view="claims"]'
          );
          tab?.click();
          if (typeof window.setProgrammeFocus === 'function') window.setProgrammeFocus(id);
          else if (typeof setFocus === 'function') setFocus(id);
        });
      });
    }
  }

  function drawStrip(p3) {
    const el = $('#pipe-strip');
    if (!el || typeof d3 === 'undefined') return;
    const w = el.clientWidth || 420;
    const h = 72;
    const svg = d3.select(el).attr('viewBox', `0 0 ${w} ${h}`);
    svg.selectAll('*').remove();
    const score = Number(p3.contamination_score);
    const pct = Number(p3.family_percentile);
    const cloud = Array.isArray(p3.family_scores)
      ? p3.family_scores.map(Number).filter((d) => !Number.isNaN(d))
      : [];
    const xMax = cloud.length ? Math.max(1, d3.max(cloud)) : 1;
    const x = d3.scaleLinear().domain([0, xMax]).range([16, w - 16]);
    svg
      .append('line')
      .attr('x1', 16)
      .attr('x2', w - 16)
      .attr('y1', h / 2)
      .attr('y2', h / 2)
      .attr('stroke', 'var(--rule)')
      .attr('stroke-width', 2);
    svg
      .selectAll('circle.cloud')
      .data(cloud)
      .join('circle')
      .attr('class', 'cloud')
      .attr('cx', (d) => x(d))
      .attr('cy', h / 2)
      .attr('r', 2.8)
      .attr('fill', 'var(--muted)')
      .attr('opacity', 0.4);
    if (!Number.isNaN(score)) {
      svg
        .append('circle')
        .attr('cx', x(Math.min(xMax, Math.max(0, score))))
        .attr('cy', h / 2)
        .attr('r', 6)
        .attr('fill', 'var(--accent)')
        .attr('stroke', 'var(--paper)')
        .attr('stroke-width', 2);
    }
  }

  function goStage(n) {
    stage = Math.max(1, Math.min(8, n));
    renderStage();
  }

  function stopPlay() {
    playing = false;
    clearTimeout(playTimer);
    const btn = $('#pipe-play');
    if (btn) btn.textContent = 'Play';
  }

  function play() {
    if (reduceMotion) {
      goStage(Math.min(8, stage + 1));
      return;
    }
    if (playing) {
      stopPlay();
      return;
    }
    playing = true;
    const btn = $('#pipe-play');
    if (btn) btn.textContent = 'Pause';
    const tick = () => {
      if (!playing) return;
      if (stage >= 8) {
        stopPlay();
        return;
      }
      goStage(stage + 1);
      if (stage < 8) playTimer = setTimeout(tick, PLAY_MS);
      else stopPlay();
    };
    playTimer = setTimeout(tick, PLAY_MS);
  }

  function selectItem(i) {
    stopPlay();
    itemIdx = i;
    stage = 1;
    variantKey = 'canonical';
    phaseSel = null;
    document.querySelectorAll('.pipe-chip').forEach((b, idx) => b.classList.toggle('active', idx === i));
    renderStage();
  }

  function showError(msg) {
    const root = $('#pipeline .inner') || $('#pipeline');
    if (!root) return;
    const existing = $('#pipe-error');
    if (existing) existing.remove();
    const p = document.createElement('p');
    p.className = 'pipe-error';
    p.id = 'pipe-error';
    p.textContent = msg;
    root.insertBefore(p, root.firstChild?.nextSibling || null);
  }

  function buildShell() {
    const host = $('#pipe-root');
    if (!host) return;
    const chips = data.items
      .map(
        (it, i) =>
          `<button type="button" class="pipe-chip${i === 0 ? ' active' : ''}" data-i="${i}">${escapeHtml(it.problem_id)}</button>`
      )
      .join('');
    const rail = STAGES.map(
      (s) =>
        `<button type="button" class="pipe-stage-btn${s.id === 1 ? ' active' : ''}" data-stage="${s.id}"><span class="n">${s.id}</span>${escapeHtml(s.label)}</button>`
    ).join('');
    host.innerHTML = `
      <div class="pipe-shell" id="pipe-shell">
        <div class="pipe-top">
          <div class="pipe-items"><span class="label">Item</span>${chips}</div>
          <div class="pipe-ctrls">
            <button type="button" class="pipe-ctrl primary" id="pipe-play">Play</button>
            <button type="button" class="pipe-ctrl" id="pipe-step">Step</button>
            <button type="button" class="pipe-ctrl" id="pipe-reset">Reset</button>
          </div>
        </div>
        <div class="pipe-body">
          <nav class="pipe-rail" aria-label="Pipeline stages">
            <div class="pipe-rail-marker" id="pipe-rail-marker"></div>
            ${rail}
          </nav>
          <div class="pipe-canvas" id="pipe-canvas" tabindex="0"></div>
        </div>
        <div class="pipe-live" id="pipe-live" aria-live="polite"></div>
      </div>`;

    host.querySelectorAll('.pipe-chip').forEach((b) => {
      b.addEventListener('click', () => selectItem(+b.dataset.i));
    });
    host.querySelectorAll('.pipe-stage-btn').forEach((b) => {
      b.addEventListener('click', () => {
        stopPlay();
        goStage(+b.dataset.stage);
      });
    });
    $('#pipe-play')?.addEventListener('click', play);
    $('#pipe-step')?.addEventListener('click', () => {
      stopPlay();
      goStage(Math.min(8, stage + 1));
    });
    $('#pipe-reset')?.addEventListener('click', () => {
      stopPlay();
      goStage(1);
    });

    const canvas = $('#pipe-canvas');
    canvas?.addEventListener('keydown', (e) => {
      if (e.key === 'ArrowRight' || e.key === 'ArrowDown') {
        e.preventDefault();
        stopPlay();
        goStage(Math.min(8, stage + 1));
      }
      if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') {
        e.preventDefault();
        stopPlay();
        goStage(Math.max(1, stage - 1));
      }
    });

    renderStage();
    requestAnimationFrame(setMarker);
  }

  async function boot() {
    const root = $('#pipe-root');
    if (!root) return;
    try {
      const res = await fetch('data/pipeline_explorer.json', { cache: 'no-store' });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      data = await res.json();
      if (!data.items || !data.items.length) throw new Error('empty items[]');
      buildShell();
    } catch (err) {
      showError(
        `Could not load data/pipeline_explorer.json (${err.message}). Rebuild with: PYTHONPATH=. python scripts/site/build_pipeline_explorer_data.py`
      );
    }
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', boot);
  else boot();
})();
