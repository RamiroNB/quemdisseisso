// Tela 2 · Audiência — the designer's wheel, participation map, summary and fairness panels, on real data
// (app/portal/data/h/<id>.json, written by scripts/export_portal_data.py).
import { html, Component, Icon, getJSON, go, replaceRoute, p3, pct, initials, shortName, firstName } from './lib.js';

const MAX_WHEEL = 18, MAX_MAP = 14, MAX_BARS = 18;
const wheelR = () => Math.round(Math.max(210, Math.min(390, window.innerWidth * 0.56)));   // 390 on desktop, as drawn
const POS = 'sustentada', NEG = 'nao_sustentada';
const NUM = ['zero', 'uma', 'duas', 'três', 'quatro', 'cinco', 'seis', 'sete', 'oito', 'nove', 'dez', 'onze', 'doze'];
const num = n => n < NUM.length ? NUM[n] : String(n);
const cap = s => s[0].toUpperCase() + s.slice(1);
const verdictOf = o => o && o.v.sonnet && o.v.sonnet.s;
// an alarm event whose rewrite has support where the uncorrected summary had none. When the rewrite shifted the
// rest of the summary (e.b == null), the counterpart is the uncorrected line at the same position.
const fixedFrom = (sum, e) => e.r == null ? null : e.b != null ? e.b : e.r;
const isFix = (sum, e) => { const b = fixedFrom(sum, e); return b != null && verdictOf(sum.baseline[b]) === NEG && verdictOf(sum.rag[e.r]) === POS; };
// the before/after toggle is offered only to people whose summary the correction fixed
const hasFix = sum => !!sum && (sum.status === 'fixed' || sum.events.some(e => isFix(sum, e)));

function pickDefault(h) {
  // the person whose correction story the hearing is featured for, else the first person with a summary
  const order = { A: 0, B: 1, C: 2 };
  const withSum = h.speakers.filter(s => s.summary);
  const best = withSum.map(s => ({ s, m: h.summaries[s.slug] }))
    .sort((a, b) => ((a.m.status === 'fixed' ? 0 : 1) - (b.m.status === 'fixed' ? 0 : 1)) || ((order[a.m.tier] ?? 9) - (order[b.m.tier] ?? 9)) || (a.s.rank - b.s.rank));
  return (best[0] && best[0].s.slug) || h.speakers[0].slug;
}

export class Hearing extends Component {
  state = { h: null, sel: 0, hover: null, mapSel: null, mapHov: null, mapMax: false, corrected: !this.props.raw, typed: 0, tipX: 0, tipY: 0 };
  angle = 0; angleTo = null; mt = 0; mEW = null;

  componentDidMount() {
    this.onResize = () => this.forceUpdate();
    window.addEventListener('resize', this.onResize);
    getJSON(`data/h/${this.props.id}.json`).then(h => this.setData(h), () => this.setState({ h: false }));
    this.onKey = e => { if (e.key === 'Escape' && this.state.mapMax) this.setState({ mapMax: false }); };
    window.addEventListener('keydown', this.onKey);
    let last = performance.now();
    const loop = t => { const dt = Math.min(0.05, (t - last) / 1000); last = t; this.tick(dt); this.raf = requestAnimationFrame(loop); };
    this.raf = requestAnimationFrame(loop);
  }
  componentWillUnmount() {
    window.removeEventListener('keydown', this.onKey); window.removeEventListener('resize', this.onResize); document.body.style.overflow = '';
    cancelAnimationFrame(this.raf); clearInterval(this.typing); if (this.io) this.io.disconnect();
    if (this.wrap && this.onWheelSp) this.wrap.removeEventListener('wheel', this.onWheelSp);
  }
  componentDidUpdate(pp, ps) {
    if (ps.mapMax !== this.state.mapMax) document.body.style.overflow = this.state.mapMax ? 'hidden' : '';
    if (pp.person !== this.props.person && this.state.h && this.props.person) {
      const i = this.people.findIndex(s => s.slug === this.props.person);
      if (i >= 0 && i !== this.state.sel) this.select(i, false);
    }
  }

  setData(h) {
    // who appears in the wheel, the map and the bars: the most floor time, and everyone with a summary
    const keep = new Set(h.speakers.filter(s => s.summary).map(s => s.slug));
    h.speakers.forEach(s => { if (keep.size < MAX_WHEEL) keep.add(s.slug); });
    this.people = h.speakers.filter(s => keep.has(s.slug));
    const mapKeep = new Set(this.people.filter(s => s.summary).map(s => s.slug));
    this.people.forEach(s => { if (mapKeep.size < MAX_MAP) mapKeep.add(s.slug); });
    this.mapPeople = this.people.filter(s => mapKeep.has(s.slug));
    const slug = this.props.person && this.people.some(s => s.slug === this.props.person) ? this.props.person : pickDefault(h);
    const sel = Math.max(0, this.people.findIndex(s => s.slug === slug));
    const N = this.people.length;
    this.copies = N >= 10 ? 1 : Math.ceil(MAX_WHEEL / N);
    this.step = 360 / (N * this.copies);
    this.angle = -sel * this.step;
    this.setState({ h, sel }, () => { if (this.props.q && this.props.q.digitar === '0') this.startTyping(); });
    replaceRoute(`#/audiencia/${h.id}/${this.people[sel].slug}` + (this.state.corrected ? '' : '/sem-correcao'));
  }

  select(i, fromUser = true, patch = {}) {
    this.focusWheel(i);
    this.mEW = null;
    this.restartTyping({ sel: i, corrected: true, mapSel: fromUser ? i : this.state.mapSel, ...patch });
    replaceRoute(`#/audiencia/${this.state.h.id}/${this.people[i].slug}`);
  }

  focusWheel(i) {
    const N = this.people.length, cands = [];
    for (let c = 0; c < this.copies; c++) cands.push(-(i + c * N) * this.step);
    let best = null, bd = 1e9;
    cands.forEach(c => { const k = Math.round((this.angle - c) / 360); const v = c + k * 360; const d = Math.abs(v - this.angle); if (d < bd) { bd = d; best = v; } });
    this.angleTo = best;
  }

  // ---- refs --------------------------------------------------------------------------------------
  wheelRef = el => { this.wheel = el; };
  wrapRef = el => {
    if (this.wrap && this.onWheelSp) this.wrap.removeEventListener('wheel', this.onWheelSp);
    this.wrap = el;
    if (!el) return;
    this.onWheelSp = e => {
      const dx = Math.abs(e.deltaX) > Math.abs(e.deltaY) ? e.deltaX : (e.shiftKey ? e.deltaY : 0);
      if (!dx) return;
      e.preventDefault();
      this.angle -= dx * (e.deltaMode === 1 ? 16 : 1) * 0.12;
    };
    el.addEventListener('wheel', this.onWheelSp, { passive: false });
  };
  sumRef = el => {
    if (this.io) this.io.disconnect();
    this.sumEl = el;
    if (!el) return;
    this.io = new IntersectionObserver(es => { this.sumVisible = es[0].isIntersecting; if (this.sumVisible) this.startTyping(); }, { threshold: 0.2 });
    this.io.observe(el);
  };
  mapStageRef = el => { this.mapStage = el; };
  mapCanvasRef = el => { this.mapCanvas = el; };

  // ---- animation ---------------------------------------------------------------------------------
  tick(dt) {
    this.drawMap(dt);
    if (this.wheel) {
      if (this.angleTo != null) { const d = this.angleTo - this.angle; this.angle += d * Math.min(1, dt * 5); if (Math.abs(d) < 0.2) { this.angle = this.angleTo; this.angleTo = null; } }
      else if (this.state.hover == null) this.angle += 5 * dt;
      this.wheel.style.transform = `rotate(${this.angle.toFixed(2)}deg)`;
    }
  }
  startTyping() {
    if (this.props.q && this.props.q.digitar === '0') { if (this.state.typed < 1e9) this.setState({ typed: 1e9 }); return; }
    if (this.typing || this.state.typed >= this.total) return;
    this.typing = setInterval(() => {
      this.setState(s => {
        const t = Math.min(this.total, s.typed + 3);
        if (t >= this.total) { clearInterval(this.typing); this.typing = null; }
        return { typed: t };
      });
    }, 16);
  }
  restartTyping(patch) {
    clearInterval(this.typing); this.typing = null;
    this.setState(Object.assign({ typed: 0 }, patch), () => { if (this.sumVisible || (this.props.q && this.props.q.digitar === '0')) this.startTyping(); });
  }

  drawMap(dt) {
    const st = this.mapStage, cv = this.mapCanvas, h = this.state.h; if (!st || !cv || !h) return;
    const W = st.clientWidth, H = st.clientHeight, dpr = window.devicePixelRatio || 1;
    if (!W || !H) return;
    if (cv.width !== Math.round(W * dpr) || cv.height !== Math.round(H * dpr)) { cv.width = Math.round(W * dpr); cv.height = Math.round(H * dpr); }
    const x = cv.getContext('2d'); x.setTransform(dpr, 0, 0, dpr, 0, 0); x.clearRect(0, 0, W, H);
    this.mt += dt; const t = this.mt;
    if (!this.parts) { this.parts = []; for (let i = 0; i < 130; i++) this.parts.push({ x: Math.random(), y: Math.random(), vx: (Math.random() - .5) * 0.014, vy: (Math.random() - .5) * 0.014, r: 0.8 + Math.random() * 1.8, b: Math.random() < 0.45 }); }
    const mx = this.mmx, my = this.mmy;
    const P = this.parts.map(p => {
      p.x = (p.x + p.vx * dt + 1) % 1; p.y = (p.y + p.vy * dt + 1) % 1;
      let px = p.x * W, py = p.y * H;
      if (mx != null) { const dx = px - mx, dy = py - my, d = Math.hypot(dx, dy) || 1; if (d < 130) { const f = (1 - d / 130) * 30; px += dx / d * f; py += dy / d * f; } }
      return [px, py, p];
    });
    x.lineWidth = 1;
    for (let i = 0; i < P.length; i++) for (let j = i + 1; j < P.length; j++) {
      const dx = P[i][0] - P[j][0], dy = P[i][1] - P[j][1], d2 = dx * dx + dy * dy;
      if (d2 < 8100) { x.strokeStyle = 'rgba(75,104,227,' + ((1 - Math.sqrt(d2) / 90) * 0.18).toFixed(3) + ')'; x.beginPath(); x.moveTo(P[i][0], P[i][1]); x.lineTo(P[j][0], P[j][1]); x.stroke(); }
    }
    P.forEach(([px, py, p]) => { x.fillStyle = p.b ? 'rgba(75,104,227,0.45)' : 'rgba(20,24,28,0.2)'; x.beginPath(); x.arc(px, py, p.r, 0, 7); x.fill(); });
    const mp = this.mapPeople, TH = h.themes;
    const selIdx = this.state.mapSel != null ? mp.indexOf(this.people[this.state.mapSel]) : -1;
    const sel = selIdx >= 0 && W > 900;
    const tgt = sel ? W - 410 : W;
    this.mEW = this.mEW == null ? tgt : this.mEW + (tgt - this.mEW) * Math.min(1, dt * 4);
    const EW = this.mEW, cx = EW / 2, cy = H / 2 - 10, narrow = W < 700;
    const rxP = EW * (narrow ? 0.36 : 0.38), ryP = H * 0.36, rxT = EW * 0.19, ryT = H * 0.19;
    const wob = (i, k) => [Math.sin(t * 0.35 + i * 1.3 + k) * 12, Math.cos(t * 0.3 + i * 1.9 + k) * 10];
    const tp = TH.map((th, j) => { const an = -Math.PI / 2 + j / TH.length * Math.PI * 2 + 0.35; const w = wob(j, 5); return [cx + Math.cos(an) * rxT + w[0], cy + Math.sin(an) * ryT + w[1]]; });
    const pp = mp.map((s, i) => { const an = -Math.PI / 2 + i / mp.length * Math.PI * 2; const w = wob(i, 0); return [cx + Math.cos(an) * rxP + w[0], cy + Math.sin(an) * ryP + w[1]]; });
    const ow = wob(0, 9), O = [cx + ow[0] * 0.4, cy + ow[1] * 0.4];
    const hovIdx = this.state.mapHov != null ? this.state.mapHov : null;
    const act = hovIdx != null ? hovIdx : (selIdx >= 0 ? selIdx : null);
    const maxMin = Math.max(...mp.map(s => s.min));
    mp.forEach((s, i) => s.themes.forEach(j => {
      if (!tp[j]) return;
      const on = i === act;
      x.setLineDash(on ? [6, 6] : []); x.lineDashOffset = on ? -t * 30 : 0;
      x.strokeStyle = on ? 'rgba(0,136,255,0.9)' : (s.named ? 'rgba(75,104,227,' + (act != null ? 0.1 : 0.3) + ')' : 'rgba(122,132,142,' + (act != null ? 0.08 : 0.25) + ')');
      x.lineWidth = on ? 2 : Math.min(2.2, 0.6 + 1.6 * s.min / maxMin);
      x.beginPath(); x.moveTo(pp[i][0], pp[i][1]); x.lineTo(tp[j][0], tp[j][1]); x.stroke();
    }));
    const actT = act != null ? mp[act].themes : [];
    TH.forEach((th, j) => {
      const on = actT.includes(j) && th.in_article;
      x.setLineDash(th.in_article ? (on ? [6, 6] : []) : [3, 5]); x.lineDashOffset = on ? -t * 30 : 0;
      x.strokeStyle = th.in_article ? (on ? 'rgba(0,136,255,0.95)' : 'rgba(75,104,227,0.45)') : 'rgba(122,132,142,0.35)';
      x.lineWidth = th.in_article ? (on ? 2.4 : 1.6) : 1;
      x.beginPath(); x.moveTo(tp[j][0], tp[j][1]); x.lineTo(O[0], O[1]); x.stroke();
    });
    x.setLineDash([]);
    x.textAlign = 'center';
    TH.forEach((th, j) => {
      const lit = actT.includes(j), [a, b] = tp[j];
      if (lit) { x.shadowColor = 'rgba(0,136,255,0.7)'; x.shadowBlur = 18; }
      x.beginPath(); x.arc(a, b, 7, 0, 7); x.fillStyle = th.in_article ? '#0088FF' : '#fff'; x.fill(); x.shadowBlur = 0;
      if (!th.in_article) { x.strokeStyle = '#B9C7F6'; x.lineWidth = 1; x.stroke(); }
      x.font = "600 12px 'Plus Jakarta Sans', sans-serif"; x.fillStyle = lit ? '#0E2740' : (act != null ? '#9AA3AD' : '#5C6771'); x.fillText(th.label, a, b - 14);
    });
    for (let k = 0; k < 2; k++) {
      const ph = ((t * 0.35 + k * 0.5) % 1);
      x.strokeStyle = 'rgba(75,104,227,' + (0.35 * (1 - ph)).toFixed(3) + ')'; x.lineWidth = 1;
      x.beginPath(); x.ellipse(O[0], O[1], 90 + ph * 70, 26 + ph * 34, 0, 0, 7); x.stroke();
    }
    x.font = "600 13px 'Plus Jakarta Sans', sans-serif";
    const lbl = 'Matéria da Agência Câmara', tw = x.measureText(lbl).width, pw = tw + 36, ph2 = 38;
    x.shadowColor = 'rgba(14,39,64,0.25)'; x.shadowBlur = 20; x.shadowOffsetY = 6;
    x.fillStyle = '#0E2740'; x.beginPath(); x.roundRect(O[0] - pw / 2, O[1] - ph2 / 2, pw, ph2, 19); x.fill();
    x.shadowBlur = 0; x.shadowOffsetY = 0;
    x.fillStyle = '#FFFFFF'; x.textBaseline = 'middle'; x.fillText(lbl, O[0], O[1] + 1); x.textBaseline = 'alphabetic';
    st.querySelectorAll('[data-pnode]').forEach((el, i) => { if (pp[i]) el.style.transform = 'translate(' + pp[i][0].toFixed(1) + 'px,' + pp[i][1].toFixed(1) + 'px)'; });
  }

  // ---- summary lines -------------------------------------------------------------------------------
  buildLines(sum, corrected, tau) {
    const L = corrected ? sum.rag : sum.baseline, nWin = sum.windows.length;
    return L.map((o, i) => {
      const vS = o.v.sonnet && o.v.sonnet.s, vO = o.v.opus && o.v.opus.s, ok = vS === POS;
      // the alarm is shown only on lines the correction fixed
      const fixes = sum.events.filter(e => isFix(sum, e));
      const ev = fixes.find(e => corrected ? e.r === i : e.b === i);
      const moved = !corrected && !ev && fixes.some(e => e.b == null && e.r === i);
      let fixed = false, note = null, verdict = ok ? 'com apoio' : 'sem apoio na fala';
      if (ev) {
        const at = `Alarme em ${p3(ev.p)} (limiar ${p3(tau)})` + (ev.next ? `, logo antes de “${ev.next}”` : '') + '.';
        const did = `Buscamos ${ev.win.length} dos ${nWin} trechos da fala e reescrevemos a linha.`;
        if (!corrected) note = `Na versão com intervenção, o alarme tocou nesta linha (${p3(ev.p)})` + (ev.next ? `, logo antes de “${ev.next}”` : '') + ', e ela foi reescrita com apoio na fala.';
        else {
          fixed = true; verdict = 'com apoio · consertada';
          const before = ev.b != null ? 'Sem a intervenção, a linha era:' : `Na versão sem intervenção, a linha ${i + 1} era:`;
          note = html`${at} ${did} ${before} <q>${sum.baseline[fixedFrom(sum, ev)].t}</q>`;
        }
      } else if (moved) note = 'Na versão com intervenção, esta linha foi reescrita e passou a ter apoio na fala.';
      const comment = !ok && o.v.sonnet && o.v.sonnet.c ? o.v.sonnet.c : null;
      const disagree = vO && vS && vO !== vS ? (vO === POS ? 'O segundo juiz automático julgou esta linha com apoio.' : 'O segundo juiz automático julgou esta linha sem apoio.') : null;
      return { n: i + 1, text: o.t, ok, fixed, alarm: !!ev, verdict, note, comment, disagree };
    });
  }

  render({ index }, st) {
    const h = st.h;
    if (h === false) return html`<div class="loading">Audiência não encontrada. <a href="#/">Voltar ao início</a></div>`;
    if (!h) return html`<div class="loading">Carregando a audiência…</div>`;
    const C = index.corpus, people = this.people, N = h.speakers.length;
    const sp = people[st.sel], sum = h.summaries[sp.slug];
    const cited = s => s.named === true, citedWord = s => s.named == null ? 'citação não verificada' : s.named ? 'citada na matéria' : 'não citada na matéria';
    const rankLabel = s => s.rank === 1 ? `o maior tempo entre as ${N} pessoas` : `o ${s.rank}º maior tempo entre as ${N} pessoas`;
    const citedSentence = s => s.named == null ? 'Não conseguimos verificar se a matéria da Agência Câmara citou esta pessoa.' : s.named ? 'A matéria da Agência Câmara citou esta pessoa.' : 'A matéria da Agência Câmara não citou esta pessoa.';
    const chairNote = s => s.chair ? (s.chair_role === 'Relator' || s.chair_role === 'Relatora' ? ' Foi relator(a) da sessão.' : ' Presidiu a sessão.') : '';

    // intro paragraph: what the article did, in this hearing
    const k = h.n_named;
    const introCited = k === 0 ? 'a matéria não cita nenhuma delas pelo nome.' : h.top_named ? (k === 1 ? 'a matéria cita só quem mais falou.' : `a matéria cita as ${num(k)} que mais falaram.`) : `a matéria cita ${num(k)} delas.`;
    const intro = `A matéria da Agência Câmara saiu com a manchete “${h.headline}”. ${cap(num(N))} pessoas falaram; ${introCited} Escolha uma pessoa para ver o replay do detector e o resumo da fala dela.`;

    // wheel
    const Nw = people.length, R = wheelR();
    const tiles = [];
    for (let j = 0; j < Nw * this.copies; j++) {
      const i = j % Nw, s = people[i];
      const cls = 'tile ' + (i === st.sel ? 'sel' : i === st.hover ? 'hov' : s.summary ? 'has' : 'none');
      tiles.push(html`<div class="slot" style=${{ transform: `rotate(${j * this.step}deg) translateY(-${R}px)` }}>
        <div class=${cls} role="button" tabindex="0" aria-label=${s.name}
          onClick=${() => this.select(i)} onKeyDown=${e => e.key === 'Enter' && this.select(i)}
          onMouseEnter=${e => { const r = e.currentTarget.getBoundingClientRect(), w = this.wrap ? this.wrap.getBoundingClientRect() : { left: 0, top: 0 }; this.setState({ hover: i, tipX: r.left + r.width / 2 - w.left, tipY: r.top - w.top - 10 }); }}
          onMouseLeave=${() => this.setState({ hover: null })}><span>${initials(s.name)}</span></div>
      </div>`);
    }
    const hs = st.hover != null ? people[st.hover] : null;

    // map
    const selMapIdx = st.mapSel != null ? this.mapPeople.indexOf(people[st.mapSel]) : -1;
    const actMap = st.mapHov != null ? st.mapHov : selMapIdx;
    const maxMin = Math.max(...this.mapPeople.map(s => s.min));
    const m = st.mapSel != null ? people[st.mapSel] : null;
    let ms = null;
    if (m) {
      const inM = m.themes.filter(j => h.themes[j] && h.themes[j].in_article).length;
      ms = { m, inM, themeNote: !m.themes.length ? 'Nenhum dos temas da audiência aparece nesta fala.' : inM === m.themes.length ? 'Todos os temas desta fala aparecem na matéria.' : inM === 0 ? 'Nenhum tema desta fala aparece na matéria.' : `${inM} de ${m.themes.length} temas desta fala aparecem na matéria.` };
    }

    // summary
    const tau = C.tau;
    const canToggle = hasFix(sum), corrected = st.corrected || !canToggle;
    const lines = sum ? this.buildLines(sum, corrected, tau) : [];
    this.total = lines.reduce((a, l) => a + l.text.length, 0);
    let acc = 0;
    const typed = st.typed;
    const retrieved = new Map();
    if (sum && corrected) sum.events.filter(e => isFix(sum, e)).forEach(e => e.win.forEach(w => { if (!retrieved.has(w)) retrieved.set(w, e.r != null ? e.r + 1 : null); }));

    // fairness bars
    const barsMax = Math.max(...h.speakers.map(s => s.min));
    const shown = people.slice(0, MAX_BARS), more = N - shown.length;
    const O = C.omission;
    const lowest = pct(O.deciles[0]), highest = pct(O.deciles[9]);

    return html`<main class="hearing" data-screen-label=${'Tela 2 · Audiência ' + h.id}>
      <div class="wrap">
        <button class="back" onClick=${() => go('#/')}><${Icon} name="arrow-left" size=${18} />Todas as audiências</button>
        <div class="hhead">
          <div class="eyebrow">Audiência ${h.id} · ${h.date} · ${N} pessoas falaram</div>
          <h1>${h.title}</h1>
          <div class="tags">${h.cats.map(c => html`<button class="tag" onClick=${() => go('#/?tema=' + encodeURIComponent(c))}>${c}</button>`)}</div>
          <p class="intro">${intro}</p>
        </div>

        <div class="wheel-wrap" ref=${this.wrapRef}>
          <div class="wheel-view" style=${{ height: R + (R < 390 ? 175 : 80) + 'px' }}>
            <div class="wheel" ref=${this.wheelRef} style=${{ top: R + 100 + 'px' }}>${tiles}</div>
            <div class="wheel-label">
              <div class="k">Selecione um falante</div>
              <div class="n">${sp.name}</div>
              <div class="r">${sp.role}</div>
              <div class="m">≈ ${sp.min} min de fala · ${citedWord(sp)}${sp.chair ? ' · presidiu a sessão' : ''}</div>
            </div>
          </div>
          ${hs && html`<div class="tip" style=${{ left: st.tipX + 'px', top: st.tipY + 'px' }}>
            <div class="a">${hs.name}</div><div class="b">${hs.role}</div>
            <div class="c">≈ ${hs.min} min de fala · ${citedWord(hs)}${hs.summary ? '' : ' · sem resumo'}</div>
          </div>`}
        </div>

        <div ref=${this.mapStageRef} class=${'map' + (st.mapMax ? ' max' : '')} data-screen-label="Mapa de participação"
          onClick=${() => { if (st.mapSel != null) this.setState({ mapSel: null }); }}
          onMouseMove=${e => { const r = this.mapStage.getBoundingClientRect(); this.mmx = e.clientX - r.left; this.mmy = e.clientY - r.top; }}
          onMouseLeave=${() => { this.mmx = null; this.mmy = null; }}>
          <canvas ref=${this.mapCanvasRef}></canvas>
          <div class="over">
            <div><div class="k10">Mapa de participação</div><div class="hint">Clique em uma pessoa para ver o resumo da participação</div></div>
            <button class="roundbtn" aria-label=${st.mapMax ? 'Reduzir mapa' : 'Expandir mapa'} onClick=${e => { e.stopPropagation(); this.mEW = null; this.setState({ mapMax: !st.mapMax }); }}>
              <${Icon} name=${st.mapMax ? 'minimize-2' : 'maximize-2'} size=${18} />
            </button>
          </div>
          ${this.mapPeople.map((p, i) => {
            const d = Math.round(14 + 34 * Math.sqrt(p.min / maxMin)), on = i === actMap, isSel = i === selMapIdx;
            const pick = e => { e.stopPropagation(); this.select(people.indexOf(p)); };
            return html`<div data-pnode="1" class="pnode">
              <button onClick=${pick} onMouseEnter=${() => this.setState({ mapHov: i })} onMouseLeave=${() => this.setState({ mapHov: null })} aria-label=${p.name}
                style=${{ left: -d / 2 + 'px', top: -d / 2 + 'px', width: d + 'px', height: d + 'px', background: cited(p) ? '#4B68E3' : '#C7CDD3', border: isSel ? '3px solid #fff' : '1px solid rgba(255,255,255,0.9)', boxShadow: on ? '0 0 0 5px rgba(0,136,255,0.2),0 0 32px rgba(0,136,255,0.6)' : '0 2px 10px rgba(40,60,140,0.14)' }}></button>
              <span class="lbl" onClick=${pick} style=${{ top: d / 2 + 8 + 'px', fontWeight: on ? 700 : 500, color: on ? '#0E2740' : '#39434D' }}>${shortName(p.name)}<span>≈ ${p.min} min</span></span>
            </div>`;
          })}
          ${ms && html`<aside class="mpanel" onClick=${e => e.stopPropagation()}>
            <div class="row"><div class="k10">Participação · ${ms.m.rank}º em tempo de fala</div>
              <button class="x" aria-label="Fechar" onClick=${e => { e.stopPropagation(); this.setState({ mapSel: null }); }}><${Icon} name="x" size=${16} /></button></div>
            <h2>${ms.m.name}</h2>
            <div class="role">${ms.m.role}</div>
            <div class="pills">
              <span class="pill">≈ ${ms.m.min} min de fala</span>
              <span class=${'pill' + (cited(ms.m) ? ' cited' : ' line')}>${citedWord(ms.m)}</span>
              ${ms.m.chair && html`<span class="pill soft">presidiu a sessão</span>`}
            </div>
            <div class="quote-k">Trecho da fala</div>
            <p class="contrib">“${ms.m.excerpt}”</p>
            ${ms.m.in_article && html`<div class="quote-k" style=${{ color: '#4B68E3', filter: 'none' }}>Na matéria</div><p class="contrib article">${ms.m.in_article}</p>`}
            ${ms.m.themes.length > 0 && html`<div class="sep">
              <div class="pills" style=${{ marginTop: 0 }}>${ms.m.themes.map(j => h.themes[j] && html`<span class=${'pill' + (h.themes[j].in_article ? ' soft' : ' line')}>${h.themes[j].label}</span>`)}</div>
              <div class="note">${ms.themeNote}</div>
            </div>`}
            ${ms.m.summary && html`<button class="cta" onClick=${() => this.sumEl && this.sumEl.scrollIntoView({ behavior: 'smooth', block: 'start' })}>Ver o resumo da IA <${Icon} name="arrow-right" size=${16} /></button>`}
          </aside>`}
        </div>
        <div class="legend">
          <span><span class="dot" style=${{ background: '#4B68E3' }}></span>Pessoa citada na matéria</span>
          <span><span class="dot" style=${{ background: '#C7CDD3' }}></span>Não citada</span>
          <span><span class="dot" style=${{ background: '#0088FF' }}></span>Tema presente na matéria</span>
          <span><span class="dot" style=${{ background: '#fff', border: '1px solid #B9C7F6' }}></span>Tema que ficou de fora</span>
          <span class="muted">Tamanho do ponto = tempo de fala (estimado: palavras ÷ ${C.wpm} por minuto)</span>
          <span class="muted">Temas extraídos automaticamente da transcrição${this.mapPeople.length < N ? ` · mostrando as ${this.mapPeople.length} pessoas que mais falaram, de ${N}` : ''}</span>
        </div>

        <div class="cols">
          <section class="panel summary" ref=${this.sumRef}>
            <div class="headrow">
              <div>
                <h2>Resumo da participação de <span>${sp.name}</span></h2>
                <div class="role">${sp.role}</div>
                <div class="disc">Resumo gerado por IA (Qwen3-4B). O erro é do modelo, não do participante. ${sum ? (!canToggle ? 'Linhas conferidas pelo nosso sistema.' : corrected ? 'Linhas conferidas pelo nosso sistema, com a intervenção.' : 'Esta é a versão sem a intervenção, como o modelo escreveu.') : ''}</div>
              </div>
              ${canToggle && html`<button class=${'toggle' + (st.corrected ? '' : ' raw')} onClick=${() => { this.restartTyping({ corrected: !st.corrected }); replaceRoute(`#/audiencia/${h.id}/${sp.slug}` + (st.corrected ? '/sem-correcao' : '')); }}>
                ${st.corrected ? 'Ver sem a intervenção' : 'Ver com a intervenção'}</button>`}
            </div>
            ${sum ? html`<div class="lines">
              ${lines.map((l, i) => {
                const start = acc; acc += l.text.length;
                const shownN = Math.max(0, Math.min(l.text.length, typed - start));
                const done = shownN >= l.text.length, started = typed >= start && (shownN > 0 || i === 0);
                if (!started) return null;
                const typingHere = (!done && shownN > 0) || (!done && typed === start && i === 0);
                const hl = !l.ok ? '#FBE4E2' : (corrected && l.fixed ? '#E4F3DF' : 'transparent');
                return html`<div class="sline" key=${(corrected ? 'c' : 'r') + i}>
                  <div class="num">${l.n}</div>
                  <div style=${{ minWidth: 0 }}>
                    <p><span class="hl" style=${{ background: hl }}>${l.text.slice(0, shownN)}</span>${typingHere && html`<span class="caret"></span>`}</p>
                    ${done && html`<div class="vrow">
                      <span class="verdict"><span class="d" style=${{ background: l.ok ? '#6cc070' : '#e5534b' }}></span>${l.verdict}</span>
                      ${l.alarm && html`<span class="alarmchip"><${Icon} name="bell-ring" size=${14} color="#e5534b" />${corrected ? 'o alarme tocou e buscamos a fala' : 'o alarme tocou aqui'}</span>`}
                    </div>
                    ${(l.note || l.comment || l.disagree) && html`<div class="lnote">
                      ${l.note}${l.comment && html`${l.note ? ' ' : ''}Juiz automático: <span class="judge">“${l.comment}”</span>`}${l.disagree && html` ${l.disagree}`}
                    </div>`}`}
                  </div>
                </div>`;
              })}
              <div class="foot-note">O alarme decide quando buscar; a busca na fala corrige. A intervenção reduz um pouco o conteúdo sem apoio na fala, e às vezes a reescrita erra. Vereditos dados por juízes automáticos cegos (Claude Sonnet; Opus como segundo juiz).</div>
              <details class="windows">
                <summary><span class="chev">›</span> As falas de ${firstName(sp.name)}: ${sum.windows.length} trechos de ~100 palavras (o que o modelo e os juízes leram)</summary>
                ${sum.windows.map((w, j) => html`<div class=${'win' + (retrieved.has(j) ? ' got' : '')}>
                  <span class="wk">Trecho ${j + 1}${retrieved.has(j) ? ` · buscado quando o alarme tocou${retrieved.get(j) ? ' na linha ' + retrieved.get(j) : ''}` : ''}</span>${w}</div>`)}
              </details>
            </div>` : html`<div class="nosum">${sp.name} falou cerca de ${sp.min} min nesta audiência, mas não tem resumo gerado. ${citedSentence(sp)}
              <small>O experimento resumiu as pessoas listadas no resumo de referência do dataset (${h.n_named ? 'em geral, as citadas pela matéria' : 'nesta audiência, poucas'}); ${Object.keys(h.summaries).length} das ${N} pessoas desta audiência têm resumo.</small></div>`}
          </section>

          <aside class="panel fair">
            <div class="k">Análises de fairness</div>
            <h3>A matéria comprime e cita quem fala mais</h3>
            <div class="bars">
              ${shown.map((s, i) => html`<div class="bar" onClick=${() => this.select(i)} title=${s.name}>
                <span class="nm" style=${{ color: i === st.sel ? '#0E2740' : '#39434D', fontWeight: i === st.sel ? 600 : 400 }}>${shortName(s.name)}${s.chair ? html`<span class="ch">P</span>` : ''}</span>
                <span class="tr"><i style=${{ width: Math.round(s.min / barsMax * 100) + '%', background: cited(s) ? '#4B68E3' : '#B5BEC7' }}></i></span>
                <span class="v">${s.min} min</span>
              </div>`)}
              ${more > 0 && html`<div class="bar more"><span class="nm">+${more} ${more === 1 ? 'pessoa' : 'pessoas'}</span><span class="tr"></span><span class="v">menos</span></div>`}
            </div>
            <div class="flegend">
              <span><i style=${{ background: '#4B68E3' }}></i>citada na matéria</span>
              <span><i style=${{ background: '#B5BEC7' }}></i>não citada</span>
              ${people.some(s => s.chair) && html`<span><b style=${{ color: '#3450C4', fontSize: '10px' }}>P</b> presidiu a sessão</span>`}
            </div>
            <p>Nas 206 audiências, quem menos fala fica fora da matéria ${lowest} das vezes, e quem mais fala, ${highest}. Com o mesmo tempo de fala, só quem preside a sessão é citado com mais frequência (fica de fora ${pct(O.chair_matched)} das vezes, contra ${pct(O.rest_matched)} dos demais).</p>

            <div class="fsec">
              <div class="t">${firstName(sp.name)} nesta audiência</div>
              <p>Cerca de ${sp.min} min de fala, ${rankLabel(sp)}. ${citedSentence(sp)}${chairNote(sp)}</p>
            </div>
            <div class="fsec">
              <div class="t">E se esta fala fosse atribuída a outra pessoa?</div>
              <p>Mesmo vendo quem falou, os modelos confirmam ${C.audit.misattr_confirmed} das atribuições erradas.</p>
            </div>
            <div class="fsec">
              <div class="t">O cargo muda a crença?</div>
              <p>Em ${C.audit.title_models} modelos, uma atribuição errada é aceita um pouco mais quando o cargo impresso é de parlamentar (${C.audit.title_flips} das decisões). É um resultado do corpus, não desta pessoa.</p>
            </div>
          </aside>
        </div>

        <button class="gamebtn compact" onClick=${() => go('#/jogo?de=' + h.id)}>
          <span class="l"><span class="a">Jogo</span><span class="b">Seja a verdade</span></span>
          <span class="r"><${Icon} name="arrow-right" size=${20} /></span>
        </button>
      </div>
    </main>`;
  }
}
