// Tela 1 · Início — the designer's carousel and particle field, fed with the 100 hearings of the experiment.
import { html, Component, Icon, go, getJSON, p3, gpos, pct1 } from './lib.js';
import { Pipeline } from './pipeline.js';
import { Field } from './field.js';

const FEAT = 'Destaque';   // the chip that shows only the featured hearings (index.json `featured`, set in scripts/export_portal_data.py)
const CHIPS = [FEAT, 'Tributação', 'Saúde', 'Energia', 'Meio ambiente'];   // the four themes are the draft's; all exist in the data
const FRAMES = [21, 38, 57, 74, 90, 112, 133, 146, 158, 171, 185, 199];
const frameOf = id => FRAMES.includes(id) ? id : FRAMES[id % FRAMES.length];
const STEP = 320;
// The game opens on this line (data/game.json, rounds 1-2): of the fixed lines both judges agree on, it is the one
// whose live alarm matches the reading the game can show, with the clearest error (an inverted tax comparison).
const SHOWCASE = { h: 38, slug: 'daniel-panizzi', line: 1 };

export class Home extends Component {
  state = { q: this.props.q?.q || '', cat: this.props.q?.tema || null };
  offset = 0; nudge = 0; paused = false;

  componentDidMount() {
    getJSON('data/h/38.json').then(h => {   // the showcase's real alarm, for the button's gauge
      const s = h.summaries[SHOWCASE.slug], e = s && s.events.find(x => x.b === SHOWCASE.line);
      if (e) this.setState({ show: { p: e.p } });
    }, () => {});
    let last = performance.now();
    const loop = t => { const dt = Math.min(0.05, (t - last) / 1000); last = t; this.tick(dt); this.raf = requestAnimationFrame(loop); };
    this.raf = requestAnimationFrame(loop);
    if (this.props.about) requestAnimationFrame(() => this.toAbout());
  }
  componentDidUpdate(prev) {
    if (this.props.about && !prev.about) this.toAbout();
    else if (!this.props.about && prev.about) window.scrollTo({ top: 0, behavior: 'smooth' });   // back to #/ from the section
  }
  toAbout() { const el = document.getElementById('como-funciona'); if (el) el.scrollIntoView({ behavior: 'smooth', block: 'start' }); }
  componentWillUnmount() { cancelAnimationFrame(this.raf); if (this.stage) this.stage.removeEventListener('wheel', this.onWheel); }

  stageRef = el => {
    if (this.stage && this.onWheel) this.stage.removeEventListener('wheel', this.onWheel);
    this.stage = el;
    if (!el) return;
    this.onWheel = e => {
      const dx = Math.abs(e.deltaX) > Math.abs(e.deltaY) ? e.deltaX : (e.shiftKey ? e.deltaY : 0);
      if (!dx || this.staticMode) return;
      e.preventDefault();
      this.offset += dx * (e.deltaMode === 1 ? 16 : 1);
    };
    el.addEventListener('wheel', this.onWheel, { passive: false });
  };

  tick(dt) {
    const st = this.stage; if (!st) return;
    const cards = st.querySelectorAll('[data-card]'), N = cards.length;
    if (!N) return;
    if (!this.paused) this.offset += 16 * dt;
    if (this.nudge) { const d = this.nudge * Math.min(1, dt * 6); this.offset += d; this.nudge -= d; if (Math.abs(this.nudge) < 0.5) { this.offset += this.nudge; this.nudge = 0; } }
    const total = N * STEP, o = ((this.offset % total) + total) % total;
    const W = cards[0].offsetWidth || 340, GAP = 14, P = 1300;
    const items = Array.from(cards).map((c, i) => {
      let x = i * STEP - o;
      x = ((x + total / 2) % total + total) % total - total / 2;
      const u = this.staticMode ? i - (N - 1) / 2 : x / STEP, au = Math.min(Math.abs(u), 3);
      const rot = Math.max(-3, Math.min(3, u)) * 10, s = 1 - 0.12 * au;
      return { c, i, u, s, rot };
    }).sort((p, q) => p.u - q.u);
    items.forEach(it => { const hw = W * it.s / 2, th = it.rot * Math.PI / 180; it.a = hw * Math.cos(th); it.b = hw * Math.sin(th); });
    const Lof = it => (it.px - it.a) * P / (P - it.b), Rof = it => (it.px + it.a) * P / (P + it.b);
    const placeRight = (prev, it) => { it.px = (Rof(prev) + GAP) * (P - it.b) / P + it.a; };
    const placeLeft = (next, it) => { it.px = (Lof(next) - GAP) * (P + it.b) / P - it.a; };
    let ai = 0;
    items.forEach((it, k) => { if (it.u <= 0) ai = k; });
    const A = items[ai], B = items[ai + 1];
    A.px = 0;
    if (B) { placeRight(A, B); A.px = A.u * B.px; placeRight(A, B); }
    for (let k = ai - 1; k >= 0; k--) placeLeft(items[k + 1], items[k]);
    for (let k = ai + 2; k < items.length; k++) placeRight(items[k - 1], items[k]);
    const half = st.clientWidth / 2 + W;
    items.forEach(it => {
      const vis = Math.abs(it.px) < half;
      it.c.style.visibility = vis ? 'visible' : 'hidden';
      if (!vis) return;
      it.c.style.transform = `translate(-50%,-50%) translateX(${it.px.toFixed(1)}px) rotateY(${it.rot.toFixed(2)}deg) scale(${it.s.toFixed(3)})`;
      it.c.style.zIndex = String(100 - Math.round(Math.abs(it.u) * 10));
    });
  }


  render({ index }, { q, cat, show }) {
    const C = index.corpus, O = C.omission;
    const qq = q.trim().toLowerCase();
    const filtered = index.hearings.filter(h => (!cat || (cat === FEAT ? h.featured != null : h.cats.includes(cat))) &&
      (!qq || [h.title, h.headline, h.date, h.cats.join(' '), h.people.join(' '), 'audiência ' + h.id].join(' ').toLowerCase().includes(qq)));
    const active = !!(qq || cat);
    this.staticMode = active && filtered.length <= 6;
    let list = filtered.slice();
    if (list.length && !this.staticMode) while (list.length < 9) list = list.concat(filtered);
    const chips = (cat && !CHIPS.includes(cat) ? CHIPS.concat([cat]) : CHIPS);
    const label = !active || !filtered.length ? '' : filtered.length === 1 ? '1 audiência encontrada' : filtered.length + ' audiências encontradas';
    const reset = patch => { this.offset = 0; this.setState(patch); };
    return html`<main class="home" data-screen-label="Tela 1 · Início">
      <${Field} />
      <div class="hero">
        <div class="eyebrow">Audiências públicas da Câmara</div>
        <h1>Detectar, intervir, auditar</h1>
        <p class="tagline">Trazendo interpretabilidade e reduzindo a alucinação da IA no debate público</p>
        <p class="lead">Escolha uma audiência e uma pessoa para ler o resumo da fala gerado por IA. Cada linha é conferida contra a transcrição oficial e, onde intervimos, você pode ver a versão sem a intervenção.</p>
        <div class="search-block">
          <label class="search">
            <${Icon} name="search" size=${18} color="#5C6771" />
            <input value=${q} onInput=${e => reset({ q: e.target.value })} placeholder="Pesquise um tema, uma pessoa ou uma data" aria-label="Pesquisar audiências" />
            ${q && html`<button class="clear" onClick=${() => reset({ q: '' })}>Limpar</button>`}
            <span class="go" onClick=${() => filtered[0] && go('#/audiencia/' + filtered[0].id)} title="Abrir a primeira audiência"><${Icon} name="arrow-right" size=${18} /></span>
          </label>
          <div class="chips">
            ${chips.map(c => html`<button class=${'chip' + (cat === c ? ' on' : '')} onClick=${() => reset({ cat: cat === c ? null : c })}>${c}</button>`)}
          </div>
          <div class="result-label">${label}</div>
        </div>
      </div>
      <div class="stage" ref=${this.stageRef} onMouseEnter=${() => { this.paused = true; }} onMouseLeave=${() => { this.paused = false; }}>
        ${list.map((h, k) => html`<div key=${h.id + '-' + k} data-card="1" class=${'hcard' + (h.featured != null ? ' feat' : '')} role="button" tabindex="0" aria-label=${'Abrir audiência ' + h.id + (h.featured != null ? ' (destaque)' : '')}
            onClick=${() => go('#/audiencia/' + h.id)} onKeyDown=${e => e.key === 'Enter' && go('#/audiencia/' + h.id)}>
          <div class="body">
            <div class="meta"><span>${h.n} pessoas falaram</span><span>${h.date}</span></div>
            <h3>${h.title}</h3>
            <div class="cats">${h.cats.join(' · ')}</div>
          </div>
          <div class="frame"><img src=${'img/frames/frame-' + frameOf(h.id) + '.png'} alt="" loading="lazy" />${h.featured != null && html`<span class="ftag">Destaque</span>`}</div>
        </div>`)}
        ${!filtered.length && html`<div class="empty">Nenhuma audiência encontrada. Tente outro tema, nome ou data.</div>`}
      </div>
      <div class="home-game">
        <button class="gamebtn compact" onClick=${() => go('#/jogo?de=inicio')}>
          <span class="l"><span class="a">Jogo</span><span class="b">Seja a verdade</span></span>
          ${show && html`<span class="gm" aria-hidden="true">
            <span class="gm-k">Leitura do probe</span>
            <span class="gm-bar"><i style=${{ width: (gpos(show.p) * 100).toFixed(1) + '%' }}></i><b style=${{ left: (gpos(index.corpus.tau) * 100).toFixed(1) + '%' }}></b></span>
            <span class="gm-v"><em>${p3(show.p)}</em> ≥ ${p3(index.corpus.tau)} · alarme</span>
          </span>`}
          <span class="r"><${Icon} name="arrow-right" size=${20} /></span>
        </button>
      </div>

      <section class="about" id="como-funciona" aria-labelledby="about-h">
        <div class="about-head">
          <div class="eyebrow">O que é esta página</div>
          <h2 id="about-h">Uma consulta aberta ao nosso método: detectar, intervir e auditar resumos feitos por IA</h2>
          <p>Resumir uma audiência pública com IA economiza horas, mas o modelo às vezes escreve o que a pessoa não disse. Aqui você confere o resultado do nosso pipeline em ${index.hearings.length} audiências reais da Câmara: o resumo da fala de cada pessoa, linha por linha. Tudo com modelos abertos.</p>
        </div>
        <${Pipeline}>
          <div class="pstep">
            <div class="pviz"></div>
            <div class="pk"><span>01</span>Detectar</div>
            <h3>Um detector lê o modelo por dentro</h3>
            <p>Enquanto o modelo escreve, um probe lê as ativações internas dele e toca um alarme um token antes de uma palavra que não está na fala.</p>
            <div class="pstat"><b>0,905</b><span>AUC do alarme um token antes, no Qwen3-4B</span></div>
          </div>
          <div class="pstep">
            <div class="pviz"></div>
            <div class="pk"><span>02</span>Intervir</div>
            <h3>A busca na própria fala reescreve a linha</h3>
            <p>Quando o alarme toca, buscamos os trechos da fala da própria pessoa e reescrevemos a linha. O alarme decide quando buscar; a busca na fala corrige.</p>
            <div class="pstat"><b>${pct1(C.rag.with_unsupported_baseline)} → ${pct1(C.rag.with_unsupported_rag)}</b><span>das pessoas com alguma linha sem apoio na fala, segundo juízes automáticos cegos</span></div>
          </div>
          <div class="pstep">
            <div class="pviz"></div>
            <div class="pk"><span>03</span>Auditar · fairness</div>
            <h3>Quem fica de fora e em quem a IA acredita</h3>
            <p>Medimos quem a matéria publicada deixa de citar e se os modelos acreditam mais em quem tem cargo.</p>
            <div class="pstat"><b>${pct1(O.deciles[0])} → ${pct1(O.deciles[O.deciles.length - 1])}</b><span>ficam fora da matéria: os 10% que menos falam e os 10% que mais falam</span></div>
          </div>
        <//>
        <div class="howto">
          <div class="howto-k">Como consultar</div>
          <ol>
            <li><b>1</b><span>Escolha uma audiência no carrossel ou pela busca.</span></li>
            <li><b>2</b><span>Escolha uma pessoa no mapa da audiência.</span></li>
            <li><b>3</b><span>Leia cada linha com o veredito e, onde houve intervenção, o alarme e a versão sem ela.</span></li>
          </ol>
          <button class="about-cta" onClick=${() => go('#/resultados')}>Ver os resultados do paper <${Icon} name="arrow-right" size=${16} /></button>
        </div>
        <p class="about-foot">Equipe Ocarandu (PUCRS e UFMG) no desafio Ideias em Rede do Instituto Kunumi, tema IA e a Esfera Pública. Dados: PublicHearingBR (Unicamp), com as transcrições oficiais e as matérias da Agência Câmara.</p>
      </section>
    </main>`;
  }
}
