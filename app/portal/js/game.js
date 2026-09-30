// Tela 4 · Seja o probe. A replay of real runs (data/game.json, scripts/export_portal_data.py): the model
// writes a real summary line word by word; before each content word the player sees the reading of the token
// probe that actually ran (Qwen3-4B, layer 21) and the 2,560 per-dimension contributions to its logit drawn as
// ASCII. When the reading crosses the threshold (0.978), the alarm rings: SPACE stops the model, retrieves the
// speaker's own passages (the windows the real run retrieved) and shows the real rewritten line. Whatever the
// player does, every line and every verdict on screen is one that exists in the runs and the blind rounds.
// The alarm's special effect (the field breaking out, the probe's plane, the fall) lives in swarm.js.
import { html, Component, Icon, go, getJSON, p3, pct1, gpos } from './lib.js';
import { Swarm } from './swarm.js';

const COLS = 64, ROWS = 40;
const RAMP = ' ·:-=+*#%@';
const C = { speech: '#4fb3a9', model: '#e0b04a', alarm: '#e5534b', ok: '#6cc070', ink: '#d9e2ea', dim: '#5d6b78' };
const rgb = { speech: [79, 179, 169], alarm: [229, 83, 75], model: [224, 176, 74], ink: [185, 198, 210] };
const fold = s => s.normalize('NFD').replace(/[̀-ͯ]/g, '').toLowerCase();
const STOPW = new Set('para pela pelo pelos pelas como mais que com uma umas uns dos das nos nas num numa sobre entre ele ela eles elas seu sua seus suas este esta esse essa isso isto muito muita ainda tambem pois quando onde deve devem pode podem sao ser foi tem sendo esta estao pelo cada todo toda todos todas outro outra outros outras mesmo mesma'.split(' '));
const cwords = s => (fold(s).match(/[a-z0-9]+/g) || []).filter(w => w.length > 3 && !STOPW.has(w));
const stem = w => w.slice(0, 6);
const esc = s => s.replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const clamp = (x, a, b) => Math.max(a, Math.min(b, x));


function sentences(t) { return t.split(/(?<=[.!?…])\s+/).map(s => s.trim()).filter(s => s.length > 20); }
function bestSentences(passages, line, k = 1) {
  const L = new Set(cwords(line).map(stem));
  const all = [];
  passages.forEach((p, pi) => sentences(p).forEach(s => {
    const W = cwords(s), hit = W.filter(w => L.has(stem(w))).length;
    all.push({ s, pi, score: hit / Math.sqrt(W.length + 4) });
  }));
  return all.sort((a, b) => b.score - a.score).slice(0, k);
}
function markWords(s, ref) {
  const R = new Set(cwords(ref).map(stem));
  return s.split(/(\s+)/).map(t => {
    const w = cwords(t)[0];
    return w && R.has(stem(w)) ? `<mark>${esc(t)}</mark>` : esc(t);
  }).join('');
}

// ---------------------------------------------------------------- sound (WebAudio, synthesized, muted with M)
class Sfx {
  on = true;
  ac() { if (!this._ac) { try { this._ac = new (window.AudioContext || window.webkitAudioContext)(); } catch (e) { this.on = false; } } return this._ac; }
  tone(f, d, type = 'square', v = 0.045, at = 0, f2 = null) {
    if (!this.on) return; const ac = this.ac(); if (!ac) return;
    const t = ac.currentTime + at, o = ac.createOscillator(), g = ac.createGain();
    o.type = type; o.frequency.setValueAtTime(f, t); if (f2) o.frequency.exponentialRampToValueAtTime(f2, t + d);
    g.gain.setValueAtTime(0.0001, t); g.gain.exponentialRampToValueAtTime(v, t + 0.012); g.gain.exponentialRampToValueAtTime(0.0001, t + d);
    o.connect(g).connect(ac.destination); o.start(t); o.stop(t + d + 0.02);
  }
  alarm() { this.tone(880, 0.13, 'square', 0.04, 0); this.tone(660, 0.13, 'square', 0.04, 0.17); this.tone(880, 0.13, 'square', 0.04, 0.38); this.tone(660, 0.13, 'square', 0.04, 0.55); }
  press() { this.tone(220, 0.08, 'triangle', 0.08, 0); }
  search() { this.tone(300, 0.55, 'sine', 0.05, 0, 1200); this.tone(450, 0.55, 'sine', 0.03, 0.08, 1500); }
  good() { this.tone(660, 0.16, 'sine', 0.06, 0); this.tone(990, 0.24, 'sine', 0.06, 0.14); }
  bad() { this.tone(140, 0.42, 'sawtooth', 0.05, 0, 90); }
  early() { this.tone(1600, 0.04, 'square', 0.02, 0); }
  read() { this.tone(2400, 0.012, 'sine', 0.012, 0); }
  noiseSrc(ac, d) {
    const n = Math.ceil(ac.sampleRate * d), buf = ac.createBuffer(1, n, ac.sampleRate), ch = buf.getChannelData(0);
    for (let i = 0; i < n; i++) ch[i] = Math.random() * 2 - 1;
    const src = ac.createBufferSource(); src.buffer = buf; return src;
  }
  noise(d, v, type, f0, f1, q = 0.8) {
    if (!this.on) return; const ac = this.ac(); if (!ac) return;
    const t = ac.currentTime, src = this.noiseSrc(ac, d), f = ac.createBiquadFilter(), g = ac.createGain();
    f.type = type; f.Q.value = q; f.frequency.setValueAtTime(f0, t); f.frequency.exponentialRampToValueAtTime(f1, t + d);
    g.gain.setValueAtTime(0.0001, t); g.gain.exponentialRampToValueAtTime(v, t + 0.01); g.gain.exponentialRampToValueAtTime(0.0001, t + d);
    src.connect(f).connect(g).connect(ac.destination); src.start(t); src.stop(t + d + 0.02);
  }
  heat(d) {   // the takeover: a hiss that rises for the whole alarm window
    this.stopHeat();
    if (!this.on) return; const ac = this.ac(); if (!ac) return;
    const t = ac.currentTime, src = this.noiseSrc(ac, d + 0.6), f = ac.createBiquadFilter(), g = ac.createGain();
    f.type = 'bandpass'; f.Q.value = 4; f.frequency.setValueAtTime(260, t); f.frequency.exponentialRampToValueAtTime(2600, t + d);
    g.gain.setValueAtTime(0.0001, t); g.gain.linearRampToValueAtTime(0.035, t + d);
    src.connect(f).connect(g).connect(ac.destination); src.start(t); src.stop(t + d + 0.5);
    this._heat = { src, g };
  }
  stopHeat() {
    const h = this._heat; if (!h) return; this._heat = null;
    try { const t = this._ac.currentTime; h.g.gain.cancelScheduledValues(t); h.g.gain.setTargetAtTime(0.0001, t, 0.03); h.src.stop(t + 0.25); } catch (e) { /* already stopped */ }
  }
  slash() { this.noise(0.2, 0.12, 'highpass', 6000, 900); this.tone(2600, 0.22, 'sawtooth', 0.022, 0, 280); }
  lock() { this.tone(523, 0.6, 'sine', 0.035, 0); this.tone(784, 0.7, 'sine', 0.028, 0.06); this.tone(1047, 0.8, 'sine', 0.02, 0.12); }
  fall() { this.tone(480, 1.0, 'sawtooth', 0.03, 0, 45); this.tone(70, 0.35, 'sine', 0.07, 0.9); }
}

// ---------------------------------------------------------------- ASCII art: the Congresso Nacional (title screen)
function drawCongress(cv, t, rising) {
  const r = cv.getBoundingClientRect(), dpr = window.devicePixelRatio || 1, W = r.width, H = r.height;
  if (!W || !H) return;
  if (cv.width !== Math.round(W * dpr)) { cv.width = Math.round(W * dpr); cv.height = Math.round(H * dpr); cv._art = null; }
  const x = cv.getContext('2d'); x.setTransform(dpr, 0, 0, dpr, 0, 0);
  const cols = 230, cw = W / cols, ch = cw * 1.85, rows = Math.ceil(H / ch);
  const hs = H * 0.64, ws = hs * 16 / 9, x0 = (W - ws) / 2, y0 = H - hs;           // the scene's box
  const wX = i => ((i + 0.5) * cw - x0) / ws * 16, wY = j => ((j + 0.5) * ch - y0) / hs * 9;
  if (!cv._art) {  // the static building, drawn once per size into an offscreen canvas
    const off = document.createElement('canvas'); off.width = cv.width; off.height = cv.height;
    const o = off.getContext('2d'); o.setTransform(dpr, 0, 0, dpr, 0, 0);
    o.font = `${(ch * 0.92).toFixed(1)}px "IBM Plex Mono", monospace`; o.textAlign = 'center'; o.textBaseline = 'middle';
    for (let j = 0; j < rows; j++) for (let i = 0; i < cols; i++) {   // world units: x in [0,16], y in [0,9]
      const X = wX(i), Y = wY(j);
      if (X < 0 || X > 16 || Y < 0 || Y > 9) continue;
      let c = null, a = 0, col = rgb.ink;
      const base = 5.25;
      // Senate dome (left, convex), lit from the upper left
      const dx = (X - 5.0) / 1.75, dy = (Y - base) / 1.3;
      if (Y < base && dx * dx + dy * dy < 1) { const nz = Math.sqrt(Math.max(0, 1 - dx * dx - dy * dy)); const sh = clamp(0.2 + 0.8 * (-dx * 0.45 - dy * 0.35 + nz * 0.55), 0, 1); c = RAMP[1 + Math.round(sh * 8)]; a = 0.3 + 0.6 * sh; }
      // Chamber bowl (right, concave): narrow at the slab, widening to the rim
      const bt = 3.75;
      if (Y >= bt && Y < base) { const u = (base - Y) / (base - bt), hw = 0.5 + 1.9 * Math.pow(u, 0.6), ddx = (X - 11.0) / hw; if (Math.abs(ddx) < 1) { const rim = Y < bt + 0.12; const sh = clamp(0.25 + 0.75 * (1 - Math.abs(ddx + 0.3)) * (0.55 + 0.45 * u), 0, 1); c = rim ? '▁' : RAMP[1 + Math.round(sh * 8)]; a = rim ? 0.95 : 0.28 + 0.6 * sh; col = rim ? rgb.speech : rgb.ink; } }
      // twin towers, 28 floors, joined by a bridge
      const inT1 = X > 7.6 && X < 7.93, inT2 = X > 8.07 && X < 8.4;
      if ((inT1 || inT2) && Y > 0.7 && Y < base) { const floor = Math.floor((Y - 0.7) / 0.16) % 2; c = floor ? '▓' : '▒'; a = inT1 ? 0.7 : 0.5; }
      if (X >= 7.93 && X <= 8.07 && Y > 2.5 && Y < 2.95) { c = '▒'; a = 0.55; }
      // slab and the building under it
      if (Y >= base && Y < base + 0.18 && X > 1.6 && X < 14.4) { c = '▀'; a = 0.9; }
      if (Y >= base + 0.18 && Y < 6.05 && X > 2.3 && X < 13.7) { c = (i % 4 === 0) ? '│' : '·'; a = (i % 4 === 0) ? 0.42 : 0.14; }
      if (Y >= 6.05 && Y < 6.2) { c = '▔'; a = 0.25; }
      if (c) { o.fillStyle = `rgba(${col[0]},${col[1]},${col[2]},${a.toFixed(2)})`; o.fillText(c, (i + 0.5) * cw, (j + 0.5) * ch); }
    }
    cv._art = off;
  }
  x.clearRect(0, 0, W, H);
  // sky: slow twinkle
  x.font = `${(ch * 0.9).toFixed(1)}px "IBM Plex Mono", monospace`; x.textAlign = 'center'; x.textBaseline = 'middle';
  for (let k = 0; k < 70; k++) {
    const i = (k * 67) % cols, j = (k * 29) % Math.floor(rows * 0.42);
    const a = 0.08 + 0.12 * (0.5 + 0.5 * Math.sin(t * 0.8 + k));
    x.fillStyle = `rgba(185,198,210,${a.toFixed(3)})`; x.fillText('·', (i + 0.5) * cw, (j + 0.5) * ch);
  }
  x.drawImage(cv._art, 0, 0, W, H);
  // the reflecting pool: shimmering reflection of the towers
  for (let j = 0; j < rows; j++) for (let i = 0; i < cols; i++) {
    const X = wX(i), Y = wY(j);
    if (Y < 6.3 || X < -1 || X > 17) continue;
    const wob = Math.sin(t * 1.6 + Y * 7 + X * 0.8) * 0.06;
    const tower = (X + wob > 7.6 && X + wob < 7.93) || (X + wob > 8.07 && X + wob < 8.4);
    const d = (Y - 6.3) / 2.7;
    if (tower && Y < 8.6) { x.fillStyle = `rgba(185,198,210,${(0.3 * (1 - d)).toFixed(3)})`; x.fillText('|', (i + 0.5) * cw, (j + 0.5) * ch); }
    else if ((i + j * 3 + Math.floor(t * 2)) % 11 === 0) { x.fillStyle = `rgba(79,179,169,${(0.1 + 0.08 * (1 - d)).toFixed(3)})`; x.fillText('~', (i + 0.5) * cw, (j + 0.5) * ch); }
  }
  // words of the speech rising from the Chamber's bowl; some turn amber (the model takes them)
  x.textAlign = 'left';
  rising.forEach(w => {
    const px = x0 + w.x * ws, py = y0 + w.y * hs;
    const a = clamp(Math.min(w.life, 1 - w.life) * 3, 0, 1) * 0.85;
    const cc = w.model ? rgb.model : rgb.speech;
    x.font = `${w.size.toFixed(1)}px "Work Sans", sans-serif`;
    x.fillStyle = `rgba(${cc[0]},${cc[1]},${cc[2]},${a.toFixed(3)})`; x.fillText(w.t, px, py);
  });
}

// ---------------------------------------------------------------- ASCII art: the plenary (left panel)
function drawPlenary(cv, t, glow, beam) {
  const r = cv.getBoundingClientRect(), dpr = window.devicePixelRatio || 1, W = r.width, H = r.height;
  if (!W || !H) return;
  if (cv.width !== Math.round(W * dpr)) { cv.width = Math.round(W * dpr); cv.height = Math.round(H * dpr); }
  const x = cv.getContext('2d'); x.setTransform(dpr, 0, 0, dpr, 0, 0); x.clearRect(0, 0, W, H);
  const cx = W / 2, cy = H * 0.9, fs = Math.max(7, H * 0.085), sx = Math.min(1.75, (W * 0.48) / (H * 0.86));
  x.textAlign = 'center'; x.textBaseline = 'middle';
  x.font = `${fs.toFixed(1)}px "IBM Plex Mono", monospace`;
  for (let k = 0; k < 7; k++) {   // seven rows of seats, the Chamber's fan, all facing the table
    const R = H * (0.3 + k * 0.095), n = 12 + k * 5;
    for (let s = 0; s <= n; s++) {
      const an = Math.PI + 0.12 + (s / n) * (Math.PI - 0.24), px = cx + Math.cos(an) * R * sx, py = cy + Math.sin(an) * R;
      const aisle = s % Math.max(4, Math.round(n / 5)) === 0 && s > 0 && s < n;
      if (aisle) continue;
      const a = 0.22 + 0.1 * Math.sin(t * 0.9 + s * 0.7 + k * 1.3) + (k < 2 ? 0.1 : 0);
      x.fillStyle = `rgba(150,165,180,${a.toFixed(3)})`; x.fillText('▪', px, py);
    }
  }
  x.fillStyle = 'rgba(185,198,210,0.55)'; x.font = `${(fs * 1.05).toFixed(1)}px "IBM Plex Mono", monospace`;
  x.fillText('▄▄▄▄▄▄▄▄▄', cx, cy + fs * 0.1);
  const px = cx + fs * 1.6, py = cy - fs * 0.9, pulse = 0.5 + 0.5 * Math.sin(t * 4);
  x.shadowColor = 'rgba(79,179,169,0.95)'; x.shadowBlur = 6 + 14 * glow * pulse;
  x.fillStyle = C.speech; x.font = `${(fs * 1.35).toFixed(1)}px "IBM Plex Mono", monospace`; x.fillText('◆', px, py); x.shadowBlur = 0;
  x.font = `600 ${(fs * 0.78).toFixed(1)}px "Plus Jakarta Sans", sans-serif`; x.fillStyle = 'rgba(79,179,169,0.85)'; x.textAlign = 'left';
  x.fillText('fala', px + fs * 0.9, py + 1);
  x.textAlign = 'center';
  if (beam > 0) {   // retrieval: characters stream from the speaker down to the passages
    x.font = `${fs.toFixed(1)}px "IBM Plex Mono", monospace`;
    for (let k = 0; k < 22; k++) {
      const u = ((t * 1.7 + k / 22) % 1), yy = py + u * (H - py + fs), xx = px + Math.sin(k * 2.1 + t * 3) * fs * 0.8 - u * fs * 2;
      x.fillStyle = `rgba(79,179,169,${(beam * (1 - u) * 0.95).toFixed(2)})`; x.fillText('01·:▪'[k % 5], xx, yy);
    }
  }
}

// the readings of the current line, one bar per word the probe read, with the word under it
function drawStrip(cv, spark, tau) {
  const r = cv.getBoundingClientRect(), dpr = window.devicePixelRatio || 1, W = r.width, H = r.height;
  if (!W || !H) return;
  if (cv.width !== Math.round(W * dpr)) { cv.width = Math.round(W * dpr); cv.height = Math.round(H * dpr); }
  const x = cv.getContext('2d'); x.setTransform(dpr, 0, 0, dpr, 0, 0); x.clearRect(0, 0, W, H);
  const lab = Math.max(9, H * 0.2), top = 4, base = H - lab - 6, bh = base - top;
  const ty = base - gpos(tau) * bh;
  x.strokeStyle = 'rgba(229,83,75,0.55)'; x.setLineDash([3, 4]); x.beginPath(); x.moveTo(0, ty); x.lineTo(W, ty); x.stroke(); x.setLineDash([]);
  x.strokeStyle = '#26313b'; x.beginPath(); x.moveTo(0, base + 0.5); x.lineTo(W, base + 0.5); x.stroke();
  const slot = Math.min(W / Math.max(spark.length, 8), W / 8), bw = Math.min(slot * 0.34, 14);
  x.font = `${(lab * 0.78).toFixed(1)}px "IBM Plex Mono", monospace`; x.textAlign = 'center'; x.textBaseline = 'top';
  spark.forEach((s, i) => {
    const cxx = slot * (i + 0.5), hgt = Math.max(2, gpos(s.p) * bh), over = s.p >= tau;
    x.fillStyle = s.hot ? '#e5534b' : over ? 'rgba(229,83,75,0.55)' : '#3d4a56';
    if (s.hot) { x.shadowColor = 'rgba(229,83,75,0.8)'; x.shadowBlur = 10; }
    x.fillRect(cxx - bw / 2, base - hgt, bw, hgt); x.shadowBlur = 0;
    let w = (s.w || '').replace(/[.,;:!?]+$/, '');
    const maxw = slot * 0.96;
    while (w.length > 2 && x.measureText(w).width > maxw) w = w.slice(0, -2) + '…';
    x.fillStyle = s.hot ? '#e5534b' : '#6f7c88'; x.fillText(w, cxx, base + 4);
  });
}

// ---------------------------------------------------------------- ASCII art: the probe's view (right panel)
function drawField(cv, vals, mode, t) {
  const r = cv.getBoundingClientRect(), dpr = window.devicePixelRatio || 1, W = r.width, H = r.height;
  if (!W || !H) return;
  if (cv.width !== Math.round(W * dpr)) { cv.width = Math.round(W * dpr); cv.height = Math.round(H * dpr); }
  const x = cv.getContext('2d'); x.setTransform(dpr, 0, 0, dpr, 0, 0); x.clearRect(0, 0, W, H);
  const cw = W / COLS, ch = H / ROWS;
  x.font = `${(ch * 1.02).toFixed(1)}px "IBM Plex Mono", monospace`; x.textAlign = 'center'; x.textBaseline = 'middle';
  const buckets = new Map();
  for (let rr = 0; rr < ROWS; rr++) for (let cc = 0; cc < COLS; cc++) {
    const k = rr * COLS + cc;
    let v = vals ? vals[k] : 0, a = Math.min(1, Math.pow(Math.abs(v) * 127 / 52, 0.7));
    let ch0, key;
    if (mode === 'none' || !vals) {  // no stored reading here: the residual stream is there, the probe is not reading it
      const n = (Math.sin(k * 12.9898 + Math.floor(t * 6) * 0.37) * 43758.5453) % 1;
      if (Math.abs(n) > 0.18) continue;
      ch0 = '·'; key = 'n2';
    } else if (a < 0.14) { if ((k + rr) % 2) continue; ch0 = '·'; key = 'n1'; }
    else {
      const lev = Math.min(RAMP.length - 1, 1 + Math.floor(a * (RAMP.length - 1)));
      ch0 = RAMP[lev];
      const alpha = Math.min(5, Math.floor((0.3 + 0.7 * a) * 5.99));
      key = (v > 0 ? 'a' : 's') + alpha + (mode === 'hot' && v > 0 ? 'h' : '');
    }
    if (!buckets.has(key)) buckets.set(key, []);
    buckets.get(key).push([ch0, (cc + 0.5) * cw, (rr + 0.5) * ch]);
  }
  buckets.forEach((cells, key) => {
    if (key === 'n1') x.fillStyle = 'rgba(120,135,150,0.16)';
    else if (key === 'n2') x.fillStyle = 'rgba(120,135,150,0.12)';
    else {
      const c = key[0] === 'a' ? rgb.alarm : rgb.speech, al = (+key[1] + 1) / 6;
      x.fillStyle = `rgba(${c[0]},${c[1]},${c[2]},${al.toFixed(2)})`;
      if (key.endsWith('h')) { x.shadowColor = 'rgba(229,83,75,0.8)'; x.shadowBlur = 6; } else x.shadowBlur = 0;
    }
    cells.forEach(([cch, px, py]) => x.fillText(cch, px, py));
  });
  x.shadowBlur = 0;
}

// ---------------------------------------------------------------- the game
const SCORE = { hit: 100, bonus: 50, miss: -50, early: -25 };
const TIPS = [
  'Aquecimento. Observe o probe: antes de cada palavra de conteúdo ele lê o estado do modelo e dá a chance de a próxima palavra não estar na fala. Se ele não passar do limiar, não aperte nada.',
  'Fique de olho no limiar. Quando o alarme tocar, as ativações escapam: aperte ESPAÇO antes que elas caiam, ou o modelo escreve a próxima palavra sem buscar na fala.',
  'O alarme toca em cerca de um terço das linhas. Nem sempre a linha estava errada.',
  'Última linha.',
];

export class GameScreen extends Component {
  state = { data: null, phase: 'title', ri: 0, score: 0, results: [], msg: null, stamps: [null, null], shout: null, found: null, sound: true, pressed: false };
  keys = e => this.onKey(e);

  componentDidMount() {
    this.auto = this.props.q && this.props.q.auto === '1';
    this.load();
    this.sfx = new Sfx();
    window.addEventListener('keydown', this.keys);
    this.rising = [];
    let last = performance.now();
    const loop = t => { const dt = Math.min(0.05, (t - last) / 1000); last = t; this.tick(dt, t / 1000); this.raf = requestAnimationFrame(loop); };
    this.raf = requestAnimationFrame(loop);
  }
  componentWillUnmount() {
    window.removeEventListener('keydown', this.keys);
    cancelAnimationFrame(this.raf); (this.timers || []).forEach(clearTimeout);
    if (this.sfx) this.sfx.stopHeat();
  }
  // the game's data; until it arrives the title says so, and if it fails SPACE tries again
  load() {
    this.setState({ loadErr: null });
    getJSON('data/game.json').then(data => {
      this.data = data;
      this.rounds = data.rounds.map(r => this.prepRound(r));
      this.swarm = data.plane ? new Swarm(data.plane, COLS, ROWS) : null;
      const start = this.props.q && +this.props.q.rodada;
      this.setState({ data, ri: start ? clamp(start - 1, 0, this.rounds.length - 1) : 0 });
      if (this.auto) this.later(1600, () => this.advance());
    }, err => { const m = String((err && err.message) || err); this.setState({ loadErr: /failed to fetch|networkerror/i.test(m) ? 'falha de rede' : m }); });
  }
  later(ms, f) { (this.timers = this.timers || []).push(setTimeout(f, ms)); }

  prepRound(r) {
    const dims = r.dims, bin = atob(r.field), q = new Int8Array(bin.length);
    for (let i = 0; i < bin.length; i++) q[i] = (bin.charCodeAt(i) << 24) >> 24;
    const names = new Set(r.name_tokens);
    const words = r.words.map((w, k) => ({ ...w, k, eligible: w.content && !names.has(fold(w.w).replace(/[^a-z0-9]/g, '')) }));
    const field = k => { const v = new Float32Array(dims); for (let i = 0; i < dims; i++) v[i] = q[k * dims + i] / 127; return v; };
    return { ...r, words, field };
  }

  // ---------------------------------------------------------------- flow
  advance() {
    const { phase, ri } = this.state;
    if (!this.data) return;
    if (phase === 'title') { this.sfx.ac(); this.setState({ phase: 'intro', ri: this.state.ri, score: 0, results: [] }); if (this.auto) this.later(3600, () => this.advance()); return; }
    if (phase === 'intro') { this.startRound(); return; }
    if (phase === 'result') {
      if (ri + 1 < this.rounds.length) { this.setState({ phase: 'intro', ri: ri + 1 }); if (this.auto) this.later(3600, () => this.advance()); }
      else { this.setState({ phase: 'end' }); }
      return;
    }
    if (phase === 'end') { this.setState({ phase: 'title', ri: 0, score: 0, results: [] }); }
  }

  startRound() {
    const r = this.rounds[this.state.ri];
    this.run = { r, sub: 'write', typed: 0, acc: 0, pause: 0, nextW: 0, gp: 0, gTarget: 0, reading: null, hot: false,
      fieldVals: null, fieldTarget: null, fieldMode: 'none', spark: [], alarmUsed: false, alarmT: 0, searched: false, missed: false,
      early: 0, points: 0, eraseN: 0, regenTyped: 0, t: 0, cps: 13 + this.state.ri * 1.5, grounded: null, verdictT: 0 };
    if (this.swarm) this.swarm.phase = 'idle';
    this.sfx.stopHeat();
    this.setState({ phase: 'play', msg: null, stamps: [null, null], shout: null, found: null, got: [], scanning: false });
    this.renderLine();
  }

  onKey(e) {
    if (e.key === 'm' || e.key === 'M') { this.sfx.on = !this.sfx.on; this.setState({ sound: this.sfx.on }); return; }
    if (e.code !== 'Space' && e.key !== ' ' && e.key !== 'Enter') return;
    const tag = (e.target && e.target.tagName) || '';
    if (tag === 'INPUT' || tag === 'TEXTAREA' || (tag === 'BUTTON' && e.key === 'Enter')) return;
    e.preventDefault();
    if (e.repeat) return;
    this.press();
  }
  press() {
    if (!this.data) { if (this.state.loadErr) this.load(); return; }
    const { phase } = this.state;
    this.setState({ pressed: true }); this.later(120, () => this.setState({ pressed: false }));
    if (phase !== 'play') { this.advance(); return; }
    const run = this.run; if (!run) return;
    if (run.sub === 'alarm') {
      const dtA = run.t - run.alarmT, win = this.alarmWindow();
      const bonus = Math.round(SCORE.bonus * clamp(1 - dtA / win, 0, 1));
      run.points += SCORE.hit + bonus; this.addScore(SCORE.hit + bonus);
      this.sfx.stopHeat(); this.sfx.press();
      this.setState({ msg: { cls: 'good', t: `+${SCORE.hit + bonus} · busca disparada ${Math.round(dtA * 1000)} ms depois do alarme, antes da próxima palavra` } });
      const info = this.planeInfo(run.r);
      if (this.swarm && this.swarm.active() && info) {   // the probe's plane: the line, the words, the reading that fired
        run.sub = 'caught'; run.caughtT = run.t;
        this.swarm.catch(info); this.sfx.slash(); this.later(240, () => this.sfx.lock()); this.shake();
      } else { this.sfx.search(); this.beginSearch(); }
    } else if (run.sub === 'caught') {
      if (run.t - run.caughtT > 1.2) this.endCatch();
    } else if ((run.sub === 'write' || run.sub === 'fall') && run.alarmUsed) {
      this.setState({ msg: { cls: 'bad', t: 'tarde: o alarme toca uma vez por linha, e o desta já passou' } });
    } else if (run.sub === 'write') {
      run.early++; run.points += SCORE.early; this.addScore(SCORE.early); this.sfx.early();
      const v = run.reading;
      this.setState({ msg: { cls: 'bad', t: v == null ? `${SCORE.early} · cedo demais: o probe ainda não leu esta palavra` : `${SCORE.early} · cedo demais: ${p3(v)} está abaixo do limiar ${p3(this.data.tau)}` } });
    }
  }
  addScore(d) { this.setState(s => ({ score: s.score + d, bump: d > 0 ? 'bump' : 'drop' })); this.later(500, () => this.setState({ bump: null })); }
  alarmWindow() { return this.state.ri >= this.rounds.length - 1 ? 2.1 : 2.6; }   // the last line is quicker
  fieldRect() {
    const b = this.box.getBoundingClientRect(), f = this.fieldCv.getBoundingClientRect();
    return { x: f.left - b.left, y: f.top - b.top, w: f.width, h: f.height };
  }
  unit() { const b = this.box.getBoundingClientRect(); return Math.min(b.width / 100, b.height * 1.7778 / 100); }
  // what the plane shows for this line: the words the probe read before the stop, and the reading that fired
  planeInfo(r) {
    const tr = r.trigger;
    if (!tr || !tr.here) return null;
    const path = r.words.filter(w => w.eligible && w.c < tr.c && w.xy).map(w => [w.xy[0], w.xy[1], w.w.replace(/[.,;:!?]+$/, '')]);
    return { path, here: tr.here, next: tr.next, novel: tr.next_novel, reading: p3(tr.p), tau: p3(this.data.tau) };
  }
  endCatch() {
    const run = this.run; if (!run || run.sub !== 'caught') return;
    if (this.swarm) this.swarm.release(this.fieldRect());
    this.sfx.search();
    this.beginSearch();
  }

  beginSearch() {
    const run = this.run, r = run.r, tr = r.trigger;
    run.sub = 'erase'; run.searched = true; run.eraseN = tr.c; run.t0 = run.t;
    const passages = tr.windows.map(i => r.windows[i]);
    const best = bestSentences(passages, r.text, 2);
    this.setState({ scanning: true, got: [], found: null });
    this.later(1300, () => this.setState({ scanning: false, got: tr.windows }));
    this.later(2100, () => this.setState({ found: { k: `Buscamos ${tr.windows.length} dos ${r.windows.length} trechos da fala de ${r.speaker.split(' ')[0]}`, html: best.map(b => `<p>“…${markWords(b.s, r.text + ' ' + tr.regen)}…” <span style="color:#5d6b78">(trecho ${tr.windows[b.pi] + 1})</span></p>`).join('') } }));
    const G = new Set(passages.flatMap(p => cwords(p)).map(stem));
    run.grounded = G;
  }

  // ---------------------------------------------------------------- per frame
  tick(dt, T) {
    this.T = T;
    const { phase } = this.state;
    if (phase === 'title' || phase === 'intro') this.tickTitle(dt, T);
    if (phase === 'play' && this.run) this.tickPlay(dt);
    if (this.plenCv) drawPlenary(this.plenCv, T, this.run && this.run.sub === 'alarm' ? 1 : 0.4, this.state.scanning || (this.run && this.run.sub === 'erase') ? 1 : 0);
    if (this.swarm && this.fxCv) this.swarm.frame(this.fxCv, dt, T);
    if (this.fieldCv && this.swarm && this.swarm.active()) {   // the particles are the field while the effect runs
      if (!this.fieldHidden) { const c = this.fieldCv.getContext('2d'); c.setTransform(1, 0, 0, 1, 0, 0); c.clearRect(0, 0, this.fieldCv.width, this.fieldCv.height); this.fieldHidden = true; }
    } else if (this.fieldCv && (!this.lastField || T - this.lastField > 0.045 || this.fieldHidden)) {
      this.fieldHidden = false;
      this.lastField = T;
      const run = this.run;
      if (run && run.fieldTarget) {
        if (!run.fieldVals) run.fieldVals = new Float32Array(run.fieldTarget.length);
        const k = Math.min(1, dt * 12 + 0.25);
        for (let i = 0; i < run.fieldVals.length; i++) run.fieldVals[i] += (run.fieldTarget[i] - run.fieldVals[i]) * k;
      }
      drawField(this.fieldCv, run ? run.fieldVals : null, run ? run.fieldMode : 'none', T);
    }
  }

  tickTitle(dt, T) {
    if (!this.artCv) return;
    if (Math.random() < dt * 3.2 && this.rounds) {
      const pool = this.titleWords || (this.titleWords = this.rounds.flatMap(r => r.windows.join(' ').split(/\s+/)).map(w => w.replace(/[.,;:!?"“”()]/g, '')).filter(w => w.length > 6 && /^[a-zà-ú]+$/.test(w) && !/mente$/.test(w)));
      const w = pool[Math.floor(Math.random() * pool.length)].replace(/[.,;:!?"“”()]/g, '');
      this.rising.push({ t: w, x: 0.62 + Math.random() * 0.14, y: 0.4, vx: (Math.random() - 0.3) * 0.02, vy: -0.05 - Math.random() * 0.04, life: 0, size: 10 + Math.random() * 7, model: Math.random() < 0.2 });
    }
    this.rising.forEach(w => { w.life += dt / 5; w.x += (w.vx + (w.model ? 0.02 : 0)) * dt; w.y += w.vy * dt; });
    this.rising = this.rising.filter(w => w.life < 1);
    drawCongress(this.artCv, T, this.rising);
  }

  tickPlay(dt) {
    const run = this.run, r = run.r, tau = this.data.tau;
    run.t += dt;
    // gauge easing
    run.gp += (run.gTarget - run.gp) * Math.min(1, dt * 9);
    if (run.sub === 'write') {
      if (run.pause > 0) { run.pause -= dt; }
      else {
        const trig = r.trigger && !run.alarmUsed ? r.trigger.c : Infinity;
        run.acc += dt * run.cps;
        while (run.acc >= 1 && run.typed < r.text.length) {
          // a stored reading sits before the next content word: the probe reads it one token ahead
          const w = r.words[run.nextW];
          if (w && run.typed >= w.c) {
            run.nextW++;
            if (w.eligible) {
              run.reading = w.p; run.gTarget = gpos(w.p); run.fieldTarget = r.field(w.k); run.fieldMode = 'read';
              run.spark.push({ p: w.p, w: w.w }); run.pause = 0.55; this.sfx.read(); run.acc = 0;
              this.renderGauge(); break;
            } else { run.fieldMode = 'none'; }  // names and framing verbs ("afirma"): the probe was not trained there
          }
          if (run.typed >= trig) {  // the real run's trigger
            run.sub = 'alarm'; run.alarmT = run.t; run.alarmUsed = true; run.reading = r.trigger.p; run.gTarget = gpos(r.trigger.p); run.hot = true;
            const w2 = r.words.find(x => x.c >= r.trigger.c) || r.words[r.words.length - 1];   // the word it was about to write
            run.fieldTarget = r.field(w2.k); run.fieldMode = 'hot'; run.spark.push({ p: r.trigger.p, hot: true, w: '▲ alarme' });
            if (this.swarm && this.fieldCv && this.box) this.swarm.start(this.fieldRect(), run.fieldTarget, this.unit(), !this.state.results.some(x => x.r.trigger));
            this.sfx.alarm(); this.sfx.heat(this.alarmWindow());
            this.setState({ msg: { cls: 'bad', t: `ALARME · ${p3(r.trigger.p)} ≥ ${p3(tau)}: a próxima palavra provavelmente não está na fala` } });
            this.renderGauge();
            if (this.auto) this.later(Math.round(this.alarmWindow() * 560), () => this.press());   // late enough to see the takeover
            break;
          }
          run.typed++; run.acc -= 1;
          if (run.typed > 1 && r.text[run.typed - 1] === ' ' && run.nextW < r.words.length && r.words[run.nextW].c > run.typed) { run.fieldMode = 'none'; }
        }
        this.renderLine();
        if (run.typed >= r.text.length && run.sub === 'write') this.finishLine(false);
      }
    } else if (run.sub === 'alarm') {
      if (run.t - run.alarmT > this.alarmWindow()) {  // missed: the activations fall, then the model writes the word nobody stopped
        run.sub = 'fall'; run.fallT = run.t; run.missed = true; run.hot = false; run.fieldMode = 'none'; run.fieldVals = null;
        run.points += SCORE.miss; this.addScore(SCORE.miss);
        this.sfx.stopHeat(); this.sfx.fall();
        if (this.swarm && this.swarm.active()) this.swarm.fall();
        this.setState({ msg: { cls: 'bad', t: `${SCORE.miss} · as ativações caíram: o modelo vai escrever sem buscar na fala` } });
        this.renderGauge();
      }
      this.renderLine();
    } else if (run.sub === 'fall') {
      if (run.t - run.fallT > 1.4) { run.sub = 'write'; run.acc = 0; }
      this.renderLine();
    } else if (run.sub === 'caught') {
      if (run.t - run.caughtT > 8) this.endCatch();
      this.renderLine();
    } else if (run.sub === 'erase') {
      run.eraseN = Math.max(0, run.eraseN - dt * 55);
      run.hot = false; run.gTarget = 0; run.reading = null; run.fieldMode = 'none';
      if (run.eraseN <= 0 && run.t - run.t0 > 2.8) { run.sub = 'regen'; run.regenTyped = 0; run.acc = 0; this.setState({ msg: { cls: 'info', t: 'reescrevendo a linha com os trechos da própria fala ao lado' } }); }
      this.renderLine(); this.renderGauge();
    } else if (run.sub === 'regen') {
      run.acc += dt * 18;
      const n = Math.floor(run.acc); if (n) { run.regenTyped = Math.min(r.trigger.regen.length, run.regenTyped + n); run.acc -= n; }
      this.renderLine();
      if (run.regenTyped >= r.trigger.regen.length) this.finishLine(true);
    } else if (run.sub === 'verdict') {
      // handled by timers
    }
    this.renderGaugeFrame();
  }

  finishLine(regen) {
    const run = this.run, r = run.r;
    run.sub = 'verdict'; run.hot = false; run.gTarget = 0; run.fieldMode = 'none';
    const v = regen ? r.trigger.regen_verdict : r.verdict, ok = v.sonnet;
    const baseOk = r.verdict.sonnet;
    const outcome = regen ? (!baseOk && ok ? 'fixed' : baseOk && !ok ? 'broke' : ok ? 'kept' : 'still') : (ok ? 'kept' : 'passed');
    run.final = { text: regen ? r.trigger.regen : r.text, v, ok, outcome, comment: regen ? r.trigger.regen_comment : r.comment };
    this.renderLine(true);
    this.later(700, () => { this.setState({ stamps: [v.sonnet, null] }); v.sonnet ? this.sfx.good() : null; });
    this.later(1500, () => this.setState({ stamps: [v.sonnet, v.opus === undefined ? v.sonnet : v.opus] }));
    if (!ok) this.later(2300, () => { this.sfx.bad(); this.setState({ shout: r.fem ? 'ISSO ELA NÃO DISSE!' : 'ISSO ELE NÃO DISSE!' }); this.shake(); });
    let src = null;
    if (!ok || outcome === 'fixed') {
      const b = bestSentences(r.windows, r.text, 1)[0];
      if (b) src = b.s;
    }
    this.later(ok ? 3800 : 5200, () => {
      const res = { r, searched: run.searched, missed: run.missed, early: run.early, points: run.points, ...run.final, src };
      this.setState(s => ({ phase: 'result', shout: null, results: s.results.concat([res]) }));
      if (this.auto) this.later(5200, () => this.advance());
    });
  }
  shake() { if (!this.box) return; this.box.classList.remove('shake'); void this.box.offsetWidth; this.box.classList.add('shake'); }

  // ---------------------------------------------------------------- imperative rendering of the hot parts
  renderLine(done) {
    const el = this.lineEl, run = this.run; if (!el || !run) return;
    const r = run.r;
    let h = '';
    if (run.sub === 'write' || run.sub === 'alarm' || run.sub === 'caught' || run.sub === 'fall' || (run.sub === 'verdict' && !run.searched)) {
      // after a missed alarm, what the model writes next is marked: nobody searched the speech for it
      const shown = r.text.slice(0, run.typed), c = run.missed && r.trigger ? Math.min(r.trigger.c, shown.length) : shown.length;
      h = esc(shown.slice(0, c)) + (shown.length > c ? `<span class="late">${esc(shown.slice(c))}</span>` : '')
        + (run.sub === 'verdict' ? '' : `<span class="g-caret${run.sub === 'alarm' || run.sub === 'fall' ? ' red' : ''}"></span>`);
    } else if (run.sub === 'erase') {
      const n = Math.ceil(run.eraseN);
      h = esc(r.text.slice(0, n)) + '<span class="g-caret"></span>' + `<span class="gone">${esc(r.text.slice(n, r.trigger.c))}</span>`;
    } else {
      const t = r.trigger.regen, shown = run.sub === 'verdict' ? t : t.slice(0, run.regenTyped);
      h = '<span class="regen">' + shown.split(/(\s+)/).map(tok => {
        const w = cwords(tok)[0];
        return w && run.grounded && run.grounded.has(stem(w)) ? `<span class="gr">${esc(tok)}</span>` : esc(tok);
      }).join('') + '</span>' + (run.sub === 'verdict' ? '' : '<span class="g-caret"></span>');
    }
    el.className = 'g-line' + (done ? (run.final.ok ? ' good' : ' bad') : '');
    if (el._h !== h) { el.innerHTML = h; el._h = h; }
    if (this.statusEl) {
      const s = run.sub === 'alarm' ? ['alarm', `o probe disparou: ${p3(r.trigger.p)} ≥ limiar ${p3(this.data.tau)}`]
        : run.sub === 'caught' ? ['speech', 'parou antes da próxima palavra · o plano do probe']
        : run.sub === 'fall' ? ['alarm', 'o alarme passou sem busca · o modelo vai continuar escrevendo']
        : run.sub === 'erase' ? ['speech', 'parou · buscando na fala da pessoa os trechos mais próximos']
          : run.sub === 'regen' ? ['speech', 'o texto reescrito não tem leituras guardadas do probe']
            : run.sub === 'verdict' ? ['', 'linha terminada · juízes automáticos cegos julgam o apoio na fala']
              : ['', run.reading == null ? 'o probe lê antes de cada palavra de conteúdo' : `leitura antes da próxima palavra: ${p3(run.reading)}`];
      this.statusEl.className = 'g-status ' + s[0]; this.statusEl.textContent = s[1];
    }
  }
  renderGauge() {
    const run = this.run; if (!run || !this.readEl) return;
    const v = run.reading, hot = run.hot, over = v != null && v >= this.data.tau;
    this.readEl.textContent = v == null ? '—' : p3(v);
    this.readEl.className = 'v' + (hot ? ' hot' : v == null ? ' off' : '');
    this.stEl.textContent = hot ? 'Alarme' : v == null ? 'sem leitura' : over ? 'acima · o alarme desta linha já tocou' : 'abaixo do limiar';
    this.stEl.className = 'st' + (hot ? ' hot' : '');
    if (this.sparkEl) drawStrip(this.sparkEl, run.spark, this.data.tau);
  }
  renderGaugeFrame() {
    const run = this.run; if (!run || !this.fillEl) return;
    this.fillEl.style.width = (run.gp * 100).toFixed(2) + '%';
    this.gaugeEl.className = 'g-gauge' + (run.hot ? ' hot' : run.reading == null ? ' off' : '');
    if (this.box) { this.box.classList.toggle('alarm', run.sub === 'alarm'); this.box.classList.toggle('panic', run.sub === 'alarm' && run.t - run.alarmT > 1.2); }
    if (this.capEl) this.capEl.classList.toggle('hot', run.sub === 'alarm');
  }

  // ---------------------------------------------------------------- views
  refs = {
    box: el => { this.box = el; }, art: el => { this.artCv = el; }, plen: el => { this.plenCv = el; }, field: el => { this.fieldCv = el; },
    fx: el => { this.fxCv = el; },
    line: el => { this.lineEl = el; if (el && this.run) { el._h = null; this.renderLine(this.run.sub === 'verdict'); } },
    status: el => { this.statusEl = el; }, read: el => { this.readEl = el; if (el) this.renderGauge(); }, st: el => { this.stEl = el; },
    fill: el => { this.fillEl = el; }, gauge: el => { this.gaugeEl = el; }, spark: el => { this.sparkEl = el; if (el) this.renderGauge(); }, cap: el => { this.capEl = el; },
  };

  titleView() {
    const tau = this.data ? p3(this.data.tau) : '0,978', back = this.backTo();
    return html`<div class="g-ov title">
      <canvas class="art" ref=${this.refs.art}></canvas>
      <button class="g-back g-back-abs" onClick=${e => { e.stopPropagation(); go(back.href); }}><${Icon} name="arrow-left" size=${14} />${back.label}</button>
      <div class="g-card center">
        <div class="kk">Jogo · replay de execuções reais</div>
        <h2>SEJA O <i>PROBE</i></h2>
        <p>Um modelo aberto (Qwen3-4B) resume o que cada pessoa disse numa audiência da Câmara. Você lê o estado interno dele, palavra por palavra. Quando a leitura passar do limiar de <b>${tau}</b>, aperte <b>ESPAÇO</b>: o modelo para, busca os trechos da <span class="s">própria fala</span> e reescreve a linha.</p>
        ${this.data ? html`<div class="g-go"><span class="g-cap">ESPAÇO</span> começar · <span class="g-cap" style=${{ padding: 'calc(var(--u) * 0.5) calc(var(--u) * 1)' }}>M</span> som</div>`
          : this.state.loadErr ? html`<div class="g-go g-err">Não deu para carregar o jogo (${this.state.loadErr}). <span class="g-cap">ESPAÇO</span> tenta de novo</div>`
          : html`<div class="g-go">carregando o jogo…</div>`}
      </div>
    </div>`;
  }
  introView(r, i) {
    return html`<div class="g-ov"><div class="g-card">
      <div class="kk">Linha ${i + 1} de ${this.rounds.length} · audiência ${r.h} · ${r.date}</div>
      <h3>${r.topic}</h3>
      <div class="g-meta"><span class="s" style=${{ color: '#4fb3a9' }}>${r.speaker}</span> · ${r.role}</div>
      <p>${TIPS[i] || ''}</p>
      <div class="g-rules">
        <div><b style=${{ color: '#4fb3a9' }}>A fala</b>${r.windows.length} trechos de ~100 palavras do que a pessoa disse. É o que o modelo leu.</div>
        <div><b style=${{ color: '#e0b04a' }}>O modelo</b>Escreve a linha ${r.line + 1} do resumo de ${r.speaker.split(' ')[0]}, palavra por palavra.</div>
        <div><b style=${{ color: '#e5534b' }}>Você, o probe</b>A leitura dá a chance de a próxima palavra não estar na fala. Alarme a partir de ${p3(this.data.tau)}.</div>
      </div>
      <div class="g-go"><span class="g-cap">ESPAÇO</span> começar a linha</div>
    </div></div>`;
  }
  resultView(res, i) {
    const r = res.r;
    const out = {
      fixed: ['o', 'Consertada', `A linha original não tinha apoio na fala. A busca trouxe os trechos certos e a reescrita foi julgada com apoio pelos dois juízes.`],
      broke: ['a', 'A intervenção errou', `A linha original tinha apoio. A reescrita, feita com os trechos ao lado, inverteu o sentido. Isso também acontece: no experimento real, a intervenção piorou 18 resumos.`],
      kept: ['o', res.searched ? 'Mantida com apoio' : 'Com apoio', res.searched ? 'A linha já tinha apoio; a reescrita manteve o sentido. A maioria dos alarmes cai em linhas que já estavam certas.' : 'O probe não passou do limiar e a linha tinha apoio na fala.'],
      still: ['a', 'Continuou sem apoio', 'A reescrita não resolveu.'],
      passed: ['a', 'Passou sem apoio', res.missed ? 'O alarme tocou, mas a busca não aconteceu a tempo, e a linha saiu sem apoio na fala.' : 'A linha saiu sem apoio na fala.'],
    }[res.outcome];
    return html`<div class="g-ov"><div class="g-two">
      <div class="g-card">
        <div class="kk">Linha ${i + 1} · ${r.speaker}</div>
        <h3 class=${out[0]}>${out[1]}</h3>
        <p>${out[2]}</p>
        <div class="g-res">
          <b class=${res.points >= 0 ? 'pos' : 'neg'}>${res.points >= 0 ? '+' : ''}${res.points}</b><span>${r.trigger ? (res.searched ? 'você buscou no alarme' : 'o alarme tocou e você não buscou') : 'sem alarme nesta linha'}${res.early ? ` · ${res.early} ${res.early > 1 ? 'buscas' : 'busca'} fora do alarme` : ''}</span>
        </div>
        <div class="g-go"><span class="g-cap">ESPAÇO</span> ${i + 1 < this.rounds.length ? 'próxima linha' : 'ver o resultado'}</div>
      </div>
      <div class="g-card">
        <div class="kk">O que ficou no resumo</div>
        <p style=${{ color: res.ok ? '#b5e3a4' : '#f19a93', fontSize: 'calc(var(--u) * 1.25)' }}>“${res.text}”</p>
        <div class="g-meta">${res.ok ? 'com apoio na fala' : 'sem apoio na fala'} · juízes automáticos cegos</div>
        ${res.comment && html`<p style=${{ fontFamily: 'var(--f-mono)', fontSize: 'calc(var(--u) * 0.85)', color: '#8b98a5' }}>juiz: “${res.comment}”</p>`}
        ${res.src && html`<div><div class="kk" style=${{ color: '#4fb3a9' }}>Na fala</div><p style=${{ color: '#cfe9e5' }}>“…${res.src}…”</p></div>`}
      </div>
    </div></div>`;
  }
  endView() {
    const R = this.state.results, D = this.data.rounds.length;
    const n = k => R.filter(x => x.outcome === k).length;
    const alarms = R.filter(x => x.r.trigger), hit = alarms.filter(x => x.searched).length;
    const early = R.reduce((a, x) => a + x.early, 0), okN = R.filter(x => x.ok).length;
    const rag = (this.props.index && this.props.index.corpus.rag) || {};
    const pc = pct1;
    return html`<div class="g-ov"><div class="g-two">
      <div class="g-card">
        <div class="kk">Fim · ${this.state.score} pontos</div>
        <h3>Você foi o probe em ${R.length} ${R.length === 1 ? 'linha' : 'linhas'}</h3>
        <div class="g-res">
          <b>${hit}/${alarms.length}</b><span>alarmes atendidos a tempo</span>
          <b>${early}</b><span>buscas fora do alarme</span>
          <b class="pos">${n('fixed')}</b><span>${n('fixed') === 1 ? 'linha consertada' : 'linhas consertadas'} pela busca</span>
          <b>${okN}/${R.length}</b><span>linhas com apoio no seu resumo</span>
        </div>
        <div class="g-final">${R.map(x => html`<div class=${x.ok ? 'ok' : 'no'}><small>${x.r.speaker} · audiência ${x.r.h}</small>${x.text}</div>`)}</div>
      </div>
      <div class="g-card">
        <div class="kk">No experimento de verdade</div>
        <p>Em <b>${rag.participants || 480} pessoas</b> de 100 audiências, julgadas por juízes automáticos cegos: a busca consertou <span class="o">${rag.fixed || 34}</span> resumos e estragou <span class="a">${rag.broken || 18}</span>. Pessoas com alguma linha sem apoio: <b>${pc(rag.with_unsupported_baseline || 0.096)} → ${pc(rag.with_unsupported_rag || 0.0625)}</b>. Reduz um pouco; não elimina.</p>
        <p>O que o probe faz bem é avisar: uma palavra antes, ele prevê que o modelo vai escrever uma palavra que não está na fala (AUC 0,905 no Qwen3-4B).</p>
        <div class="g-btns">
          <button class="g-btn" onClick=${() => this.setState({ phase: 'intro', ri: 0, score: 0, results: [] })}>Jogar de novo</button>
          <button class="g-btn ghost" onClick=${() => go(this.backTo().href)}>${this.backTo().long}</button>
        </div>
      </div>
    </div></div>`;
  }

  // where the back buttons go: the hearing the game was opened from, or the home page (?de=inicio)
  backTo() {
    const de = this.props.q && this.props.q.de;
    if (de === 'inicio') return { href: '#/', label: 'Início', long: 'Voltar ao início' };
    const id = +de || 38;
    return { href: '#/audiencia/' + id, label: 'Audiência ' + id, long: 'Voltar à audiência' };
  }

  render({ q }, st) {
    const back = this.backTo();
    const data = this.data, r = data && this.rounds[st.ri];
    const tauPos = data ? gpos(data.tau) : 0.6;
    const stamp = (v, who) => html`<div class=${'g-stamp' + (v === true ? ' ok' : v === false ? ' no' : '')}>
      <b>${v === true ? '✓' : v === false ? '✕' : '·'}</b><span>${v === true ? 'com apoio na fala' : v === false ? 'sem apoio na fala' : 'aguardando'}<span class="who">${who}</span></span></div>`;
    return html`<main class="gpage" data-screen-label="Tela 4 · Seja o probe">
      <div class="gbox" ref=${this.refs.box} tabindex="0" onClick=${e => { if (e.target.closest && e.target.closest('button')) return; this.press(); }}>
        <div class="g-scanlines"></div><div class="flash"></div>
        ${data && r && html`<div class="g-in">
          <div class="g-top">
            <div class="g-brand"><button class="g-back" onClick=${e => { e.stopPropagation(); go(back.href); }}><${Icon} name="arrow-left" size=${14} />${back.label}</button><b>SEJA O <i>PROBE</i></b><span>replay · Qwen3-4B · camada ${data.layer}</span></div>
            <div class="mid">${this.rounds.map((_, i) => html`<span class=${'pip' + (i === st.ri ? ' on' : i < st.results.length ? ' done' : '')}></span>`)}</div>
            <div class="right">
              <span class=${'g-score' + (st.bump ? ' ' + st.bump : '')}>PONTOS ${String(Math.max(0, st.score)).padStart(4, '0')}</span>
              <button class="g-snd" onClick=${e => { e.stopPropagation(); this.sfx.on = !this.sfx.on; this.setState({ sound: this.sfx.on }); }}><${Icon} name=${st.sound ? 'volume-2' : 'volume-x'} size=${14} /> M</button>
              <button class="g-infob" onClick=${e => { e.stopPropagation(); this.setState({ info: !st.info }); }}>Como funciona</button>
            </div>
          </div>
          <div class="g-cols">
            <section class="g-col g-left">
              <div class="g-k"><span class="sw" style=${{ background: '#4fb3a9' }}></span>A fala <small>· o que foi dito</small></div>
              <div class="g-who"><b>${r.speaker}</b><span>${r.role}</span></div>
              <canvas class="g-plen" ref=${this.refs.plen}></canvas>
              <div class="g-wins">${r.windows.slice(0, 16).map((w, i) => html`<div class=${'g-win' + ((st.got || []).includes(i) ? ' got' : st.scanning ? ' scan' : '')}><span class="n">${i + 1}</span>${w}</div>`)}</div>
              <div class=${'g-found' + (st.found ? ' on' : '')}>${st.found && html`<span class="fk">${st.found.k}</span><div dangerouslySetInnerHTML=${{ __html: st.found.html }}></div>`}</div>
            </section>
            <section class="g-col g-mid">
              <div class="g-k"><span class="sw" style=${{ background: '#e0b04a' }}></span>O modelo escreve <small>· resumo gerado por IA; o erro é do modelo, não do participante</small></div>
              <div class="g-prior">${r.prior.map((p, i) => html`<div data-n=${i + 1}>${p}</div>`)}</div>
              <div class="g-linebox">
                <div class="g-num">LINHA ${r.line + 1} DE ${r.n_lines}</div>
                <div class="g-line" ref=${this.refs.line}></div>
              </div>
              <div class="g-status" ref=${this.refs.status}></div>
              <div class="g-stripk">O que o probe leu antes de cada palavra</div>
              <canvas class="g-strip" ref=${this.refs.spark}></canvas>
              <div class="g-judges">
                <div class="jk">Juízes automáticos cegos</div>
                <div class="g-stamps">${stamp(st.stamps[0], 'Claude Sonnet')}${stamp(st.stamps[1], 'Claude Opus')}</div>
              </div>
            </section>
            <section class="g-col g-right">
              <div class="g-k"><span class="sw" style=${{ background: '#e5534b' }}></span>O probe <small>· você</small></div>
              <canvas class="g-field" ref=${this.refs.field}></canvas>
              <div class="g-legend"><span><i style=${{ background: '#4fb3a9' }}></i>puxa para a fala</span><span><i style=${{ background: '#e5534b' }}></i>puxa para fora</span></div>
              <div class="g-what">Os 2.560 números da camada ${data.layer}, cada um multiplicado pelo peso do probe. A soma vira a leitura:</div>
              <div class="g-gauge" ref=${this.refs.gauge}>
                <div class="fill" ref=${this.refs.fill}></div>
                <div class="tau" style=${{ left: (tauPos * 100).toFixed(1) + '%' }}><span>limiar ${p3(data.tau)}</span></div>
                <span class="tk" style=${{ left: '0%' }}>0</span><span class="tk" style=${{ left: (gpos(0.5) * 100) + '%' }}>0,5</span><span class="tk" style=${{ left: (gpos(0.9) * 100) + '%' }}>0,9</span><span class="tk" style=${{ left: (gpos(0.99) * 100) + '%' }}>0,99</span><span class="tk" style=${{ left: '100%' }}>0,999</span>
              </div>
              <div class="g-read"><span class="v off" ref=${this.refs.read}></span><span class="st" ref=${this.refs.st}></span></div>
              <div class="g-what">P(a próxima palavra não está na fala), lida um token antes de ela ser escrita.</div>
            </section>
          </div>
          <div class="g-bottom">
            <div class="g-key"><span class=${'g-cap' + (st.pressed ? ' down' : '')} ref=${this.refs.cap}>ESPAÇO</span>parar e buscar na fala</div>
            <div class=${'g-msg ' + (st.msg ? st.msg.cls : '')}>${st.msg ? st.msg.t : ''}</div>
            <div class="g-honest">O alarme decide quando buscar; a busca na fala corrige.</div>
          </div>
        </div>`}
        <canvas class="g-fx" ref=${this.refs.fx}></canvas>
        ${st.shout && html`<div class="g-shout">${st.shout}</div>`}
        ${st.info && data && html`<div class="ginfo" onClick=${e => { e.stopPropagation(); this.setState({ info: false }); }}>
          <b>Replay de execuções reais.</b> As leituras são do probe que rodou de verdade como gatilho (camada 21 do Qwen3-4B, limiar ${p3(data.tau)}, ajustado para disparar em cerca de 35% das linhas). O probe lê antes de cada palavra de conteúdo; nomes e verbos como “afirma” não contam, e a busca só pode disparar depois das duas primeiras palavras de conteúdo da linha, uma vez por linha. Quando você aperta no alarme, aparece o plano do probe: ${data.plane ? data.plane.n.toLocaleString('pt-BR') : '2.560'} palavras de resumos do Qwen3-4B, metade que estava na fala e metade que não estava, todas do treino do probe (mostram o que ele aprendeu, não um teste). Um eixo é o que o probe lê (w · h + b), o outro só espalha os pontos, e o limiar vira uma linha reta. O caminho amarelo são as palavras da linha que você está jogando; o ponto grande usa a leitura que disparou. O texto reescrito não tem leituras guardadas. Os trechos buscados e as linhas reescritas são os da execução real; os vereditos são dos juízes automáticos cegos (Claude Sonnet e Opus). As ${this.rounds.length} linhas vêm de três audiências: uma sem alarme, duas em que a busca conserta e uma em que o alarme toca numa linha que já estava certa. No experimento completo a busca também erra: consertou ${(this.props.index && this.props.index.corpus.rag.fixed) || 34} resumos e estragou ${(this.props.index && this.props.index.corpus.rag.broken) || 18}. Teclas: ESPAÇO (ou clique) aperta, M liga e desliga o som.
        </div>`}
        ${(!data || st.phase === 'title') && this.titleView()}
        ${data && st.phase === 'intro' && this.introView(r, st.ri)}
        ${data && st.phase === 'result' && st.results.length > 0 && this.resultView(st.results[st.results.length - 1], st.results.length - 1)}
        ${data && st.phase === 'end' && this.endView()}
      </div>
    </main>`;
  }
}
