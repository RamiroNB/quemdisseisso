// The alarm's special effect in "Seja o probe" (game.js), drawn on one canvas over the whole game.
// When the probe fires, the 2,560 characters of the ASCII field (one per dimension of layer 21, as in the panel)
// turn red, shake, break out of the panel and take the screen. SPACE draws the probe's threshold across the
// screen and the characters land on the probe's plane: 2,560 real word positions of Qwen3-4B summaries
// (data/game.json "plane", scripts/export_portal_data.py probe_plane), teal where the next word was in the
// speech and red where it was not, with this line's own words as a path and the reading that fired as the big
// point. Without SPACE they fall off the screen and the model goes on writing.
const RAMP = ' ·:-=+*#%@';
const PAL = [[79, 179, 169], [229, 83, 75], [255, 222, 214], [224, 176, 74], [150, 74, 66], [120, 135, 150]];
const TEAL = 0, RED = 1, HOT = 2, AMBER = 3, DIM = 4;
const clamp = (x, a, b) => Math.max(a, Math.min(b, x));
const ramp = (t, a, b) => clamp((t - a) / (b - a), 0, 1);
const easeOut = t => 1 - Math.pow(1 - t, 3);
const easeInOut = t => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);
const hash = i => { const s = Math.sin(i * 127.1 + 311.7) * 43758.5453; return s - Math.floor(s); };   // stable per particle
const rgba = (c, a) => `rgba(${c[0]},${c[1]},${c[2]},${a.toFixed(3)})`;

// the visible part of the line p + s d inside [0, W] x [0, H]
function clipLine(px, py, dx, dy, W, H) {
  let s0 = -1e9, s1 = 1e9;
  for (const [p, d, hi] of [[px, dx, W], [py, dy, H]]) {
    if (Math.abs(d) < 1e-9) { if (p < 0 || p > hi) return null; continue; }
    let a = -p / d, b = (hi - p) / d;
    if (a > b) [a, b] = [b, a];
    s0 = Math.max(s0, a); s1 = Math.min(s1, b);
  }
  return s0 < s1 ? [px + dx * s0, py + dy * s0, px + dx * s1, py + dy * s1] : null;
}

export class Swarm {
  constructor(plane, cols, rows) {
    this.plane = plane; this.cols = cols; this.rows = rows;
    const N = this.N = cols * rows, F = () => new Float32Array(N);
    Object.assign(this, { hx: F(), hy: F(), x: F(), y: F(), sx: F(), sy: F(), tx: F(), ty: F(), rx: F(), ry: F(),
      vx: F(), vy: F(), a: F(), rel: F(), dur: F(), sw: F(), dl: F() });
    this.gl = new Uint8Array(N); this.pos = new Uint8Array(N); this.vis = new Uint8Array(N); this.lab = new Uint8Array(N);
    for (let k = 0; k < N; k++) this.lab[k] = plane && plane.novel[k] === '1' ? 1 : 0;
    this.phase = 'idle'; this.t = 0;
  }
  active() { return this.phase !== 'idle'; }

  // ---------------------------------------------------------------- sprites: each glyph in each colour
  sprites(dpr) {
    if (this.sdpr === dpr) return;
    this.sdpr = dpr;
    const base = Math.round(24 * dpr), S = Math.ceil(base * 2.2);
    const mk = (w, f) => { const c = document.createElement('canvas'); c.width = c.height = w; f(c.getContext('2d')); return c; };
    this.gs = PAL.map(c => RAMP.split('').map(g => mk(S, x => {
      if (g === ' ') return;
      x.font = `${base}px "IBM Plex Mono", monospace`; x.textAlign = 'center'; x.textBaseline = 'middle';
      x.shadowColor = rgba(c, 0.9); x.shadowBlur = base * 0.5; x.fillStyle = `rgb(${c})`; x.fillText(g, S / 2, S / 2);
    })));
    // the plane's ASCII cells: seven colours from teal (all in the speech) to red (none), little glow so they read as type
    this.gm = [0, 1, 2, 3, 4, 5, 6].map(i => {
      const f = i / 6, c = PAL[TEAL].map((v, j) => Math.round(v + (PAL[RED][j] - v) * f));
      return RAMP.split('').map(g => mk(S, x => {
        if (g === ' ') return;
        x.font = `500 ${base}px "IBM Plex Mono", monospace`; x.textAlign = 'center'; x.textBaseline = 'middle';
        x.shadowColor = rgba(c, 0.8); x.shadowBlur = base * 0.22; x.fillStyle = `rgb(${c})`; x.fillText(g, S / 2, S / 2);
      }));
    });
    this.gBox = S / base;   // sprite box per px of font size
  }
  glyph(x, k, col, px, py, fs, al) {
    if (al <= 0.01) return;
    const b = fs * this.gBox; x.globalAlpha = Math.min(1, al); x.drawImage(this.gs[col][this.gl[k]], px - b / 2, py - b / 2, b, b);
  }
  cellGlyph(x, m, gi, px, py, fs, al) {
    if (al <= 0.01) return;
    const b = fs * this.gBox; x.globalAlpha = Math.min(1, al); x.drawImage(this.gm[m][gi], px - b / 2, py - b / 2, b, b);
  }

  // ---------------------------------------------------------------- phases
  // the alarm: particles start on their cells of the field (rect, relative to the game box) with the field's values
  start(rect, vals, u, hint) {
    this.phase = 'escape'; this.t = 0; this.u = u; this.rect = rect; this.hint = !!hint;
    const { N, cols: C, rows: R } = this, cw = rect.w / C, ch = rect.h / R;
    this.fs = Math.max(4, ch * 1.02); this.ch = ch;
    const order = new Array(N);
    for (let k = 0; k < N; k++) {
      const v = vals ? vals[k] : 0, a = Math.min(1, Math.pow(Math.abs(v) * 127 / 52, 0.7));   // the field's own mapping
      const rr = Math.floor(k / C), cc = k % C;
      this.a[k] = a; this.pos[k] = v > 0 ? 1 : 0; order[k] = k;
      this.gl[k] = a < 0.14 ? 1 : Math.min(RAMP.length - 1, 1 + Math.floor(a * (RAMP.length - 1)));
      this.vis[k] = a >= 0.14 || (k + rr) % 2 === 0 ? 1 : 0;
      this.hx[k] = this.x[k] = rect.x + (cc + 0.5) * cw; this.hy[k] = this.y[k] = rect.y + (rr + 0.5) * ch;
      this.rx[k] = (cc + 0.5 + (hash(k) - 0.5) * 9) / C; this.ry[k] = (rr + 0.5 + (hash(k + 9.3) - 0.5) * 6) / R;   // screen fractions
      this.dur[k] = 0.6 + 0.6 * hash(k + 17.1); this.sw[k] = (hash(k + 5.7) - 0.5) * 0.8;
    }
    order.sort((i, j) => this.a[j] - this.a[i]);   // the strongest dimensions break out first
    order.forEach((k, r) => { this.rel[k] = 0.3 + 1.0 * r / N + 0.1 * hash(k + 3.3); });
  }
  // SPACE in time: info = { path: [[X, Y, word]], here: [X, Y], next, reading, tau, novel }
  catch(info) {
    this.snap();
    this.phase = 'plane'; this.t = 0; this.info = info; this.lay = null;
  }
  // back to the field after the plane (the search starts under it)
  release(rect) {
    this.snap();
    if (rect) this.rect = rect;
    const { cols: C, rows: R } = this, r = this.rect, cw = r.w / C, ch = r.h / R;
    for (let k = 0; k < this.N; k++) { this.hx[k] = r.x + (k % C + 0.5) * cw; this.hy[k] = r.y + (Math.floor(k / C) + 0.5) * ch; }
    this.phase = 'home'; this.t = 0;
  }
  // no SPACE: they fall
  fall() {
    this.snap();
    this.phase = 'fall'; this.t = 0;
    for (let k = 0; k < this.N; k++) {
      this.vx[k] = (hash(k + 71.3) - 0.5) * 0.3; this.vy[k] = -(0.04 + 0.22 * hash(k + 12.9)); this.dl[k] = 0.3 * hash(k + 61.1);
    }
  }
  snap() { this.sx.set(this.x); this.sy.set(this.y); }

  // ---------------------------------------------------------------- per frame
  frame(cv, dt, T) {
    const r = cv.getBoundingClientRect(), dpr = window.devicePixelRatio || 1, W = r.width, H = r.height;
    if (!W || !H) return;
    const resized = cv.width !== Math.round(W * dpr) || cv.height !== Math.round(H * dpr);
    if (resized) { cv.width = Math.round(W * dpr); cv.height = Math.round(H * dpr); this.lay = null; }
    const x = cv.getContext('2d');
    x.setTransform(dpr, 0, 0, dpr, 0, 0);
    if (this.phase === 'idle') { if (this.dirty) { x.clearRect(0, 0, W, H); this.dirty = false; } return; }
    this.dirty = true;
    this.sprites(dpr);
    this.t += dt;
    this.W = W; this.H = H; this.u = Math.min(W / 100, H * 1.7778 / 100);
    x.globalCompositeOperation = 'source-over'; x.globalAlpha = 1; x.clearRect(0, 0, W, H);
    if (this.phase === 'escape') this.drawEscape(x, T);
    else if (this.phase === 'plane' || this.phase === 'home') this.drawPlane(x, T);
    else if (this.phase === 'fall') this.drawFall(x, T);
    x.globalCompositeOperation = 'source-over'; x.globalAlpha = 1;
  }

  // the takeover: red, shaking, out of the panel, over everything
  drawEscape(x, T) {
    const { t, W, H, N, u, rect } = this, fs = this.fs;
    const dark = 0.42 * ramp(t, 0.3, 1.5);
    if (dark > 0) { x.fillStyle = `rgba(14,5,6,${dark.toFixed(3)})`; x.fillRect(0, 0, W, H); }
    // the panel's frame glows and then gives way
    const fr = 1 - ramp(t, 0.35, 0.9);
    if (fr > 0) {
      x.save(); x.strokeStyle = rgba(PAL[RED], (0.55 + 0.45 * Math.sin(T * 40)) * fr); x.lineWidth = u * 0.18;
      x.shadowColor = 'rgba(229,83,75,0.9)'; x.shadowBlur = u * 1.4;
      const j = u * 0.35 * ramp(t, 0, 0.3) * fr;
      x.strokeRect(rect.x + (Math.random() - 0.5) * j, rect.y + (Math.random() - 0.5) * j, rect.w, rect.h); x.restore();
    }
    // the shockwave when they break out
    const sw = ramp(t, 0.26, 0.9);
    if (sw > 0 && sw < 1) {
      const e = easeOut(sw), R = e * Math.hypot(W, H) * 0.8;
      x.strokeStyle = rgba(PAL[RED], 0.5 * (1 - sw)); x.lineWidth = u * 0.5 * (1 - sw) + 1;
      x.beginPath(); x.arc(rect.x + rect.w / 2, rect.y + rect.h / 2, R, 0, Math.PI * 2); x.stroke();
    }
    const streak = new Path2D();
    let streaks = 0;
    x.globalCompositeOperation = 'lighter';
    const trem = 1 + 3.2 * ramp(t, 1.2, 2.5);
    for (let k = 0; k < N; k++) {
      let px, py, size, al, col;
      const a = this.a[k];
      if (t < this.rel[k]) {   // still in the panel: red spreads, the shaking grows
        const A = this.ch * (0.15 + 1.4 * ramp(t, 0, 0.3));
        px = this.hx[k] + (Math.random() - 0.5) * A; py = this.hy[k] + (Math.random() - 0.5) * A;
        size = fs; al = this.vis[k] ? (a < 0.14 ? 0.18 : 0.3 + 0.7 * a) : 0;
        col = t < 0.06 ? HOT : t > 0.3 * hash(k + 31.7) || this.pos[k] ? RED : TEAL;
      } else {
        const tt = clamp((t - this.rel[k]) / this.dur[k], 0, 1), e = easeOut(tt);
        const dx = this.rx[k] * W - this.hx[k], dy = this.ry[k] * H - this.hy[k], s = this.sw[k] * Math.sin(Math.PI * tt);
        px = this.hx[k] + dx * e - dy * s; py = this.hy[k] + dy * e + dx * s;
        const orb = (0.3 + 0.9 * hash(k + 2.2)) * u * e, an = hash(k + 8.8) * 6.283 + T * (2 + 2.5 * hash(k + 4.4));
        px += Math.cos(an) * orb + (Math.random() - 0.5) * trem * 2; py += Math.sin(an) * orb + (Math.random() - 0.5) * trem * 2;
        size = fs * (1 + (0.4 + 1.1 * a) * e);
        al = (0.32 + 0.6 * a) * (this.vis[k] ? 1 : e) * (0.85 + 0.15 * Math.sin(T * 9 + k));
        col = hash(k + Math.floor(t * 14) * 1.7) < 0.035 ? HOT : RED;
        if (Math.random() < 0.06) this.gl[k] = 1 + Math.floor(Math.random() * (RAMP.length - 1));   // glitching glyphs
        if (tt < 1 && Math.abs(px - this.x[k]) + Math.abs(py - this.y[k]) > 3) { streak.moveTo(this.x[k], this.y[k]); streak.lineTo(px, py); streaks++; }
      }
      this.glyph(x, k, col, px, py, size, al);
      this.x[k] = px; this.y[k] = py;
    }
    if (streaks) { x.globalAlpha = 1; x.strokeStyle = 'rgba(229,83,75,0.22)'; x.lineWidth = 1; x.stroke(streak); }
    x.globalCompositeOperation = 'source-over';
    const hi = this.hint ? ramp(t, 0.5, 0.8) : 0;   // first alarm of the game: say what to do
    if (hi > 0) {
      const pulse = 0.8 + 0.2 * Math.sin(T * 10), cx = W / 2, cy = H * 0.63;
      x.globalAlpha = hi; x.textAlign = 'center'; x.textBaseline = 'middle';
      x.save(); x.shadowColor = 'rgba(229,83,75,0.9)'; x.shadowBlur = u * 2.5 * pulse;
      this.halo(x, 'ESPAÇO', cx, cy, `800 ${(u * 4.4).toFixed(1)}px "Plus Jakarta Sans", sans-serif`, `rgba(255,240,236,${pulse.toFixed(3)})`); x.restore();
      this.halo(x, 'antes que as ativações caiam', cx, cy + u * 3.2, `500 ${(u * 1.3).toFixed(1)}px "Work Sans", sans-serif`, 'rgba(255,190,182,0.95)');
      x.globalAlpha = 1;
    }
  }

  // where the plane sits on this screen: the threshold runs across it, the cloud fills the rest
  layout() {
    const { W, H, plane: P } = this, u = this.u;
    const an = Math.atan2(H, W) * 0.78;
    const el = [Math.cos(an), Math.sin(an)], en = [Math.sin(an), -Math.cos(an)];   // along the line (down-right); toward "fora da fala" (up-right)
    const q = (arr, f) => { const s = Float32Array.from(arr).sort(); return s[Math.floor(f * (s.length - 1))]; };
    const n = P.n, sxs = new Float32Array(n), sys = new Float32Array(n);
    for (let i = 0; i < n; i++) { sxs[i] = P.y[i] * el[0] + P.x[i] * en[0]; sys[i] = P.y[i] * el[1] + P.x[i] * en[1]; }
    const yHi = q(P.y, 0.985), yLo = q(P.y, 0.015);
    const arrow = [[-6.5, yHi + 1.4], [3, yHi + 1.4]];
    const extra = [[3.6, yLo], [3.6, (yLo + yHi) / 2], [3.6, yHi], ...arrow];
    let x0 = q(sxs, 0.02), x1 = q(sxs, 0.98), y0 = q(sys, 0.02), y1 = q(sys, 0.975);   // the far tails may leave the screen
    extra.forEach(([X, Y]) => { const a = Y * el[0] + X * en[0], b = Y * el[1] + X * en[1]; x0 = Math.min(x0, a); x1 = Math.max(x1, a); y0 = Math.min(y0, b); y1 = Math.max(y1, b); });
    const top = u * 6.2, bot = u * 6.4, side = u * 3;
    const k = Math.min((W - 2 * side) / (x1 - x0), (H - top - bot) / (y1 - y0));
    const ox = side + ((W - 2 * side) - (x1 - x0) * k) / 2 - x0 * k, oy = top + ((H - top - bot) - (y1 - y0) * k) / 2 - y0 * k;
    const at = (X, Y) => [ox + (Y * el[0] + X * en[0]) * k, oy + (Y * el[1] + X * en[1]) * k];
    const line = X => { const [px, py] = at(X, 0); return clipLine(px, py, el[0], el[1], W, H); };
    // where each class's legend goes: left edge of its cloud, at its median height
    const cls = [0, 1].map(c => {
      const xs = [], ys = [];
      for (let i = 0; i < n; i++) if ((P.novel[i] === '1') === (c === 1)) { const [a, b] = at(P.x[i], P.y[i]); xs.push(a); ys.push(b); }
      return [q(xs, 0.03), q(ys, 0.5)];
    });
    this.lay = { el, en, k, at, an, tau: line(0), iso5: line(P.iso['0.5']), iso9: line(P.iso['0.9']), arrow, yLo, yHi, cls };
    for (let i = 0; i < this.N; i++) { const [a, b] = i < n ? at(P.x[i], P.y[i]) : at(-20, 0); this.tx[i] = a; this.ty[i] = b; }
    // the cloud is drawn as ASCII: every word lands on a character cell; a cell's character says how many words
    // it holds and its colour what share of them was not in the speech
    const cw = u * 0.78, ch = u * 1.3, nc = Math.ceil(W / cw), nr = Math.ceil(H / ch), occ = new Set();
    this.cell = new Int32Array(this.N);
    const G = this.grid = { cw, ch, nc, fs: ch * 0.9, cnt: new Uint16Array(nc * nr), red: new Uint16Array(nc * nr), beyond: new Uint8Array(nc * nr) };
    for (let i = 0; i < this.N; i++) {
      const c = Math.floor(this.tx[i] / cw), r = Math.floor(this.ty[i] / ch);
      if (c < 0 || r < 0 || c >= nc || r >= nr) { this.cell[i] = -1; continue; }
      const id = r * nc + c; this.cell[i] = id; occ.add(id);
      this.tx[i] = (c + 0.5) * cw; this.ty[i] = (r + 0.5) * ch;
      if (i < n && P.x[i] > 0) G.beyond[id] = 1;
    }
    G.occ = Int32Array.from(occ);
    // the landing follows the blade along the line
    const L = this.lay.tau, lx = L ? L[2] - L[0] : 1, ly = L ? L[3] - L[1] : 0, ll = lx * lx + ly * ly || 1;
    for (let i = 0; i < this.N; i++) {
      const s = L ? clamp(((this.tx[i] - L[0]) * lx + (this.ty[i] - L[1]) * ly) / ll, 0, 1) : 0.5;
      this.dl[i] = 0.1 + 0.45 * s + 0.3 * hash(i + 41.3);
    }
  }

  drawPlane(x, T) {
    if (!this.lay) this.layout();
    const { W, H, N, u, lay: L, info } = this, home = this.phase === 'home';
    // 'home': labels go first, the points fly back into the panel, then the screen comes back
    const t = home ? 9 : this.t, out = home ? 1 - ramp(this.t, 0, 0.2) : 1, back = home ? 1 - ramp(this.t, 0.3, 0.7) : 1;
    if (home && this.t > 0.75) { this.phase = 'idle'; return; }
    // backdrop, and the probe's reading as a light across the plane: teal far from the line, red past it
    x.fillStyle = `rgba(8,11,15,${(0.96 * easeOut(ramp(t, 0.02, 0.26)) * back).toFixed(3)})`; x.fillRect(0, 0, W, H);
    const sh = ramp(t, 0.25, 0.9) * back;
    if (sh > 0) {
      const [ax, ay] = L.at(-13, 0), [bx, by] = L.at(5.5, 0), g = x.createLinearGradient(ax, ay, bx, by), f = X => (X + 13) / 18.5;
      g.addColorStop(0, rgba(PAL[TEAL], 0.1 * sh)); g.addColorStop(f(this.plane.iso['0.5']), rgba(PAL[TEAL], 0.025 * sh));
      g.addColorStop(f(this.plane.iso['0.9']), rgba(PAL[RED], 0.03 * sh)); g.addColorStop(f(0), rgba(PAL[RED], 0.1 * sh)); g.addColorStop(1, rgba(PAL[RED], 0.17 * sh));
      x.fillStyle = g; x.fillRect(0, 0, W, H);
    }
    if (t < 0.14) { x.fillStyle = `rgba(255,236,230,${(0.22 * (1 - t / 0.14)).toFixed(3)})`; x.fillRect(0, 0, W, H); }
    const lab = ramp(t, 1.3, 1.9) * out;
    if (lab > 0) {   // the iso-lines of the reading, 0.5 and 0.9
      x.save(); x.setLineDash([u * 0.4, u * 0.5]); x.lineWidth = 1;
      [[L.iso5, '0,5'], [L.iso9, '0,9']].forEach(([s, name]) => {
        if (!s) return;
        x.strokeStyle = `rgba(150,165,180,${(0.35 * lab).toFixed(3)})`; x.beginPath(); x.moveTo(s[0], s[1]); x.lineTo(s[2], s[3]); x.stroke();
        this.along(x, s, 0.9, name, `${(u * 0.8).toFixed(1)}px "IBM Plex Mono", monospace`, `rgba(150,165,180,${(0.8 * lab).toFixed(3)})`);
      });
      x.restore();
    }
    // the particles fly to the plane as characters; the ones that landed are counted into their cell
    const G = this.grid, counting = !home && t >= 0.09;
    if (counting) { G.cnt.fill(0); G.red.fill(0); }
    x.globalCompositeOperation = 'lighter';
    for (let k = 0; k < N; k++) {
      let px, py, col, al;
      if (home) {
        const d0 = 0.15 * hash(k + 51.1), tt = easeInOut(ramp(this.t, d0, d0 + 0.45));
        px = this.sx[k] + (this.hx[k] - this.sx[k]) * tt; py = this.sy[k] + (this.hy[k] - this.sy[k]) * tt;
        col = tt < 0.5 ? (this.lab[k] ? RED : TEAL) : (this.pos[k] ? RED : TEAL);
        al = (0.3 + 0.6 * this.a[k]) * (1 - ramp(tt, 0.6, 1));
      } else if (t < 0.09) {   // hit-stop
        px = this.sx[k]; py = this.sy[k]; col = HOT; al = 0.5 + 0.5 * this.a[k];
      } else {
        const tt = ramp(t, this.dl[k], this.dl[k] + 0.9), e = easeInOut(tt);
        if (tt >= 1 && this.cell[k] >= 0) { G.cnt[this.cell[k]]++; G.red[this.cell[k]] += this.lab[k]; this.x[k] = this.tx[k]; this.y[k] = this.ty[k]; continue; }
        const dx = this.tx[k] - this.sx[k], dy = this.ty[k] - this.sy[k], s = 0.12 * Math.sin(Math.PI * tt) * (hash(k + 6.6) - 0.5);
        px = this.sx[k] + dx * e - dy * s; py = this.sy[k] + dy * e + dx * s;
        col = tt < 0.3 ? RED : this.lab[k] ? RED : TEAL; al = 0.4 + 0.5 * this.a[k];
      }
      this.glyph(x, k, col, px, py, this.fs * (home ? 1.4 : 1.4 - 0.4 * ramp(t, 0.2, 1.2)), al);
      if (!home) { this.x[k] = px; this.y[k] = py; }
    }
    if (counting) {   // the cloud in ASCII: · one word … @ a dozen or more; past the line the cells pulse
      for (const c of G.occ) {
        const n = G.cnt[c]; if (!n) continue;
        const gi = n >= 12 ? 9 : n >= 9 ? 8 : n >= 7 ? 7 : n >= 6 ? 6 : n >= 5 ? 5 : n;
        const b = G.beyond[c], tw = 0.84 + 0.16 * Math.sin(T * (b ? 5 : 1.4) + c * 0.73);
        this.cellGlyph(x, Math.round(G.red[c] / n * 6), gi, (c % G.nc + 0.5) * G.cw, (Math.floor(c / G.nc) + 0.5) * G.ch,
          G.fs * (b ? 1.15 : 1), (0.45 + 0.55 * Math.min(1, n / 7)) * tw * (b ? 1.3 : 1));
      }
    }
    x.globalCompositeOperation = 'source-over';
    x.globalAlpha = 1;
    // the threshold: a blade across the screen
    if (L.tau) {
      const s = L.tau, sp = easeOut(ramp(t, 0.09, 0.45)), hx = s[0] + (s[2] - s[0]) * sp, hy = s[1] + (s[3] - s[1]) * sp;
      x.save(); x.lineCap = 'round'; x.globalAlpha = out;
      x.shadowColor = 'rgba(229,83,75,0.95)'; x.shadowBlur = u * (1.4 + 0.4 * Math.sin(T * 5));
      x.strokeStyle = 'rgba(229,83,75,0.95)'; x.lineWidth = u * 0.26; x.beginPath(); x.moveTo(s[0], s[1]); x.lineTo(hx, hy); x.stroke();
      x.shadowBlur = 0; x.strokeStyle = 'rgba(255,238,232,0.95)'; x.lineWidth = u * 0.07; x.stroke();
      if (sp < 1) {
        const g = x.createRadialGradient(hx, hy, 0, hx, hy, u * 5); g.addColorStop(0, 'rgba(255,245,240,0.95)'); g.addColorStop(0.2, 'rgba(229,83,75,0.6)'); g.addColorStop(1, 'rgba(229,83,75,0)');
        x.fillStyle = g; x.fillRect(hx - u * 5, hy - u * 5, u * 10, u * 10);
      }
      x.restore();
      if (lab > 0) this.along(x, s, 0.9, `limiar do alarme · ${info.tau}`, `600 ${(u * 0.95).toFixed(1)}px "Plus Jakarta Sans", sans-serif`, rgba([255, 150, 140], lab));
    }
    if (lab > 0) this.drawLabels(x, lab);
    this.drawPath(x, T, t, out);
    const cap = ramp(t, 2.6, 3.2) * out;
    if (cap > 0) this.drawCaption(x, cap);
  }

  // text parallel to a line segment, a little above it, at fraction f of its visible part
  along(x, s, f, text, font, color) {
    const px = s[0] + (s[2] - s[0]) * f, py = s[1] + (s[3] - s[1]) * f, u = this.u;
    x.save(); x.translate(px + this.lay.en[0] * u * 0.6, py + this.lay.en[1] * u * 0.6); x.rotate(this.lay.an);
    x.font = font; x.fillStyle = color; x.textAlign = 'right'; x.textBaseline = 'bottom'; x.fillText(text, 0, 0); x.restore();
  }

  drawLabels(x, al) {
    const { u, lay: L, W, H } = this;
    x.textBaseline = 'alphabetic';
    // the two kinds of point, labelled beside their clouds (the colour is the truth; the line is the probe's call)
    const legend = ([px, py], big, small, c) => {
      const left = px > u * 22;   // room to the left of the cloud, else inside it
      px = left ? px - u * 2.6 : u * 3; py = clamp(py, u * 8, H - u * 8);
      x.textAlign = left ? 'right' : 'left';
      this.halo(x, big, px, py, `800 ${(u * 1.9).toFixed(1)}px "Plus Jakarta Sans", sans-serif`, rgba(c, 0.95 * al));
      this.halo(x, small, px, py + u * 1.5, `${(u * 0.95).toFixed(1)}px "Work Sans", sans-serif`, rgba(c, 0.72 * al));
    };
    legend(L.cls[1], 'FORA DA FALA', 'a próxima palavra não estava na fala', [240, 120, 110]);
    legend(L.cls[0], 'NA FALA', 'a próxima palavra estava na fala', [110, 200, 190]);
    // the direction the probe reads: w · h + b
    const [a0, a1] = L.arrow.map(([X, Y]) => L.at(X, Y)), ang = Math.atan2(a1[1] - a0[1], a1[0] - a0[0]);
    x.strokeStyle = `rgba(217,226,234,${(0.55 * al).toFixed(3)})`; x.fillStyle = x.strokeStyle; x.lineWidth = 1.2;
    x.beginPath(); x.moveTo(a0[0], a0[1]); x.lineTo(a1[0], a1[1]); x.stroke();
    x.beginPath(); x.moveTo(a1[0], a1[1]); x.lineTo(a1[0] - Math.cos(ang - 0.4) * u * 0.9, a1[1] - Math.sin(ang - 0.4) * u * 0.9);
    x.lineTo(a1[0] - Math.cos(ang + 0.4) * u * 0.9, a1[1] - Math.sin(ang + 0.4) * u * 0.9); x.closePath(); x.fill();
    const mx = a0[0] + (a1[0] - a0[0]) * 0.34 + u * 1.2, my = a0[1] + (a1[1] - a0[1]) * 0.34;
    x.textAlign = 'left';
    this.halo(x, 'w · h + b', mx, my, `italic ${(u * 1.25).toFixed(1)}px "IBM Plex Mono", monospace`, `rgba(238,243,247,${(0.92 * al).toFixed(3)})`);
    this.halo(x, 'a leitura do probe', mx, my + u * 1.4, `${(u * 0.8).toFixed(1)}px "Work Sans", sans-serif`, `rgba(150,165,180,${(0.85 * al).toFixed(3)})`);
    // title, in the empty corner past the line
    x.textAlign = 'right'; x.fillStyle = `rgba(238,243,247,${(0.95 * al).toFixed(3)})`;
    x.font = `800 ${(u * 1.25).toFixed(1)}px "Plus Jakarta Sans", sans-serif`; x.fillText('O PLANO DO PROBE', W - u * 2.3, u * 3.3);
    x.fillStyle = `rgba(127,140,153,${al.toFixed(3)})`; x.font = `${(u * 0.82).toFixed(1)}px "Work Sans", sans-serif`;
    x.fillText(`camada 21 do Qwen3-4B · ${this.plane.n.toLocaleString('pt-BR')} palavras de resumos`, W - u * 2.3, u * 4.6);
  }
  // text with a dark rim, readable over the points
  halo(x, s, px, py, font, color) {
    x.font = font; x.lineJoin = 'round'; x.lineWidth = this.u * 0.4; x.strokeStyle = 'rgba(8,11,15,0.82)';
    x.strokeText(s, px, py); x.fillStyle = color; x.fillText(s, px, py);
  }

  // this line's words, one by one, and the reading that fired
  drawPath(x, T, t, out) {
    const { u, lay: L, info } = this;
    if (!info || !info.path) return;
    const pts = info.path.map(([X, Y, w]) => [...L.at(X, Y), w]);
    const here = L.at(info.here[0], info.here[1]);
    const t0 = 2.0, step = 0.24, shown = Math.floor((t - t0) / step) + 1;
    if (shown <= 0) return;
    x.save(); x.globalAlpha = out;
    x.strokeStyle = 'rgba(224,176,74,0.65)'; x.lineWidth = u * 0.1; x.beginPath();
    pts.slice(0, shown).forEach(([px, py], i) => (i ? x.lineTo(px, py) : x.moveTo(px, py))); x.stroke();
    const font = `${(u * 0.78).toFixed(1)}px "IBM Plex Mono", monospace`;
    x.font = font; x.textAlign = 'left'; x.textBaseline = 'middle';
    const right = here[0] < this.W - u * 30;   // the big label goes right of the point unless the screen ends
    const placed = [[right ? here[0] - u * 1.2 : here[0] - u * 29, here[1] - u * 2, u * 30.2, u * 4]];   // keep its place free
    pts.slice(0, shown).forEach(([px, py, w]) => {
      x.fillStyle = '#0f1419'; x.beginPath(); x.arc(px, py, u * 0.42, 0, 6.283); x.fill();
      x.fillStyle = '#e0b04a'; x.beginPath(); x.arc(px, py, u * 0.3, 0, 6.283); x.fill();
      const tw = x.measureText(w).width, th = u * 0.95;
      for (const [ox, oy] of [[0.6, -0.8], [0.6, 0.8], [-0.6, -0.8], [-0.6, 0.8]]) {   // first free spot around the point
        const lx = ox > 0 ? px + u * ox : px + u * ox - tw, ly = py + u * oy;
        const box = [lx - 2, ly - th / 2, tw + 4, th];
        if (placed.some(b => box[0] < b[0] + b[2] && b[0] < box[0] + box[2] && box[1] < b[1] + b[3] && b[1] < box[1] + box[3])) continue;
        placed.push(box); this.halo(x, w, lx, ly, font, 'rgba(240,210,140,0.9)'); break;
      }
    });
    if (shown > pts.length) {   // the jump over the line
      const last = pts[pts.length - 1], k = ramp(t, t0 + step * pts.length, t0 + step * pts.length + 0.5);
      if (last) {
        x.setLineDash([u * 0.35, u * 0.3]); x.strokeStyle = 'rgba(229,83,75,0.9)'; x.lineWidth = u * 0.12;
        x.beginPath(); x.moveTo(last[0], last[1]); x.lineTo(last[0] + (here[0] - last[0]) * k, last[1] + (here[1] - last[1]) * k); x.stroke(); x.setLineDash([]);
      }
      if (k >= 1 || !last) {
        const pulse = 0.5 + 0.5 * Math.sin(T * 6), ring = (T * 0.9) % 1;
        x.strokeStyle = `rgba(229,83,75,${(0.8 * (1 - ring)).toFixed(3)})`; x.lineWidth = u * 0.12;
        x.beginPath(); x.arc(here[0], here[1], u * (0.9 + 2.6 * ring), 0, 6.283); x.stroke();
        x.fillStyle = 'rgba(8,11,15,0.9)'; x.beginPath(); x.arc(here[0], here[1], u * 1.12, 0, 6.283); x.fill();   // dark rim: visible even on the line
        x.shadowColor = 'rgba(229,83,75,0.95)'; x.shadowBlur = u * (1.5 + pulse);
        x.fillStyle = '#e5534b'; x.beginPath(); x.arc(here[0], here[1], u * 0.82, 0, 6.283); x.fill(); x.shadowBlur = 0;
        x.strokeStyle = 'rgba(255,238,232,0.9)'; x.lineWidth = u * 0.1; x.stroke();
        x.fillStyle = '#fff'; x.beginPath(); x.arc(here[0], here[1], u * 0.28, 0, 6.283); x.fill();
        const dx = right ? u * 1.6 : -u * 1.6;
        x.textAlign = right ? 'left' : 'right'; x.textBaseline = 'alphabetic';
        this.halo(x, `ia escrever “${info.next}”`, here[0] + dx, here[1] - u * 0.35, `500 ${(u * 1.3).toFixed(1)}px "Work Sans", sans-serif`, '#f0c96b');
        this.halo(x, `${info.reading} ≥ ${info.tau}${info.novel ? ' · a palavra não estava na fala' : ''}`, here[0] + dx, here[1] + u * 1.15,
          `${(u * 0.95).toFixed(1)}px "IBM Plex Mono", monospace`, '#ff8f86');
      }
    }
    x.restore();
  }

  drawCaption(x, al) {
    const { u, W, H } = this, grey = '#a8b4bf';
    const rows = [   // two centred lines, so the right end stays free for "ESPAÇO continua"
      [['Cada caractere junta as palavras daquele ponto do plano, lidas um token antes de serem escritas:', grey], ['·', '#eef3f7'], ['uma,', grey], ['@', '#eef3f7'], ['muitas.', grey]],
      [['A cor vai de', grey], ['●', '#4fb3a9'], ['estavam na fala', grey], ['a', grey], ['●', '#e5534b'], ['não estavam (metade de cada).', grey], ['Passou da linha, o alarme toca.', '#eef3f7']],
    ];
    x.textBaseline = 'middle'; x.textAlign = 'left'; x.globalAlpha = al;
    x.font = `${(u * 0.95).toFixed(1)}px "Work Sans", sans-serif`;
    const gap = u * 0.4;
    rows.forEach((runs, r) => {
      const widths = runs.map(([t]) => x.measureText(t).width), total = widths.reduce((a, b) => a + b, 0) + gap * (runs.length - 1);
      let px = (W - total) / 2;
      const y = H - u * (3.9 - 1.5 * r);
      runs.forEach(([t, c], k) => { this.halo(x, t, px, y, x.font, c); px += widths[k] + gap; });
    });
    x.textAlign = 'right';
    this.halo(x, 'ESPAÇO continua', W - u * 2.3, H - u * 2.4, `600 ${(u * 0.8).toFixed(1)}px "Plus Jakarta Sans", sans-serif`, '#7f8c99');
    x.globalAlpha = 1;
  }

  // no SPACE: they lose their grip and fall
  drawFall(x, T) {
    const { t, W, H, N } = this, g = 2.8 * H;
    const dark = 0.42 * (1 - ramp(t, 0.2, 0.9));
    if (dark > 0) { x.fillStyle = `rgba(14,5,6,${dark.toFixed(3)})`; x.fillRect(0, 0, W, H); }
    x.globalCompositeOperation = 'lighter';
    let alive = 0;
    const flick = t < 0.14 && Math.random() < 0.5 ? 0.35 : 1;
    for (let k = 0; k < N; k++) {
      let px, py;
      if (t < this.dl[k]) { px = this.sx[k] + (Math.random() - 0.5) * 2; py = this.sy[k] + (Math.random() - 0.5) * 2; }
      else {
        const tt = t - this.dl[k];
        px = this.sx[k] + this.vx[k] * W * tt; py = this.sy[k] + this.vy[k] * H * tt + 0.5 * g * tt * tt;
      }
      if (py > H + 40) continue;
      alive++;
      const a = this.a[k], al = (0.3 + 0.55 * a) * flick * (1 - ramp(py, H * 0.82, H + 20) * 0.6);
      this.glyph(x, k, t < this.dl[k] + 0.25 ? RED : DIM, px, py, this.fs * (1.6 + 0.9 * a), al);
      this.x[k] = px; this.y[k] = py;
    }
    if (!alive || t > 2.2) this.phase = 'idle';
  }
}
