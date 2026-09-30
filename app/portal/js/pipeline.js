// Início · "Como funciona": the three steps drawn as small moving networks, in the style of the page's background
// particles. 1 a probe lens drifts over a cloud of activations and an alarm lights red; 2 the alarm travels to the
// speech's excerpts, the two nearest link to it and it turns blue (the search in the speech corrects the line);
// 3 speakers sized by speaking time, the larger ones linked to the article (a hub node). No text on the canvas. Decorative only: every claim and number
// is in the text beside it. One canvas covers the three columns; each step draws into its own .pviz box.
import { html, Component } from './lib.js';

const IND = '75,104,227', BLUE = '0,136,255', RED = '229,83,75', TEAL = '79,179,169', INK = '20,24,28';
const rnd = (a, b) => a + Math.random() * (b - a);
const ease = s => s < 0.5 ? 2 * s * s : 1 - Math.pow(-2 * s + 2, 2) / 2;
const mix = (a, b, s) => a.map((v, i) => Math.round(v + (b[i] - v) * s)).join(',');
const cloud = (n, r0, r1) => Array.from({ length: n }, () => ({ x: rnd(0.08, 0.92), y: rnd(0.1, 0.9), vx: rnd(-1, 1) * 0.025, vy: rnd(-1, 1) * 0.025, r: rnd(r0, r1) }));

export class Pipeline extends Component {
  boxes = [cloud(30, 0.9, 2.2), cloud(20, 1.3, 2.5),
    // speakers: a few with long speeches (large, linked to the article), most short
    Array.from({ length: 15 }, (_, i) => ({ x: rnd(0.06, 0.6), y: rnd(0.12, 0.88), vx: rnd(-1, 1) * 0.015, vy: rnd(-1, 1) * 0.015, r: i < 4 ? rnd(4.6, 7) : rnd(1.4, 3.4), cited: i < 4 || i === 6 }))];
  t = 0; nextAlarm = 1.0; alarm = null; packets = []; mx = -9999; my = -9999;

  componentDidMount() {
    this.onMove = e => { this.mx = e.clientX; this.my = e.clientY; };
    window.addEventListener('mousemove', this.onMove);
    this.still = window.matchMedia && matchMedia('(prefers-reduced-motion: reduce)').matches;
    if (this.still) {   // one representative frame: the alarm lit, one line being corrected
      for (let k = 0; k < 150; k++) this.step(1 / 30);
      this.draw();
      this.onResize = () => this.draw(); window.addEventListener('resize', this.onResize);
      return;
    }
    let last = performance.now();
    const loop = now => { const dt = Math.min(0.05, (now - last) / 1000); last = now; if (this.visible !== false) { this.step(dt); this.draw(); } this.raf = requestAnimationFrame(loop); };
    this.raf = requestAnimationFrame(loop);
    if ('IntersectionObserver' in window) {
      this.io = new IntersectionObserver(es => { this.visible = es[0].isIntersecting; });
      this.io.observe(this.canvas);
    }
  }
  componentWillUnmount() {
    cancelAnimationFrame(this.raf); window.removeEventListener('mousemove', this.onMove);
    if (this.onResize) window.removeEventListener('resize', this.onResize);
    if (this.io) this.io.disconnect();
  }

  lens(t) { return { x: 0.5 + 0.3 * Math.sin(t * 0.42), y: 0.5 + 0.22 * Math.sin(t * 0.67 + 1) }; }

  step(dt) {
    this.t += dt;
    const t = this.t;
    for (const pts of this.boxes) for (const p of pts) {
      p.x += p.vx * dt; p.y += p.vy * dt;
      if (p.x < 0.05 || p.x > (pts === this.boxes[2] ? 0.64 : 0.95)) p.vx *= -1;
      if (p.y < 0.08 || p.y > 0.92) p.vy *= -1;
    }
    if (!this.alarm && t > this.nextAlarm) {   // the particle closest to the lens is about to leave the source
      const L = this.lens(t);
      let best = 0, bd = 9;
      this.boxes[0].forEach((p, i) => { const d = Math.hypot(p.x - L.x, (p.y - L.y) * 0.6); if (d < bd) { bd = d; best = i; } });
      this.alarm = { i: best, t0: t, sent: false };
      this.nextAlarm = t + 3.4;
    }
    if (this.alarm) {
      const age = t - this.alarm.t0;
      if (age > 1.3) { const p = this.boxes[0][this.alarm.i]; this.packets.push({ t0: t, x0: p.x, y0: p.y }); this.alarm = null; }   // the red dot leaves the cloud
    }
    this.packets = this.packets.filter(k => t - k.t0 < 5.2);
  }

  draw() {
    const cv = this.canvas; if (!cv || !this.host) return;
    const rect = cv.getBoundingClientRect(), W = rect.width, H = rect.height, dpr = window.devicePixelRatio || 1;
    if (!W || !H) return;
    if (cv.width !== Math.round(W * dpr) || cv.height !== Math.round(H * dpr)) { cv.width = Math.round(W * dpr); cv.height = Math.round(H * dpr); }
    const x = cv.getContext('2d'); x.setTransform(dpr, 0, 0, dpr, 0, 0); x.clearRect(0, 0, W, H);
    const B = Array.from(this.host.querySelectorAll('.pviz')).map(el => { const r = el.getBoundingClientRect(); return { x: r.left - rect.left, y: r.top - rect.top, w: r.width, h: r.height }; });
    if (B.length < 3) return;
    const t = this.t, mx = this.mx - rect.left, my = this.my - rect.top;
    const at = (b, p) => {   // box coordinates → canvas, pushed away from the pointer like the background particles
      let px = b.x + p.x * b.w, py = b.y + p.y * b.h;
      const dx = px - mx, dy = py - my, d = Math.hypot(dx, dy) || 1;
      if (d < 110) { const f = (1 - d / 110) * 22; px += dx / d * f; py += dy / d * f; }
      return [px, py];
    };
    const row = Math.abs(B[0].y - B[1].y) < 4;   // side by side (desktop) or stacked (phone)

    B.forEach(b => {   // a soft light behind each network
      const g = x.createRadialGradient(b.x + b.w / 2, b.y + b.h / 2, 0, b.x + b.w / 2, b.y + b.h / 2, Math.max(b.w, b.h) * 0.55);
      g.addColorStop(0, 'rgba(255,255,255,0.75)'); g.addColorStop(1, 'rgba(255,255,255,0)');
      x.fillStyle = g; x.fillRect(b.x - 40, b.y - 40, b.w + 80, b.h + 80);
    });
    const web = (P, rgb, reach, a) => {
      x.lineWidth = 1;
      for (let i = 0; i < P.length; i++) for (let j = i + 1; j < P.length; j++) {
        const d = Math.hypot(P[i][0] - P[j][0], P[i][1] - P[j][1]);
        if (d < reach) { x.strokeStyle = `rgba(${rgb},${((1 - d / reach) * a).toFixed(3)})`; x.beginPath(); x.moveTo(P[i][0], P[i][1]); x.lineTo(P[j][0], P[j][1]); x.stroke(); }
      }
    };

    // flow between the steps: dashed, moving left to right
    if (row) {
      x.save(); x.setLineDash([3, 6]); x.lineDashOffset = -t * 18; x.lineWidth = 1.2; x.strokeStyle = `rgba(${IND},0.28)`;
      for (let k = 0; k < 2; k++) { const a = B[k], b = B[k + 1]; x.beginPath(); x.moveTo(a.x + a.w * 0.9, a.y + a.h / 2); x.quadraticCurveTo((a.x + a.w + b.x) / 2, a.y + a.h / 2 - 26, b.x + b.w * 0.1, b.y + b.h / 2); x.stroke(); }
      x.restore();
    }

    // 1 · activations and the probe's lens
    const b0 = B[0], P0 = this.boxes[0].map(p => at(b0, p));
    web(P0, IND, 64, 0.16);
    const L = this.lens(t), lx = b0.x + L.x * b0.w, ly = b0.y + L.y * b0.h, R = Math.min(b0.w, b0.h) * 0.3;
    const lg = x.createRadialGradient(lx, ly, 0, lx, ly, R); lg.addColorStop(0, `rgba(${BLUE},0.10)`); lg.addColorStop(1, `rgba(${BLUE},0.02)`);
    x.fillStyle = lg; x.beginPath(); x.arc(lx, ly, R, 0, 7); x.fill();
    x.strokeStyle = `rgba(${BLUE},0.45)`; x.lineWidth = 1.2; x.stroke();
    P0.forEach(([px, py], i) => {
      const d = Math.hypot(px - lx, py - ly), inside = d < R, p = this.boxes[0][i];
      if (inside) { x.strokeStyle = `rgba(${BLUE},${((1 - d / R) * 0.4).toFixed(3)})`; x.lineWidth = 1; x.beginPath(); x.moveTo(lx, ly); x.lineTo(px, py); x.stroke(); }
      x.fillStyle = inside ? `rgba(${IND},0.85)` : (i % 2 ? `rgba(${IND},0.3)` : `rgba(${INK},0.16)`);
      x.beginPath(); x.arc(px, py, p.r * (inside ? 1.25 : 1), 0, 7); x.fill();
    });
    if (this.alarm) {
      const [ax, ay] = P0[this.alarm.i], age = t - this.alarm.t0;
      for (const off of [0, 0.5]) { const s = ((age + off) % 1); x.strokeStyle = `rgba(${RED},${(0.55 * (1 - s)).toFixed(3)})`; x.lineWidth = 1.5; x.beginPath(); x.arc(ax, ay, 5 + s * 20, 0, 7); x.stroke(); }
      x.fillStyle = `rgb(${RED})`; x.beginPath(); x.arc(ax, ay, 4, 0, 7); x.fill();
    }

    // 2 · the speech's excerpts
    const b1 = B[1], P1 = this.boxes[1].map(p => at(b1, p));
    web(P1, TEAL, 60, 0.2);
    P1.forEach(([px, py], i) => { x.fillStyle = `rgba(${TEAL},0.55)`; x.beginPath(); x.arc(px, py, this.boxes[1][i].r, 0, 7); x.fill(); });

    // 3 · speakers by speaking time and the article
    const b2 = B[2], P2 = this.boxes[2].map(p => at(b2, p));
    const dx = b2.x + b2.w * 0.84, dy = b2.y + b2.h * 0.48;
    P2.forEach(([px, py], i) => { if (this.boxes[2][i].cited) { x.strokeStyle = `rgba(${IND},0.22)`; x.lineWidth = 1; x.beginPath(); x.moveTo(px, py); x.lineTo(dx, dy); x.stroke(); } });
    P2.forEach(([px, py], i) => { const s = this.boxes[2][i]; x.fillStyle = s.cited ? `rgba(${IND},0.6)` : `rgba(${INK},0.2)`; x.beginPath(); x.arc(px, py, s.r, 0, 7); x.fill(); });
    const pulse = (t * 0.6) % 1;   // the article: a hub the cited speakers link into
    x.strokeStyle = `rgba(${IND},${(0.3 * (1 - pulse)).toFixed(3)})`; x.lineWidth = 1; x.beginPath(); x.arc(dx, dy, 10 + pulse * 16, 0, 7); x.stroke();
    x.strokeStyle = `rgba(${IND},0.45)`; x.lineWidth = 1.2; x.beginPath(); x.arc(dx, dy, 10, 0, 7); x.stroke();
    x.fillStyle = `rgba(${IND},0.9)`; x.beginPath(); x.arc(dx, dy, 5, 0, 7); x.fill();

    // the alarms travelling through the pipeline
    const red = [229, 83, 75], blue = [0, 136, 255];
    for (const k of this.packets) {
      const age = t - k.t0;
      const a0 = [b0.x + k.x0 * b0.w, b0.y + k.y0 * b0.h], in1 = [b1.x + b1.w * 0.1, b1.y + b1.h / 2], fix = [b1.x + b1.w * 0.42, b1.y + b1.h * 0.5];
      const out1 = [b1.x + b1.w * 0.9, b1.y + b1.h / 2], in2 = [b2.x + b2.w * 0.1, b2.y + b2.h / 2];
      let px, py, col = red, alpha = 1, links = 0;
      const T1 = row ? 1.3 : 0, T2 = 1.9, T3 = row ? 1.3 : 0;
      if (age < T1) { const s = ease(age / T1); px = a0[0] + (in1[0] - a0[0]) * s; py = a0[1] + (in1[1] - a0[1]) * s - Math.sin(Math.PI * s) * 24; }
      else if (age < T1 + T2) {   // the search: two excerpts link to the line and it turns blue
        const u = (age - T1) / T2, s = ease(Math.min(1, u * 1.6));
        px = in1[0] + (fix[0] - in1[0]) * s; py = in1[1] + (fix[1] - in1[1]) * s;
        links = Math.min(1, Math.max(0, (u - 0.25) / 0.3)) * (u > 0.85 ? (1 - u) / 0.15 : 1);
        col = mix(red, blue, Math.min(1, Math.max(0, (u - 0.45) / 0.35))).split(',').map(Number);
      } else if (age < T1 + T2 + T3) { const s = ease((age - T1 - T2) / T3); px = fix[0] + (in2[0] - fix[0]) * s; py = fix[1] + (in2[1] - fix[1]) * s - Math.sin(Math.PI * s) * 24; col = blue; }
      else { const u = (age - T1 - T2 - T3) / 0.8; if (u > 1) continue; px = in2[0] + (dx - in2[0]) * ease(u) * 0.35; py = in2[1]; col = blue; alpha = 1 - u; }
      if (links > 0) {
        const near = P1.map((q, i) => [Math.hypot(q[0] - px, q[1] - py), i]).sort((a, b) => a[0] - b[0]).slice(0, 2);
        near.forEach(([, i]) => { x.strokeStyle = `rgba(${TEAL},${(0.8 * links).toFixed(3)})`; x.lineWidth = 1.6; x.beginPath(); x.moveTo(px, py); x.lineTo(P1[i][0], P1[i][1]); x.stroke();
          x.fillStyle = `rgba(${TEAL},${(0.9 * links).toFixed(3)})`; x.beginPath(); x.arc(P1[i][0], P1[i][1], 3.2, 0, 7); x.fill(); });
      }
      x.fillStyle = `rgba(${col.join(',')},${alpha.toFixed(3)})`; x.beginPath(); x.arc(px, py, 4, 0, 7); x.fill();
      x.strokeStyle = `rgba(${col.join(',')},${(0.25 * alpha).toFixed(3)})`; x.lineWidth = 5; x.beginPath(); x.arc(px, py, 7, 0, 7); x.stroke();
    }
  }

  render({ children }) {
    return html`<div class="psteps" ref=${el => { this.host = el; }}>
      <canvas ref=${el => { this.canvas = el; }} aria-hidden="true"></canvas>
      ${children}
    </div>`;
  }
}
