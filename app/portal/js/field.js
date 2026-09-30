// The designer's particle field: drifting points linked when close, pushed away from the pointer and linked to it.
// The home page draws it behind the hero and the carousel; the results page pins it to the window (`fixed`).
import { html, Component } from './lib.js';

export class Field extends Component {
  tx = -9999; ty = -9999;
  componentDidMount() {
    this.onMove = e => { this.tx = e.clientX; this.ty = e.clientY; };
    window.addEventListener('mousemove', this.onMove);
    let last = performance.now();
    const loop = t => { const dt = Math.min(0.05, (t - last) / 1000); last = t; this.draw(dt); this.raf = requestAnimationFrame(loop); };
    this.raf = requestAnimationFrame(loop);
  }
  componentWillUnmount() { cancelAnimationFrame(this.raf); window.removeEventListener('mousemove', this.onMove); }

  draw(dt) {
    const cv = this.canvas; if (!cv) return;
    const rect = cv.getBoundingClientRect(), W = rect.width, H = rect.height, dpr = window.devicePixelRatio || 1;
    if (!W || !H) return;
    if (cv.width !== Math.round(W * dpr) || cv.height !== Math.round(H * dpr)) { cv.width = Math.round(W * dpr); cv.height = Math.round(H * dpr); }
    const x = cv.getContext('2d'); x.setTransform(dpr, 0, 0, dpr, 0, 0); x.clearRect(0, 0, W, H);
    if (!this.parts) { this.parts = []; for (let i = 0; i < 90; i++) this.parts.push({ x: Math.random(), y: Math.random(), vx: (Math.random() - .5) * 0.01, vy: (Math.random() - .5) * 0.01, r: 0.8 + Math.random() * 1.6, b: Math.random() < 0.5 }); }
    const mx = this.tx - rect.left, my = this.ty - rect.top, inside = mx > 0 && my > 0 && mx < W && my < H;
    const P = this.parts.map(p => {
      p.x = (p.x + p.vx * dt + 1) % 1; p.y = (p.y + p.vy * dt + 1) % 1;
      let px = p.x * W, py = p.y * H;
      if (inside) { const dx = px - mx, dy = py - my, d = Math.hypot(dx, dy) || 1; if (d < 150) { const f = (1 - d / 150) * 34; px += dx / d * f; py += dy / d * f; } }
      return [px, py, p];
    });
    x.lineWidth = 1;
    for (let i = 0; i < P.length; i++) for (let j = i + 1; j < P.length; j++) {
      const dx = P[i][0] - P[j][0], dy = P[i][1] - P[j][1], d2 = dx * dx + dy * dy;
      if (d2 < 12100) { x.strokeStyle = 'rgba(75,104,227,' + ((1 - Math.sqrt(d2) / 110) * 0.1).toFixed(3) + ')'; x.beginPath(); x.moveTo(P[i][0], P[i][1]); x.lineTo(P[j][0], P[j][1]); x.stroke(); }
    }
    if (inside) P.forEach(([px, py]) => { const d = Math.hypot(px - mx, py - my); if (d < 190) { x.strokeStyle = 'rgba(0,136,255,' + ((1 - d / 190) * 0.18).toFixed(3) + ')'; x.beginPath(); x.moveTo(mx, my); x.lineTo(px, py); x.stroke(); } });
    P.forEach(([px, py, p]) => { x.fillStyle = p.b ? 'rgba(75,104,227,0.28)' : 'rgba(20,24,28,0.12)'; x.beginPath(); x.arc(px, py, p.r, 0, 7); x.fill(); });
  }
  render({ fixed }) { return html`<canvas class=${'field' + (fixed ? ' fixed' : '')} ref=${el => { this.canvas = el; }} aria-hidden="true"></canvas>`; }
}
