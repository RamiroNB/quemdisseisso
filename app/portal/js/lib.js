// Shared helpers. Preact + htm are vendored (vendor/preact-htm.module.js), so the site needs no build step.
import { html } from '../vendor/preact-htm.module.js';
export { h, html, render, Component } from '../vendor/preact-htm.module.js';

export const Icon = ({ name, size = 18, color, style }) => html`<span class="ico" aria-hidden="true"
  style=${{ width: size + 'px', height: size + 'px', color, WebkitMaskImage: `url(icons/${name}.svg)`, maskImage: `url(icons/${name}.svg)`, ...(style || {}) }}></span>`;

const cache = {};
// a load that fails or hangs for 20 s is dropped from the cache, so calling again retries it
export function getJSON(url) {
  if (!cache[url]) {
    const ac = new AbortController(), timer = setTimeout(() => ac.abort(), 20000);
    cache[url] = fetch(url, { signal: ac.signal })
      .then(r => { if (!r.ok) throw new Error(url + ' ' + r.status); return r.json(); })
      .catch(e => { delete cache[url]; throw e.name === 'AbortError' ? new Error('tempo esgotado ao carregar ' + url) : e; })
      .finally(() => clearTimeout(timer));
  }
  return cache[url];
}

// pt-BR numbers: 0,984 · 54% · 2.390
export const p3 = x => x.toFixed(3).replace('.', ',');
export const pct = x => Math.round(x * 100) + '%';
// one decimal, rounding exact halves to even like Python's format(), so 1.25% reads 1,2%
export const pct1 = x => { const v = x * 1000, f = Math.floor(v + 1e-9), d = v - f; const r = Math.abs(d - 0.5) < 1e-6 ? (f % 2 ? f + 1 : f) : Math.round(v); return (r / 10).toFixed(1).replace('.', ',') + '%'; };
export const int = x => x.toLocaleString('pt-BR');
// the probe's gauge: linear up to 0.9, then log-odds up to 0.999, so the threshold (0.978) is visibly apart from 0.99
const lg = x => Math.log(x / (1 - x));
export function gpos(p) {
  if (p <= 0.9) return 0.45 * (p / 0.9);
  return 0.45 + 0.55 * Math.max(0, Math.min(1, (lg(Math.min(p, 0.99999)) - lg(0.9)) / (lg(0.999) - lg(0.9))));
}
export const ordinal = n => n + 'º';

const TITLES = /^(Professora?|Prof\.?|Dra?\.?|Doutora?|Coronel|General|Delegad[oa]|Padre|Pastora?|Cabo|Sargento|Capitão|Major|Ministr[oa]|Deputad[oa]|Senador[a]?)$/i;
export function initials(name) {
  const p = name.split(/\s+/).filter(w => !TITLES.test(w)).filter(w => w.length > 2 || /^[A-ZÀ-Ú]/.test(w));
  if (!p.length) return name.slice(0, 2).toUpperCase();
  return (p[0][0] + (p.length > 1 ? p[p.length - 1][0] : '')).toUpperCase();
}
const PARTICLE = /^(de|da|do|dos|das|e|van|von|del|di|du)$/i;
export const shortName = name => { const w = name.split(/\s+/); return w.slice(0, w.length > 2 && PARTICLE.test(w[1]) ? 3 : 2).join(' '); };
export const firstName = name => (name.split(/\s+/).find(w => !TITLES.test(w)) || name.split(/\s+/)[0]);

// #/ · #/como-funciona · #/resultados · #/audiencia/38 · #/audiencia/38/daniel-panizzi[/sem-correcao] · #/jogo?de=38&auto=1
export function parseRoute(hash = location.hash) {
  const [path, query = ''] = hash.replace(/^#/, '').split('?');
  const q = Object.fromEntries(new URLSearchParams(query));
  const seg = path.split('/').filter(Boolean);
  if (seg[0] === 'audiencia' && seg[1]) return { screen: 'hearing', id: +seg[1], person: seg[2] || null, raw: seg[3] === 'sem-correcao', q };
  if (seg[0] === 'jogo') return { screen: 'game', q };
  if (seg[0] === 'resultados') return { screen: 'results', q };
  if (seg[0] === 'como-funciona') return { screen: 'home', about: true, q };   // the home page, scrolled to its "como funciona" section
  return { screen: 'home', q };
}
export const go = hash => { if (location.hash !== hash) location.hash = hash; };
export const replaceRoute = hash => { if (location.hash !== hash) history.replaceState(null, '', hash); };
