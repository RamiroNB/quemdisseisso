import { html, render, Component, getJSON, parseRoute, go } from './lib.js';
import { Home } from './home.js';
import { Hearing } from './hearing.js';
import { GameScreen } from './game.js';
import { Results } from './results.js';

class App extends Component {
  state = { route: parseRoute(), index: null, err: null };
  componentDidMount() {
    this.onHash = () => {
      const prev = this.state.route, route = parseRoute();
      this.setState({ route });
      const samePage = prev.screen === route.screen && prev.id === route.id;
      if (!samePage) window.scrollTo({ top: 0, behavior: 'instant' });   // a new page starts at its top, without the smooth scroll of html {}
    };
    window.addEventListener('hashchange', this.onHash);
    getJSON('data/index.json').then(index => this.setState({ index }), err => this.setState({ err: String(err) }));
  }
  componentWillUnmount() { window.removeEventListener('hashchange', this.onHash); }
  render(_, { route, index, err }) {
    if (route.screen === 'game' && index) return html`<${GameScreen} index=${index} q=${route.q} />`;   // the game takes the whole window
    let body;
    if (err) body = html`<div class="loading">Não foi possível carregar os dados (${err}).</div>`;
    else if (!index) body = html`<div class="loading">Carregando…</div>`;
    else if (route.screen === 'hearing') body = html`<${Hearing} key=${'h' + route.id} id=${route.id} person=${route.person} raw=${route.raw} q=${route.q} index=${index} />`;
    else if (route.screen === 'game') body = html`<${GameScreen} index=${index} q=${route.q} />`;
    else if (route.screen === 'results') body = html`<${Results} index=${index} />`;
    else body = html`<${Home} index=${index} q=${route.q} about=${route.about} />`;
    const home = () => go('#/');
    // the hash does not change on a second click, so scroll by hand when the section is already on the page
    const about = () => { const el = document.getElementById('como-funciona'); if (location.hash === '#/como-funciona' && el) el.scrollIntoView({ behavior: 'smooth' }); else go('#/como-funciona'); };
    return html`<div class="page">
      <header class="top">
        <div class="wordmark"><button onClick=${home}>Quem</button><button class="mid" onClick=${home}>Disse</button><button onClick=${home}>Isso</button></div>
        <nav class="nav" aria-label="Seções">
          <button onClick=${about}>Como funciona</button>
          <button class=${route.screen === 'results' ? 'on' : ''} onClick=${() => go('#/resultados')}>Resultados</button>
        </nav>
        <div class="team">
          <span>Ocarandu · PUCRS e UFMG</span>
          <span class="sep" aria-hidden="true"></span>
          <img class="kunumi" src="img/kunuminst.png" alt="Instituto Kunumi" title="Desafio Ideias em Rede, Instituto Kunumi" />
        </div>
      </header>
      ${body}
      <footer class="foot">
        <span>Equipe Ocarandu · Desafio Ideias em Rede, Instituto Kunumi · Dados: PublicHearingBR</span>
        <span>Tudo é pré-computado a partir de execuções reais; nada roda ao vivo.</span>
      </footer>
    </div>`;
  }
}

render(html`<${App} />`, document.getElementById('app'));
