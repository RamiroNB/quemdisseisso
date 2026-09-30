// Results page: the paper's four contributions for a general reader, in the paper's order. Numbers are quoted
// from the paper; the omission deciles and the retrieve-and-regenerate rates come from data/index.json, which
// scripts/export_portal_data.py checks against the paper.
import { html, Component, Icon, go, pct, pct1, int } from './lib.js';
import { Field } from './field.js';

const dec = (x, d) => x.toFixed(d).replace('.', ',');
const auc = v => dec(v, 3);

// opinion-level probe vs the benchmark's human labels, folds by hearing;
// the NLI cross-encoder; the probe trained on English RAGTruth only (Llama-3.1)
const PROBE = [
  { k: 'Llama-3.1-8B', v: 0.893, tone: 'ours' }, { k: 'Qwen2.5-7B', v: 0.878, tone: 'ours' },
  { k: 'Llama-2-7B', v: 0.874, tone: 'ours' }, { k: 'Qwen3-4B', v: 0.870, tone: 'ours' },
  { k: 'Contador de palavras', v: 0.822 }, { k: 'Verificador NLI', v: 0.766 },
];
const ZERO_SHOT = 0.823;
// rule-made corruptions of the pipeline's own text (Llama-3.1): share of pairs where the corrupted twin is
// scored less supported than its original (ties count as misses); examples are illustrative
const CORRUPT = [
  { k: 'Negação ou antônimo', ex: '“há um descompasso” → “não há um descompasso”', probe: 0.970, lex: 0.148 },
  { k: 'Número trocado', ex: '“12%” → “21%”', probe: 0.987, lex: 0.000 },
];
// RAGTruth, official test split, summarisation, response-level F1 (published rows: Niu et al., ACL 2024, Table 5)
const RAGTRUTH = [
  { k: 'Nosso probe · Llama-3.1-8B', v: 53.7, tone: 'ours' },
  { k: 'GPT-4-turbo com prompt', sub: 'publicado', v: 47.6 },
  { k: 'Contador de palavras novas', v: 41.9 },
  { k: 'SelfCheckGPT (GPT-3.5-turbo)', sub: 'publicado', v: 40.1 },
];
// pre-emptive token probe, read one position before each content word, and its same-pass competitors
const ALARM = [
  { k: 'Llama-3.1-8B', probe: 0.914, copy: 0.862, ent: 0.816 },
  { k: 'Qwen3-4B', probe: 0.905, copy: 0.879, ent: 0.808 },
];
// fixed/broken counts in the other two blind paired tests (Llama-3.1 / Sonnet; the same Qwen sets / Opus)
const RAG_OTHER = [{ k: 'Llama-3.1', fixed: 42, broken: 27 }, { k: 'Qwen3-4B com o juiz Opus', fixed: 59, broken: 46 }];
// withhold the fifth of opinions a detector ranks highest: share of the unsupported ones removed (the residual
// rate on human labels is quoted in the panel)
const ROUTE_BENCH = [{ k: 'Probe', v: 0.75, tone: 'ours' }, { k: 'Contador de palavras', v: 0.63 }];
const ROUTE_FRESH = [{ k: 'Probe', v: 32 / 50, tone: 'ours' }, { k: 'Contador de palavras', v: 21 / 50 }, { k: 'Verificador NLI', v: 21 / 50 }];
// a FALSE attribution endorsed under only one of the two printed titles (500 items per model)
const TITLE = [{ k: 'Llama-3.1-8B', parl: 30, civil: 13 }, { k: 'Qwen2.5-7B', parl: 42, civil: 19 }];

const Bars = ({ rows, lo = 0, hi = 1, fmt }) => html`<div class="rbars">
  ${rows.map(r => html`<div class=${'rbar ' + (r.tone || '')}>
    <span class="rk">${r.k}${r.sub && html`<small>${r.sub}</small>`}</span>
    <span class="rt"><i style=${{ width: Math.max(0, Math.min(1, (r.v - lo) / (hi - lo))) * 100 + '%' }}></i></span>
    <span class="rv">${fmt(r.v)}</span>
  </div>`)}
</div>`;

const Fig = ({ k, title, children }) => html`<article class="panel r-fig">
  <div class="k">${k}</div>
  <h3>${title}</h3>
  ${children}
</article>`;

const Section = ({ id, n, label, title, text, stats, children }) => html`<section class="rs" id=${id}>
  <div class="rs-text">
    <div class="pk"><span>${n}</span>${label}</div>
    <h2>${title}</h2>
    <p>${text}</p>
    <div class="rs-stats">${stats.map(s => html`<div class="pstat"><b>${s[0]}</b><span>${s[1]}</span></div>`)}</div>
  </div>
  <div class="rs-viz">${children}</div>
</section>`;

const toSec = id => { const el = document.getElementById(id); if (el) el.scrollIntoView({ behavior: 'smooth', block: 'start' }); };

export class Results extends Component {
  render({ index }) {
    const C = index.corpus, O = C.omission, R = C.rag, D = O.deciles;
    const ragDrop = html`${pct1(R.with_unsupported_baseline)}<i class="to">→</i>${pct1(R.with_unsupported_rag)}`;
    const omitDrop = html`${pct1(D[0])}<i class="to">→</i>${pct1(D[D.length - 1])}`;
    const INDEX = [
      ['r-1', '01', 'Detector depois da geração', auc(0.893), 'AUC contra os rótulos humanos do benchmark (Llama-3.1)'],
      ['r-2', '02', 'Alarme antes da palavra', '0,91', 'AUC um token antes de a palavra sem apoio ser escrita'],
      ['r-3', '03', 'Intervenção', ragDrop, 'das pessoas com alguma linha sem apoio na fala, segundo juízes cegos'],
      ['r-4', '04', 'Fairness', omitDrop, 'ficam fora da matéria: quem menos fala e quem mais fala'],
    ];
    return html`<main class="results" data-screen-label="Tela 5 · Resultados">
      <${Field} fixed />
      <div class="r-hero">
        <div class="eyebrow">Resultados do paper</div>
        <h1>Detectar, intervir, auditar</h1>
        <p class="tagline">Da leitura do modelo por dentro à auditoria de quem fica de fora</p>
        <p class="lead">Testamos o método em quatro modelos abertos, de 4 a 8 bilhões de parâmetros, nas audiências públicas da Câmara do PublicHearingBR. Cada número aparece ao lado de uma linha de base simples e foi medido em audiências separadas das usadas no treino.</p>
      </div>
      <nav class="r-index" aria-label="Contribuições">
        ${INDEX.map(([id, n, label, big, cap]) => html`<button onClick=${() => toSec(id)}>
          <span class="pk"><span>${n}</span>${label}</span>
          <span class="pstat"><b>${big}</b><span>${cap}</span></span>
        </button>`)}
      </nav>

      <${Section} id="r-1" n="01" label="Detector depois da geração" title="Um detector que lê o modelo por dentro, em português e em inglês"
        text=${`Depois que o resumo é escrito, um probe lê as ativações internas do modelo e aponta as opiniões sem apoio na fala. É uma regressão logística sobre uma única camada: roda em CPU, em qualquer computador. Ganha do contador de palavras e de um verificador NLI nos quatro modelos, percebe quando só o sentido muda e, treinado só em inglês, já funciona em português (AUC ${auc(ZERO_SHOT)} sem nenhum rótulo em português).`}
        stats=${[[auc(0.893), 'AUC contra os rótulos humanos do benchmark (Llama-3.1)'], [html`53,7 <small>vs 47,6</small>`, 'F1 no RAGTruth, benchmark de alucinação em inglês: nosso probe contra o GPT-4-turbo com prompt']]}>
        <${Fig} k=${`Rótulos humanos · ${int(4237)} opiniões`} title="Nos quatro modelos, acima das alternativas">
          <${Bars} lo=${0.5} hi=${1} fmt=${auc} rows=${PROBE} />
          <p class="r-note">ROC-AUC, de 0,5 (acaso) a 1. Opiniões do PublicHearingBR rotuladas por pessoas, avaliadas em audiências fora do treino.</p>
        <//>
        <${Fig} k="Erros sutis" title="Ele percebe quando só o sentido muda">
          ${CORRUPT.map(c => html`<div class="rgroup">
            <div class="rgroup-h">${c.k}<small>${c.ex}</small></div>
            <${Bars} fmt=${pct} rows=${[{ k: 'Probe', v: c.probe, tone: 'ours' }, { k: 'Contador de palavras', v: c.lex }]} />
          </div>`)}
          <p class="r-note">Em que fração dos pares o detector aponta a versão adulterada de uma linha real. Numa negação as palavras continuam as mesmas, e o contador não vê diferença.</p>
        <//>
        <${Fig} k="Benchmark de alucinação · RAGTruth" title="Em inglês, acima do GPT-4-turbo">
          <${Bars} lo=${0} hi=${60} fmt=${v => dec(v, 1)} rows=${RAGTRUTH} />
          <p class="r-note">O RAGTruth (ACL 2024) é um benchmark de detecção de alucinação em inglês. F1 na tarefa de resumo, divisão oficial de teste: 900 respostas com alucinações marcadas por pessoas. Só um detector treinado especificamente para a tarefa fica acima (59,1).</p>
        <//>
      <//>

      <${Section} id="r-2" n="02" label="Alarme antes da palavra" title="Um alarme que toca antes de a IA escrever o que não foi dito"
        text="Enquanto o modelo escreve, o probe lê o estado interno uma posição antes de cada palavra de conteúdo e prevê se ela vai estar fora da fala da pessoa. O aviso chega antes do texto: dá para agir durante a escrita, e não só depois."
        stats=${[['0,91', 'AUC um token antes: 0,914 no Llama-3.1 e 0,905 no Qwen3-4B'], ['0,95', 'AUC para números inventados, antes de serem escritos (Llama-3.1)']]}>
        <${Fig} k=${`Um token antes · ${int(3021)} resumos`} title="Acima dos sinais que o próprio modelo já dá">
          ${ALARM.map(m => html`<div class="rgroup">
            <div class="rgroup-h">${m.k}</div>
            <${Bars} lo=${0.5} hi=${1} fmt=${auc} rows=${[{ k: 'Probe', v: m.probe, tone: 'ours' }, { k: 'Massa de cópia', v: m.copy }, { k: 'Entropia', v: m.ent }]} />
          </div>`)}
          <p class="r-note">ROC-AUC para “a próxima palavra de conteúdo não está na fala”. Os dois concorrentes vêm da mesma passada pelo modelo: a massa de cópia é a probabilidade que o modelo atribui a tokens presentes na fala da pessoa, e a entropia mede a incerteza da distribuição do próximo token.</p>
        <//>
        <div class="r-game">
          <p>No jogo, você faz o papel do alarme numa linha real e tenta apertar antes de o modelo sair da fala.</p>
          <button class="gamebtn compact" onClick=${() => go('#/jogo?de=inicio')}>
            <span class="l"><span class="a">Jogo</span><span class="b">Seja a verdade</span></span>
            <span class="r"><${Icon} name="arrow-right" size=${20} /></span>
          </button>
        </div>
      <//>

      <${Section} id="r-3" n="03" label="Intervenção" title="Prever a alucinação e intervir a tempo"
        text=${`Testamos duas formas de agir sobre o que o detector prevê. Sem reescrever nada: segurar para revisão os 20% de opiniões que ele marca, o que tira três quartos das alucinações. Reescrevendo: quando o alarme toca numa linha, buscamos os trechos da fala da própria pessoa e geramos a linha de novo. O alarme decide quando buscar; a busca na fala corrige. Juízes automáticos cegos compararam os resumos de ${R.participants} pessoas com e sem essa reescrita, com a mesma fala ao lado e sem saber qual versão era qual.`}
        stats=${[['75%', 'sem reescrever: alucinações que saem ao segurar os 20% de opiniões que o probe marca (rótulos humanos)'], [ragDrop, 'reescrevendo: pessoas com alguma linha sem apoio na fala, antes e depois da busca e da nova geração (Qwen3-4B)']]}>
        <${Fig} k="Sem reescrever · revisão do que o probe marca" title="Segurar o que o probe marca tira 75% das alucinações">
          <div class="r-ba">
            <div><span class="bk">Sem filtro</span><b>11,9%</b></div>
            <span class="ar" aria-hidden="true">→</span>
            <div class="ours"><span class="bk">Segurando 20%</span><b>3,7%</b></div>
          </div>
          <div class="rgroup">
            <div class="rgroup-h">Rótulos humanos do benchmark<small>alucinações que saem ao segurar os 20% de opiniões mais suspeitas</small></div>
            <${Bars} fmt=${pct} rows=${ROUTE_BENCH} />
          </div>
          <div class="rgroup">
            <div class="rgroup-h">Nossos resumos (Llama-3.1), juízes cegos<small>linhas sem apoio que saem do mesmo jeito (50 no total)</small></div>
            <${Bars} fmt=${pct} rows=${ROUTE_FRESH} />
          </div>
          <p class="r-note">Taxa de alucinação das opiniões que ficam, nos rótulos humanos. Nada é reescrito, então nada novo pode sair errado.</p>
        <//>
        <${Fig} k=${`Reescrevendo · busca na fala e nova geração · Qwen3-4B, ${R.participants} pessoas`} title="Mais resumos consertados">
          <div class="r-ba">
            <div><span class="bk">Sem intervenção</span><b>${pct1(R.with_unsupported_baseline)}</b></div>
            <span class="ar" aria-hidden="true">→</span>
            <div class="ours"><span class="bk">Com a reescrita</span><b>${pct1(R.with_unsupported_rag)}</b></div>
          </div>
          <${Bars} lo=${0} hi=${40} fmt=${v => String(v)} rows=${[{ k: 'Resumos consertados', v: R.fixed, tone: 'ours' }, { k: 'Resumos estragados', v: R.broken }]} />
          <p class="r-note">Pessoas com alguma linha sem apoio na fala, antes e depois de buscar na fala e gerar de novo as linhas em que o alarme tocou. A mesma direção aparece ${RAG_OTHER.map((t, i) => `${i ? ' e no ' : 'no '}${t.k} (${t.fixed} contra ${t.broken})`).join('')}.</p>
        <//>
      <//>

      <${Section} id="r-4" n="04" label="Fairness" title="Quem a mediação deixa de fora, e em quem a IA acredita"
        text="Medimos quem a matéria publicada deixa de citar em 206 audiências e se os modelos acreditam mais numa fala conforme o cargo de quem falou. O que decide quem aparece na matéria é o tempo de fala."
        stats=${[[omitDrop, 'ficam fora da matéria: os 10% que menos falam e os 10% que mais falam'], ['90 a 92%', 'das atribuições erradas são confirmadas pelos modelos, mesmo com o nome de quem falou no texto']]}>
        <${Fig} k=${`Omissão · 206 audiências · ${int(O.n)} pessoas`} title="A matéria comprime e cita quem fala mais">
          <div class="r-cols" role="img" aria-label=${'Chance de ficar fora da matéria por décimo de tempo de fala: ' + D.map(pct).join(', ')}>
            ${D.map((d, i) => html`<div class="r-col"><span class="v">${pct(d)}</span><span class="cw"><span class="o" style=${{ flex: d }}></span><span class="c" style=${{ flex: 1 - d }}></span></span><span class="x">${i + 1}º</span></div>`)}
          </div>
          <div class="r-axis"><span>← quem menos fala</span><span>quem mais fala →</span></div>
          <div class="flegend"><span><i style=${{ background: '#4B68E3' }}></i>citada na matéria</span><span><i style=${{ background: '#B5BEC7' }}></i>não citada (o número acima)</span></div>
          <p class="r-note">Chance de ficar fora da matéria da Agência Câmara, por décimo de tempo de fala. Com o mesmo tempo de fala, só quem preside a sessão é citado com mais frequência (fica de fora ${pct(O.chair_matched)} das vezes, contra ${pct(O.rest_matched)} dos demais).</p>
        <//>
        <${Fig} k="O peso do cargo" title="Um cargo de parlamentar compra um pouco de crença">
          ${TITLE.map(t => html`<div class="rgroup">
            <div class="rgroup-h">${t.k}<small>atribuições falsas aceitas só com um dos dois cargos</small></div>
            <${Bars} lo=${0} hi=${45} fmt=${v => String(v)} rows=${[{ k: 'Cargo de parlamentar', v: t.parl, tone: 'ours' }, { k: 'Cargo de sociedade civil', v: t.civil }]} />
          </div>`)}
          <p class="r-note">Mesma fala, mesma evidência, mesma pessoa: só o cargo impresso muda.</p>
        <//>
        <${Fig} k="Quem disse o quê" title="Os modelos veem quem falou e não usam essa informação">
          <div class="fgrid">
            <b>O teste</b><span>Mostramos uma fala com o nome de quem a disse e perguntamos se foi outra pessoa da mesma audiência. Os modelos respondem “sim” em 90 a 92% dos casos, mesmo com o nome certo no texto.</span>
            <b>Limiar</b><span>A informação está lá dentro: com um único limiar calibrado em outras audiências, o acerto do Qwen2.5 vai de 0,53 para 0,935.</span>
            <b>Instrução</b><span>Pedindo o que disse alguém que não falou, os modelos inventam. Uma instrução de uma linha reduz as falas inventadas de 14,7% para 1,0% (Llama-3.1).</span>
          </div>
        <//>
      <//>

      <section class="r-how">
        <p>Dados: PublicHearingBR (Unicamp), 206 audiências públicas da Câmara, com as transcrições oficiais e as matérias da Agência Câmara. Modelos abertos: Llama-2-7B, Llama-3.1-8B, Qwen2.5-7B e Qwen3-4B.</p>
        <div class="r-cta">
          <button class="about-cta" onClick=${() => go('#/audiencia/38')}>Ver o método numa audiência <${Icon} name="arrow-right" size=${16} /></button>
          <button class="about-cta ghost" onClick=${() => go('#/')}>Explorar as audiências <${Icon} name="arrow-right" size=${16} /></button>
        </div>
      </section>
    </main>`;
  }
}
