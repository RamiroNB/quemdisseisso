"""Speaker-identity attributes of PublicHearingBR speakers.

Pure functions only (no network): parse what the free-text `cargo`
string and the `nome` string give -- parliamentarian status, party
acronym, UF, grammatical-gender marking of the job title (and a flip of
it for the title-swap experiment), and a coarse institutional-role
taxonomy. Network enrichment (Camara open-data API for deputies,
pt.wikipedia presence as an entity-popularity proxy) lives in
scripts/derive_speaker_attributes.py.

All attributes are derived by rules, not annotated by the dataset
authors; tables built on them should say so and report coverage.
"""

import re
import unicodedata

PARTIES = [
    "PT", "PL", "PSOL", "PSDB", "MDB", "PP", "PSD", "PDT", "PSB", "UNIAO", "REPUBLICANOS",
    "PODE", "PCDOB", "NOVO", "PV", "REDE", "AVANTE", "SOLIDARIEDADE", "CIDADANIA", "PATRIOTA",
    "PROS", "PTB", "PSC", "PMN", "DC", "AGIR", "PRTB", "PRD", "PMB", "PSL", "DEM", "PRB", "PHS",
]
_PARTY_ALIASES = {"UNIÃO": "UNIAO", "UNIÃO BRASIL": "UNIAO", "UNIAO BRASIL": "UNIAO", "PODEMOS": "PODE",
                  "PCDOB": "PCDOB", "PC DO B": "PCDOB", "PCdoB": "PCDOB", "REPUBLICANO": "REPUBLICANOS"}

ROLE_TAXONOMY = ("parliamentarian", "state", "civil_society", "private_sector", "academia", "other")


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text or "")
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", text).strip().lower()


def is_parliamentarian(cargo: str) -> bool:
    c = normalize(cargo)
    return bool(re.search(r"\bdeputad[oa]s?\b|\bsenador[a]?\b", c))


def parse_party(cargo: str) -> str | None:
    """Party acronym written in the cargo string, e.g. 'Deputado (PL-RN)',
    'Deputada Federal (Bloco/PT - DF)', 'Deputado Federal pelo PSOL-RJ'."""
    if not cargo:
        return None
    upper = unicodedata.normalize("NFKD", cargo)
    upper = "".join(c for c in upper if not unicodedata.combining(c)).upper()
    for alias, canon in _PARTY_ALIASES.items():
        upper = upper.replace(alias.upper(), canon)
    upper = upper.replace("BLOCO/", " ").replace("BLOCO ", " ")
    # prefer parenthesised content, then the whole string
    candidates = re.findall(r"\(([^)]*)\)", upper) + [upper]
    for cand in candidates:
        tokens = re.split(r"[^A-Z]+", cand)
        for tok in tokens:
            if tok in PARTIES:
                return tok
    return None


_FEMININE_TITLES = r"\b(deputada|senadora|ministra|secretaria(?!\s+(de|da|do|nacional|especial|executiva)\b)|secretaria-executiva|diretora|presidenta|coordenadora|professora|pesquisadora|consultora|advogada|medica|psicologa|vereadora|procuradora|promotora|juiza|defensora|assessora|fundadora|delegada|engenheira|enfermeira|socia|empresaria|superintendenta|reitora|pro-reitora|conselheira|relatora|gerente-geral|ouvidora|prefeita|governadora|vice-presidenta|vice-presidente eleita|desembargadora|auditora|analista-chefe|chefe de gabinete|ex-ministra|ex-deputada|ex-secretaria|cientista politica|economista-chefe|gestora|educadora|nutricionista|farmaceutica|dentista|biologa|arquiteta|sociologa|antropologa|historiadora|filosofa|jornalista e escritora|escritora|ativista|lideranca|moradora|atingida|trabalhadora|agricultora|pescadora|estudante|mae|servidora|tecnica|especialista|coordenadora-geral|diretora-geral|diretora-executiva|presidente da associacao de maes|lider comunitaria|cacica|indigena)\b"
_MASCULINE_TITLES = r"\b(deputado|senador|ministro|secretario|diretor|coordenador|professor|pesquisador|consultor|advogado|medico|psicologo|vereador|procurador|promotor|juiz|defensor|assessor|fundador|delegado|engenheiro|enfermeiro|socio|empresario|reitor|pro-reitor|conselheiro|relator|ouvidor|prefeito|governador|desembargador|auditor|ex-ministro|ex-deputado|ex-secretario|gestor|educador|farmaceutico|biologo|arquiteto|sociologo|antropologo|historiador|filosofo|escritor|morador|atingido|trabalhador|agricultor|pescador|pai|servidor|tecnico|coordenador-geral|diretor-geral|diretor-executivo|general|coronel|brigadeiro|almirante|capitao|tenente|sargento|major|padre|pastor|bispo|frei|monsenhor|cacique)\b"
# words that are epicene in Portuguese and must not be used for gender
_EPICENE = {"presidente", "representante", "gerente", "chefe", "membro", "dirigente", "agente", "estudante",
            "especialista", "jornalista", "economista", "cientista", "dentista", "assistente", "docente",
            "superintendente", "vice-presidente", "integrante", "participante", "comandante", "ouvinte"}


def gender_from_title(cargo: str) -> str | None:
    """'F' / 'M' when a gender-marked job title is present; None when only
    epicene titles ('presidente', 'representante') or nothing usable.
    Note: 'secretaria' is ambiguous (the office vs. the officeholder); we
    only accept it when it is not followed by 'de/da/do/nacional/...'."""
    c = normalize(cargo)
    fem = re.search(_FEMININE_TITLES, c)
    mas = re.search(_MASCULINE_TITLES, c)
    if fem and not mas:
        return "F"
    if mas and not fem:
        return "M"
    if fem and mas:
        # earliest marker wins (the head title usually comes first)
        return "F" if fem.start() <= mas.start() else "M"
    return None


# Masculine/feminine pairs of the office words above, for the minimal-pair
# title manipulation in scripts/title_swap.py: flipping one morpheme changes
# the grammatical gender of the office while leaving the person, the words and
# the evidence untouched. Replacement forms carry their real accents; matching
# is accent-insensitive against the original string (never against normalize(),
# which strips accents and so shifts every index). Ordered longest-first so
# "vice-presidente" wins over "presidente".
_GENDER_PAIRS = [
    ("vice-presidente", "vice-presidenta"), ("pró-reitor", "pró-reitora"),
    ("coordenador-geral", "coordenadora-geral"), ("diretor-geral", "diretora-geral"),
    ("diretor-executivo", "diretora-executiva"), ("ex-ministro", "ex-ministra"),
    ("ex-deputado", "ex-deputada"), ("ex-secretário", "ex-secretária"),
    ("deputado", "deputada"), ("senador", "senadora"), ("vereador", "vereadora"),
    ("ministro", "ministra"), ("secretário", "secretária"), ("diretor", "diretora"),
    ("coordenador", "coordenadora"), ("professor", "professora"),
    ("pesquisador", "pesquisadora"), ("consultor", "consultora"),
    ("advogado", "advogada"), ("médico", "médica"), ("psicólogo", "psicóloga"),
    ("procurador", "procuradora"), ("promotor", "promotora"), ("juiz", "juíza"),
    ("defensor", "defensora"), ("assessor", "assessora"), ("fundador", "fundadora"),
    ("delegado", "delegada"), ("engenheiro", "engenheira"), ("enfermeiro", "enfermeira"),
    ("empresário", "empresária"), ("reitor", "reitora"), ("conselheiro", "conselheira"),
    ("relator", "relatora"), ("ouvidor", "ouvidora"), ("prefeito", "prefeita"),
    ("governador", "governadora"), ("desembargador", "desembargadora"),
    ("auditor", "auditora"), ("gestor", "gestora"), ("educador", "educadora"),
    ("historiador", "historiadora"), ("sociólogo", "socióloga"),
    ("antropólogo", "antropóloga"), ("arquiteto", "arquiteta"), ("biólogo", "bióloga"),
    ("escritor", "escritora"), ("morador", "moradora"), ("trabalhador", "trabalhadora"),
    ("agricultor", "agricultora"), ("pescador", "pescadora"), ("servidor", "servidora"),
    ("presidente", "presidenta"),
]
_ACCENT_CLASS = {"a": "aáàâãä", "e": "eéèêë", "i": "iíìîï", "o": "oóòôõö",
                 "u": "uúùûü", "c": "cç", "n": "nñ"}


def _accent_insensitive(word: str) -> str:
    """Regex matching `word` regardless of which accents the text actually uses."""
    parts = []
    for ch in word:
        cls = _ACCENT_CLASS.get(ch)
        parts.append(f"[{cls}]" if cls else re.escape(ch))
    return "".join(parts)


def _match_case(matched: str, replacement: str) -> str:
    """Give `replacement` the capitalisation pattern of `matched`, per word so
    that 'Vice-Presidente' -> 'Vice-Presidenta' rather than 'Vice-presidenta'."""
    if matched.isupper():
        return replacement.upper()
    split = re.compile(r"([-\s]+)")
    src_parts, dst_parts = split.split(matched), split.split(replacement)
    if len(src_parts) == len(dst_parts):
        out = []
        for src, dst in zip(src_parts, dst_parts):
            if src[:1].isupper():
                dst = dst[:1].upper() + dst[1:]
            out.append(dst)
        return "".join(out)
    if matched[:1].isupper():
        return replacement[:1].upper() + replacement[1:]
    return replacement


def flip_title_gender(cargo: str, to: str, head_only: bool = True) -> str | None:
    """Rewrite the head office word of `cargo` into grammatical gender `to`
    ('F' or 'M'), preserving capitalisation and the rest of the string.

    The word flipped is the gender-marked office word appearing earliest in the
    string (ties to the longest match), matching `gender_from_title`'s rule that
    the head title comes first: "Diretor Administrativo e Secretário da X" flips
    Diretor, not Secretário.

    Idempotent: a title whose head word is already in the requested gender comes
    back unchanged. Returns None when there is no gender-markable office word
    (an epicene head such as 'Representante'), so the caller can skip the item.

    With head_only=False every gender-marked office word is rewritten, which is
    what a minimal-pair experiment needs: "Professor e Deputado Estadual" must
    become "Professora e Deputada Estadual", not a half-flipped mixture.
    """
    if not cargo or to not in ("F", "M"):
        return None
    if not head_only:
        out, changed = cargo, False
        for masc, fem in _GENDER_PAIRS:
            src, dst = (masc, fem) if to == "F" else (fem, masc)
            pattern = rf"\b{_accent_insensitive(src)}\b"
            while True:
                m = re.search(pattern, out, re.IGNORECASE)
                if not m:
                    break
                out = out[:m.start()] + _match_case(m.group(0), dst) + out[m.end():]
                changed = True
        if changed:
            return out
        # already entirely in the target gender?
        return cargo if gender_from_title(cargo) == to else None
    best = None  # (start, -len, already_in_target, replacement_form, match)
    for masc, fem in _GENDER_PAIRS:
        src, dst = (masc, fem) if to == "F" else (fem, masc)
        for form, already in ((dst, True), (src, False)):
            m = re.search(rf"\b{_accent_insensitive(form)}\b", cargo, re.IGNORECASE)
            if m:
                cand = (m.start(), -len(m.group(0)), already, dst, m)
                if best is None or cand[:2] < best[:2]:
                    best = cand
    if best is None:
        return None
    _, _, already, dst, m = best
    if already:
        return cargo
    return cargo[:m.start()] + _match_case(m.group(0), dst) + cargo[m.end():]


_STATE_ORG_RE = re.compile(
    r"\bministr[oa]s?\b|ministerio|\bsecretari[oa] (de estado|nacional|especial|executiv[oa]|municipal|estadual|adjunt[oa])"
    r"|secretaria (nacional|especial|executiva|de estado|municipal|estadual|geral|adjunta)|casa civil|presidencia da republica"
    r"|\bgoverno\b|agencia nacional|autoridade nacional|comissao nacional|instituto nacional|fundacao nacional|departamento nacional"
    r"|empresa brasileira|companhia nacional|servico federal|agencia (brasileira|espacial)|banco (central|nacional|do brasil|do nordeste|da amazonia)"
    r"|anatel|anvisa|aneel|\bana\b|\bans\b|ibama|icmbio|funai|incra|\binss\b|\binep\b|\bfnde\b|receita federal|\bbndes\b|petrobras"
    r"|\bcaixa\b|tesouro nacional|\btcu\b|\bcgu\b|\bstf\b|\bstj\b|\btse\b|\bmpf\b|\bmpt\b|\bagu\b|\bpgr\b|\bcnj\b|\bcnmp\b|\bipea\b|\bibge\b"
    r"|\bcnpq\b|\bcapes\b|\bfinep\b|embrapa|fiocruz|\bfunasa\b|\bebserh\b|\bdnit\b|\bantt\b|\banac\b|\bantaq\b|\binmetro\b|\binpi\b"
    r"|\bserpro\b|\bdataprev\b|\bcorreios\b|\beletrobras\b|\bitaipu\b|\bconab\b|\bfunarte\b|\bancine\b|\bcade\b|\bsusep\b|\bcvm\b|\bprevic\b|\banpd\b|\bbnb\b|\bbasa\b"
    r"|conselho nacional de (justica|saude|educacao|assistencia|direitos|politica|seguranca|recursos)|forcas armadas|exercito|marinha|aeronautica"
    r"|policia (federal|civil|militar|rodoviaria)|\bprf\b|defesa civil|corpo de bombeiros|hospital universitario|\bsus\b"
    r"|prefeitura|camara municipal|assembleia legislativa|tribunal|ministerio publico|defensoria publica|advocacia-geral|procuradoria"
    r"|\bgovernador|\bprefeit[oa]\b|\bvereador|\bdesembargador|\bjuiz|\bprocurador|\bpromotor|\bdelegad[oa]\b|\bdefensor[a]? public"
    r"|\bgeneral\b|\bcoronel\b|\bbrigadeiro\b|\balmirante\b|\bcapitao\b|\btenente\b|\bsargento\b|\bmajor\b|\bmilitar\b|\bpolicial\b"
    r"|\bservidor|\bauditor|\bcontrolador|\bouvidor|\bembaixador|\bconsul\b|\bdiplomata"
)
_STATE_TITLE_RE = re.compile(r"\bsecretari[oa]\b|\bdiretor[a]?(-geral| de departamento)|\bcoordenador[a]?-geral\b|\bsuperintendente\b|\banalista\b|\bassessor")
_ACADEMIA_RE = re.compile(
    r"professor|pesquisador|universidade|\bacademic|\bdoutor|\bcientista|\bufrj\b|\busp\b|\bunb\b|unicamp|\bufmg\b"
    r"|\bufrgs\b|\bufba\b|\bufpe\b|\bufsc\b|\bufpr\b|\bufc\b|\bufpa\b|\bufam\b|\bunesp\b|\bunifesp\b|\bpuc|\bfgv\b"
    r"|\binstituto federal\b|\bifsp\b|\bcefet\b|faculdade|\breitor|pos-graduacao|\bcatedra|\bcnpq\b|\bcapes\b"
    r"|centro de pesquisa|laboratorio|\bacademia (brasileira|nacional)"
)
_PRIVATE_RE = re.compile(
    r"\bempres|\bceo\b|\bcfo\b|\bcoo\b|executiv|\bindustria|\bcomercio\b|\bfenabrave\b|\bcnc\b|\bcni\b|\bcna\b|febraban"
    r"|\bsetor (de |do |da )?(navegacao|aereo|eletrico|financeiro|bancario|farmaceutico|automotivo|sucroenergetico|de bebidas|de alimentos|de telecomunicacoes|de seguros|imobiliario|portuario|de mineracao|varejista|atacadista|de turismo|hoteleiro|de transporte|de servicos|de tecnologia|petroleiro|de energia|de saude suplementar)"
    r"|\bmercado\b|\bbanco\b|startup|\bassociacao brasileira d[aeo]s? (industri|empres|distribuidor|produtor|fabricant|operador|provedor|comerci|supermercad|atacad|revendedor|lojist|shopping|bares|hotei|agencia|companhia|concession|incorporador|construtor|planos|seguradora|bancos|instituicoes financeiras|corretor|exportador|importador|criador|cervejar|frigorifico|agroneg|mineracao|energia|energia solar|energia eolica|geracao|infraestrutura|telecomunicacoes|internet|tecnologia|startups|fintechs|ecommerce|e-commerce|maquinas|equipamentos|medicamentos|produtos|servicos|bebidas|alimentos|automoveis|veiculos|aviacao|transporte|logistica|petroleo|gas|combustiveis|celulose|papel|aco|siderurgia|quimica|textil|calcados|couro|madeira|moveis|construcao|cimento|vidro|plastico|borracha|embalagens|eletroeletronica|software|games|publicidade|midia|radiodifusao|tv|emissoras|jornais|revistas|editoras|livrarias|farmacias|drogarias|laboratorios|hospitais privados|clinicas|planos de saude|operadoras de saude|escolas particulares|ensino superior particular|mantenedoras)"
    r"|\bconfederacao nacional d[ao] (industria|comercio|agricultura|transporte|servicos|saude|instituicoes financeiras|cooperativas|municipios)"
    r"|\bfederacao (das industrias|do comercio|dos bancos|da agricultura|das empresas|nacional dos bancos|brasileira de bancos)"
    r"|\bsindicato (das empresas|da industria|das industrias|patronal|nacional das empresas|dos produtores|rural patronal)"
    r"|\bconsultoria\b|\bconsultor\b|\bsocio\b|\bsocia\b|\bltda\b|\bs\.a\.\b|\bs/a\b|\bholding\b|\bgrupo\b|\bcompanhia\b"
    r"|abrasel|abrasca|abinee|abiquim|abimaq|anfavea|abrace|abradee|abrapp|abrafarma|abras\b|abia\b|abrasf|abrint|abrapch|abeeolica|absolar|abraceel|abcon|abdib|abear|abramet|abrafrigo|abpa\b|abag\b|abiove|sindicom|sindipecas|fecomercio|fiesp|firjan|fiemg|fiergs|\bcbic\b|\bsinduscon\b|\bsecovi\b|\bfenaseg\b|\bcnseg\b|\bfebrafar\b|\binterfarma\b|\balanac\b|\bsindusfarma\b|\bfenacor\b|\babecs\b|\bzetta\b|\bcamara-e\.net\b|\babes\b|\bbrasscom\b|\bfebratel\b|\bconexis\b|\btelebrasil\b|\bsinditelebrasil\b"
    r"|123milhas|viabahia|datagro|\bambev\b|\bvale\b|\bpetrobras distribuidora\b|\bgoogle\b|\bmeta\b|\bfacebook\b|\bx corp\b|\btwitter\b|\bamazon\b|\bmicrosoft\b|\bapple\b|\buber\b|\bifood\b|\b99\b|\bnubank\b|\bitau\b|\bbradesco\b|\bsantander\b|\bbtg\b|\bxp\b|\bmagazine luiza\b|\bmercado livre\b|\bnatura\b|\bjbs\b|\bbrf\b|\bmarfrig\b|\bsuzano\b|\bklabin\b|\bgerdau\b|\bcsn\b|\busiminas\b|\bembraer\b|\bweg\b|\blatam\b|\bgol\b|\bazul\b|\bclaro\b|\bvivo\b|\btim\b|\boi\b|\btelefonica\b|\bmineradora\b|\bconcessionaria\b|\boperadora\b|\bincorporadora\b|\bconstrutora\b|\bseguradora\b|\bcervejaria\b|\bfrigorifico\b|\bfabricante\b|\bdistribuidora\b|\bfintech\b|\bplataforma\b"
)
_CIVIL_RE = re.compile(
    r"sindicat|\bassociacao|\bfederacao|\bconfederacao|\bmovimento|\bong\b|\binstituto\b|\bconselho\b|\bforum\b|\bfrente\b"
    r"|\brede\b|\bcoletivo|\bentidade|\brepresentante|\blideranca|\blider\b|indigena|quilombola|\bpastoral|\bigreja|\badvogad"
    r"|ativista|jornalista|comunicador|\bcidad|trabalhador|agricultor|pescador|\bestudante|\bmst\b|\bcut\b|\bcontag\b|\bune\b"
    r"|\bcnbb\b|\boab\b|\bcfm\b|\bcrm\b|\bsbp|sociedade brasileira|greenpeace|observatorio|\bidec\b|\bconectas\b|\bunicef\b"
    r"|\bacnur\b|\bopas\b|\boms\b|\bonu\b|\bfundacao\b|atingid|morador|vitima|familiar|paciente|\bmae\b|\bpai\b|lesionad"
    r"|artista|atleta|musico|escritor|educador|assistente social|psicolog|\bmedic|enfermeir|nutricionist|fisioterap|farmac"
    r"|dentist|veterinari|biolog|engenheir|arquitet|economist|historiador|sociolog|antropolog|filosof|teolog|\bpastor|\bpadre"
    r"|\bbispo|\brabino|\bimam|lider religios|central unica|central dos trabalhadores|forca sindical|\bugt\b|\bctb\b|\bcsb\b"
    r"|\bnova central\b|\bapib\b|\bconaq\b|\bcimi\b|\bcpt\b|\bmab\b|\bmtst\b|\bunmp\b|\bcontraf\b|\bcnte\b|\bandes\b|\bfasubra\b|\bfenaj\b"
    r"|\babi\b|\banj\b|\babraji\b|\bartigo 19\b|\bintervozes\b|\bsaferNet\b|\binstituto alana\b|\bcriola\b|\bgeledes\b|\babglt\b|\bantra\b"
    r"|\bfenapaes\b|\bapae\b|\bfebraban\b|\bcfess\b|\bcofen\b|\bcoren\b|\bcrp\b|\bcfp\b|\bcfo\b|\bcau\b|\bcrea\b|\bconfea\b|\bcorecon\b"
    r"|\bfenafar\b|\bfenafisco\b|\bfenajufe\b|\bfenasps\b|\bfenapef\b|\bcobrapol\b|\bfeneme\b|\bfenapol\b|\bfenacon\b|\bfenaban\b"
    r"|voluntari|beneficiari|usuari|consumidor|\bautist|\bpessoa com deficiencia|\bpcd\b|\bsurd|\bceg|\bcadeirante|\bsobrevivente|\bex-|\bmilitante"
)


def role_from_title(cargo: str, nome: str = "") -> str:
    """Coarse institutional-role taxonomy. Order matters: parliamentarian
    first, then state, academia, private sector, civil society/professional.
    Unclassifiable -> 'other' (must be reported, never silently dropped)."""
    c = normalize(cargo)
    if not c or c in {"desconhecido", "nao especificado", "nao informado", "-", "n/a", "na"}:
        return "other"
    if is_parliamentarian(c):
        return "parliamentarian"
    if _STATE_ORG_RE.search(c):
        return "state"
    if _ACADEMIA_RE.search(c):
        return "academia"
    if _PRIVATE_RE.search(c):
        return "private_sector"
    if _CIVIL_RE.search(c):
        return "civil_society"
    if _STATE_TITLE_RE.search(c):
        return "state"
    return "other"


_UF_RE = re.compile(r"\b(AC|AL|AP|AM|BA|CE|DF|ES|GO|MA|MT|MS|MG|PA|PB|PR|PE|PI|RJ|RN|RS|RO|RR|SC|SP|SE|TO)\b")


def parse_uf(cargo: str) -> str | None:
    """Two-letter state code when written next to a party, e.g. '(PL-RN)'."""
    if not cargo:
        return None
    for cand in re.findall(r"\(([^)]*)\)", cargo) + [cargo]:
        m = _UF_RE.search(cand.upper())
        if m and parse_party(cand) is not None:
            return m.group(1)
    return None
