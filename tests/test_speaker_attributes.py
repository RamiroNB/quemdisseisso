from ocarandu.data.speaker_attributes import (
    gender_from_title,
    is_parliamentarian,
    parse_party,
    parse_uf,
    role_from_title,
)


def test_party_parsing_variants():
    assert parse_party("Deputado (PL-RN)") == "PL"
    assert parse_party("Deputada Federal (Bloco/PT - DF)") == "PT"
    assert parse_party("Deputado Federal pelo PSOL-RJ") == "PSOL"
    assert parse_party("Deputado, Bloco/SOLIDARIEDADE - MG") == "SOLIDARIEDADE"
    assert parse_party("Deputada (União Brasil-BA)") == "UNIAO"
    assert parse_party("Deputado Federal") is None
    assert parse_party("Ministra da Saúde") is None


def test_uf_only_when_party_present():
    assert parse_uf("Deputado (PL-RN)") == "RN"
    assert parse_uf("Deputada Federal (PT-DF)") == "DF"
    assert parse_uf("Diretor de Políticas de SP") is None


def test_parliamentarian_detection():
    assert is_parliamentarian("Deputado Federal")
    assert is_parliamentarian("Deputada, Presidente da Comissão")
    assert is_parliamentarian("Senadora")
    assert not is_parliamentarian("Presidente da ANATEL")


def test_gender_uses_marked_titles_only():
    assert gender_from_title("Deputada Federal") == "F"
    assert gender_from_title("Deputado") == "M"
    assert gender_from_title("Ministra da Saúde") == "F"
    assert gender_from_title("Presidente da ANATEL") is None  # epicene
    assert gender_from_title("Representante da Conectas Direitos Humanos") is None
    assert gender_from_title("Secretaria Nacional de Políticas para Mulheres") is None  # the office, not a person
    assert gender_from_title("Coordenadora-Geral de Avaliação de Tecnologias em Saúde") == "F"
    assert gender_from_title("Diretor de Políticas de Educação Especial") == "M"


def test_role_taxonomy():
    assert role_from_title("Deputado (PL-RN)") == "parliamentarian"
    assert role_from_title("Ministra da Saúde") == "state"
    assert role_from_title("Presidente da ANATEL") == "state"
    assert role_from_title("Professora da UNB") == "academia"
    assert role_from_title("Sócio e administrador da 123Milhas") == "private_sector"
    assert role_from_title("Representante do setor de navegação") == "private_sector"
    assert role_from_title("Representante da Conectas Direitos Humanos") == "civil_society"
    assert role_from_title("Coordenador-Geral da Federação Única dos Petroleiros (FUP)") == "civil_society"
    assert role_from_title("Metalúrgica lesionada") == "civil_society"
    assert role_from_title("Desconhecido") == "other"
    assert role_from_title("Neurologista") == "other"  # unaffiliated professional: no rule matches
    assert role_from_title("Representante da Autoridade Nacional de Proteção de Dados (ANPD)") == "state"
    assert role_from_title("Presidente da Coordenação de Aperfeiçoamento de Pessoal de Nível Superior (CAPES)") == "state"
    assert role_from_title("Coordenadora-Geral de Avaliação de Tecnologias em Saúde do Ministério da Saúde") == "state"
    assert role_from_title("Presidente da Sociedade Brasileira de Retina e Vítreo") == "civil_society"


def test_flip_title_gender_basic_pairs():
    from ocarandu.data.speaker_attributes import flip_title_gender

    assert flip_title_gender("Deputado (PT - SP)", "F") == "Deputada (PT - SP)"
    assert flip_title_gender("Deputada (PT - SP)", "M") == "Deputado (PT - SP)"
    assert flip_title_gender("Diretor da Associação X", "F") == "Diretora da Associação X"
    assert flip_title_gender("professora da UFRJ", "M") == "professor da UFRJ"


def test_flip_title_gender_preserves_accents_and_case():
    from ocarandu.data.speaker_attributes import flip_title_gender

    # the pair table is unaccented but the match must land on the accented original
    assert flip_title_gender("Secretário Nacional de Saúde", "F") == "Secretária Nacional de Saúde"
    assert flip_title_gender("DEPUTADO FEDERAL", "F") == "DEPUTADA FEDERAL"


def test_flip_title_gender_longest_match_first():
    from ocarandu.data.speaker_attributes import flip_title_gender

    assert flip_title_gender("Vice-Presidente do Conselho", "F") == "Vice-Presidenta do Conselho"
    assert flip_title_gender("Diretor-Geral da Agência", "F") == "Diretora-Geral da Agência"


def test_flip_title_gender_returns_none_when_unmarkable():
    from ocarandu.data.speaker_attributes import flip_title_gender

    assert flip_title_gender("Representante da Sociedade Civil", "F") is None
    assert flip_title_gender("", "F") is None
    assert flip_title_gender("Deputado (PT - SP)", "X") is None


def test_flip_title_gender_is_idempotent():
    from ocarandu.data.speaker_attributes import flip_title_gender

    assert flip_title_gender("Deputada (PT - SP)", "F") == "Deputada (PT - SP)"
    assert flip_title_gender("Deputado (PT - SP)", "M") == "Deputado (PT - SP)"


def test_flip_title_gender_ignores_extra_whitespace_and_long_titles():
    from ocarandu.data.speaker_attributes import flip_title_gender

    cargo = "Diretor  Administrativo e Secretário da Associação — ATAM/DF"
    assert flip_title_gender(cargo, "F") == "Diretora  Administrativo e Secretário da Associação — ATAM/DF"


def test_flip_title_gender_all_words():
    from ocarandu.data.speaker_attributes import flip_title_gender

    cargo = "Professor e Deputado Estadual pelo PSOL"
    assert flip_title_gender(cargo, "F", head_only=False) == "Professora e Deputada Estadual pelo PSOL"
    assert flip_title_gender(cargo, "M", head_only=False) == cargo
    # head_only keeps the old behaviour: only the earliest word flips
    assert flip_title_gender(cargo, "F") == "Professora e Deputado Estadual pelo PSOL"


def test_flip_title_gender_all_words_is_idempotent_and_skips_epicene():
    from ocarandu.data.speaker_attributes import flip_title_gender

    assert flip_title_gender("Diretora-Geral e Conselheira", "F", head_only=False) == "Diretora-Geral e Conselheira"
    assert flip_title_gender("Representante da Sociedade Civil", "F", head_only=False) is None
