"""Loader for the PublicHearingBR NLI subset (unicamp-dl/PublicHearingBR).

Portuguese public-hearing transcripts. The NLI file pairs a claim ("opiniao",
produced by the dataset authors' ChatGPT summarisation experiment) with 4 nearby
transcript chunks ("chunks_proximos") and a human label, "verificacao_manual",
for whether the chunks support it (False = supported, True = hallucination).
There are no spans: the label applies to the whole opinion.

Files are read from the pinned snapshot under $HF_HOME (default
~/.cache/huggingface), or from OCARANDU_PHBR_DIR if set. DEFAULT_LDS_PATH is the
long-document file with the full transcripts.
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path

# Snapshot of unicamp-dl/PublicHearingBR used in the paper, as laid out by huggingface_hub under $HF_HOME.
SNAPSHOT = "2f84a44bc34df483e25c987f0ff86caad0ab3433"
DATASET_DIR = Path(os.environ.get(
    "OCARANDU_PHBR_DIR",
    Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    / "hub" / "datasets--unicamp-dl--PublicHearingBR" / "snapshots" / SNAPSHOT,
))
DEFAULT_NLI_PATH = DATASET_DIR / "PublicHearingBR_NLI.jsonl"
DEFAULT_LDS_PATH = DATASET_DIR / "PublicHearingBR_LDS.jsonl"


@dataclass(frozen=True)
class Opinion:
    sample_id: int
    person_name: str
    person_role: str
    opinion: str
    context_chunks: tuple[str, ...]
    is_hallucination: bool  # verificacao_manual


def load_nli_opinions(path: Path = DEFAULT_NLI_PATH) -> list[Opinion]:
    opinions = []
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            for envolvido in row["metadados_extraidos"]["envolvidos"]:
                for op in envolvido["opinioes"]:
                    opinions.append(
                        Opinion(
                            sample_id=row["id"],
                            person_name=envolvido["nome"],
                            person_role=envolvido["cargo"],
                            opinion=op["opiniao"],
                            context_chunks=tuple(op["chunks_proximos"]),
                            is_hallucination=bool(op["verificacao_alucinacao"]["verificacao_manual"]),
                        )
                    )
    return opinions
