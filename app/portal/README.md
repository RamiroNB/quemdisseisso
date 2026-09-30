# Portal

Static site (HTML, CSS and JavaScript, no build step; Preact + htm in `vendor/`, Lucide icons in `icons/`).
Published at <https://ramironb.github.io/quemdisseisso.io/>.

```bash
uv run python scripts/export_demo_success.py   # from the repository root
uv run python scripts/export_portal_data.py    # writes app/portal/data/ (CPU, about 2 min)
cd app/portal && python3 serve.py 8803          # http://localhost:8803
```

`export_portal_data.py` reads the runs and blind rounds already on disk; nothing is generated or judged again. It
asserts that it reproduces the paper's fixed/broken counts and omission rates.

- Hearings: the 100 hearings of the Qwen3-4B retrieve-and-regenerate experiment (480 participants).
- Summaries: without intervention = the cross-fit baseline; with intervention = `rag:probe`. Each line carries the
  blind verdicts (Sonnet raters; Opus where it disagrees).
- Speakers: every floor-holder with more than 1,500 characters, with the labels of `lds_omission_relabel.py`.
- Speaking time is estimated from the text (140 words per minute); the page says so.
- Game (`data/game.json`): real lines replayed word by word with the readings of the probe that ran as the trigger
  (Qwen3-4B, layer 21, threshold 0.978).

Routes: `#/` (home), `#/resultados`, `#/audiencia/<id>`, `#/jogo`.

To publish: `scripts/deploy_portal.sh` copies `app/portal/` into a clone of the Pages repository next to this one and
pushes it.
