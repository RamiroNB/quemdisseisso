#!/usr/bin/env bash
# Publish app/portal to the GitHub Pages repo RamiroNB/quemdisseisso.io (https://ramironb.github.io/quemdisseisso.io/).
#
# Usage: scripts/deploy_portal.sh ["commit message"]      DRY=1 scripts/deploy_portal.sh  (copy and stamp, no commit)
#
# The Pages repo is a clone next to this one (../quemdisseisso.io) and holds only the built site: this script mirrors
# app/portal into it (its own README.md and .nojekyll are kept), stamps a version on every CSS/JS/data URL so a
# visitor never mixes files of two deploys (Pages caches for 10 min), commits and pushes.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$ROOT/app/portal"
DST="${PORTAL_REPO:-$ROOT/../quemdisseisso.io}"
MSG="${1:-Update the portal}"
[ -d "$DST/.git" ] || { echo "Pages repo not found at $DST (git clone git@github.com:RamiroNB/quemdisseisso.io.git)"; exit 1; }

rsync -a --delete --exclude .git --exclude README.md --exclude .nojekyll --exclude __pycache__ "$SRC/" "$DST/"
touch "$DST/.nojekyll"

V="$(date +%Y%m%d%H%M%S)"
python3 - "$DST" "$V" <<'EOF'
import pathlib, re, sys
dst, v = pathlib.Path(sys.argv[1]), sys.argv[2]
html = dst / "index.html"
s = html.read_text()
s = re.sub(r'((?:href|src)="(?:css|js)/[^"?]+)"', rf'\1?v={v}"', s)
html.write_text(s)
for js in (dst / "js").glob("*.js"):
    s = js.read_text()
    s = re.sub(r"(from\s+')(\.{1,2}/[^'?]+\.js)(')", rf"\1\2?v={v}\3", s)                  # static module imports
    s = re.sub(r"getJSON\((['`])(data/[^'`?]+)\1\)", rf"getJSON(\1\2?v={v}\1)", s)       # data files
    js.write_text(s)
print(f"stamped version {v}")
EOF

cd "$DST"
git add -A
if [ "${DRY:-0}" = "1" ]; then git status --short | head -20; echo "(DRY=1: not committed)"; exit 0; fi
if git diff --cached --quiet; then echo "nothing changed"; exit 0; fi
git commit -q -m "$MSG"
git push -q -u origin main
echo "pushed: https://ramironb.github.io/quemdisseisso.io/"
