#!/usr/bin/env bash
# publish — commit and push blog changes to olliegreen.info
#
# Usage:  ./scripts/publish.sh
#
# What it does:
#   1. Adds today's date prefix to any _posts/ filenames missing one
#      (so "my-post.md" becomes "2026-04-27-my-post.md").
#   2. Stages _posts/ and assets/.
#   3. Commits with a message based on the first changed post's title.
#   4. Pushes to origin/main. GitHub Pages rebuilds in ~30–60s.

set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

# Sanity: must be on main
BRANCH=$(git symbolic-ref --short HEAD)
if [ "$BRANCH" != "main" ]; then
    echo "Refusing to publish: not on main (currently on '$BRANCH')."
    exit 1
fi

# 1. Auto-prefix today's date on any unprefixed post filenames
TODAY=$(date +%Y-%m-%d)
shopt -s nullglob
for f in _posts/*.md _posts/*.markdown; do
    base=$(basename "$f")
    if ! [[ "$base" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}- ]]; then
        new="_posts/${TODAY}-${base}"
        echo "Adding date prefix: $base → $(basename "$new")"
        mv "$f" "$new"
    fi
done

# 2. Stage posts and any new image assets
git add _posts/ assets/ 2>/dev/null || true

if git diff --cached --quiet; then
    echo "Nothing to publish."
    exit 0
fi

# 3. Build commit message from the first staged post's title
TITLE=""
for f in $(git diff --cached --name-only -- '_posts/'); do
    if [ -f "$f" ]; then
        TITLE=$(awk -F': *' '
            /^title:/ {
                sub(/^["\x27]/, "", $2)
                sub(/["\x27]$/, "", $2)
                print $2
                exit
            }
        ' "$f")
        [ -n "$TITLE" ] && break
    fi
done

if [ -n "$TITLE" ]; then
    MSG="Publish: $TITLE"
else
    MSG="Publish update"
fi

# 4. Commit and push
git commit -m "$MSG"
git push origin main

cat <<EOF

✓ $MSG
  Live in 30–60s at https://olliegreen.info/
EOF
