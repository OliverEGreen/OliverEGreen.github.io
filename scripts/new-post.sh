#!/usr/bin/env bash
# new-post — create a new draft post and open it in Typora
#
# Usage:  ./scripts/new-post.sh "My Post Title"

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 \"Post Title\""
    exit 1
fi

TITLE="$*"
DATE=$(date +%Y-%m-%d)

# Slugify: lowercase, non-alnum -> dash, trim leading/trailing dashes
SLUG=$(printf '%s' "$TITLE" \
    | tr '[:upper:]' '[:lower:]' \
    | LC_ALL=C sed -E 's/[^a-z0-9]+/-/g; s/^-+//; s/-+$//')

REPO="$(cd "$(dirname "$0")/.." && pwd)"
FILE="$REPO/_posts/${DATE}-${SLUG}.md"

if [ -e "$FILE" ]; then
    echo "Already exists: $FILE"
else
    cat > "$FILE" <<EOF
---
title: ${TITLE}
date: ${DATE}
---

EOF
    echo "Created: $FILE"
fi

if command -v open >/dev/null 2>&1; then
    open -a "Typora" "$FILE" || open "$FILE"
fi
