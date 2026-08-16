#!/bin/bash
# Cuts a new release on the AICryptoTrading-releases distribution repo.
# Usage: scripts/cut_release.sh <version-tag> <critical|routine> <notes-file>
# Example: scripts/cut_release.sh v1.2.0 routine /tmp/release-notes.txt
#
# The notes file's content becomes the release body, with a machine-readable
# "severity: <level>" line prepended -- src/update/release_client.py's
# parse_release_notes() reads this line to set the severity shown on any
# install's dashboard.
#
# Always pushes a fresh mirror of the allow-listed paths before tagging so a
# cut release can never silently point at stale code (same lesson AITrading
# learned 2026-08-12 -- see that repo's cut_release.sh comment).

set -euo pipefail

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <version-tag> <critical|routine> <notes-file>"
    exit 1
fi

VERSION_TAG="$1"
SEVERITY="$2"
NOTES_FILE="$3"

if [ "$SEVERITY" != "critical" ] && [ "$SEVERITY" != "routine" ]; then
    echo "Severity must be 'critical' or 'routine', got: $SEVERITY"
    exit 1
fi

if [ ! -f "$NOTES_FILE" ]; then
    echo "Notes file not found: $NOTES_FILE"
    exit 1
fi

RELEASES_REPO=$(grep -A2 "^update:" config/settings.yaml | grep "releases_repo:" | sed 's/.*releases_repo:\s*"\?\([^"]*\)"\?/\1/')

if [ -z "$RELEASES_REPO" ]; then
    echo "Could not read update.releases_repo from config/settings.yaml"
    exit 1
fi

echo "Syncing a fresh code snapshot to $RELEASES_REPO before tagging..."
CLONE_DIR=$(mktemp -d)
trap 'rm -rf "$CLONE_DIR"' EXIT

git clone --quiet "https://github.com/${RELEASES_REPO}.git" "$CLONE_DIR"
find "$CLONE_DIR" -mindepth 1 -maxdepth 1 -not -name '.git' -exec rm -rf {} +

FILELIST=$(mktemp)
git ls-files -- src/ web/ scripts/ requirements.txt start.py > "$FILELIST"
while IFS= read -r f; do
    mkdir -p "$CLONE_DIR/$(dirname "$f")"
    cp "$f" "$CLONE_DIR/$f"
done < "$FILELIST"
rm -f "$FILELIST"

pushd "$CLONE_DIR" > /dev/null
git add -A
if git diff --cached --quiet; then
    echo "No code changes since the last release sync -- releases repo already current."
else
    git -c user.email="flyguy91355@gmail.com" -c user.name="flyguy91355" \
        commit --quiet -m "Sync $(date -u +%Y-%m-%d) for $VERSION_TAG"
    git push --quiet origin main
    echo "Pushed fresh code snapshot."
fi
popd > /dev/null

BODY_FILE=$(mktemp)
echo "severity: $SEVERITY" > "$BODY_FILE"
echo "" >> "$BODY_FILE"
cat "$NOTES_FILE" >> "$BODY_FILE"

gh release create "$VERSION_TAG" \
    --repo "$RELEASES_REPO" \
    --title "$VERSION_TAG" \
    --notes-file "$BODY_FILE"

rm -f "$BODY_FILE"

echo "Released $VERSION_TAG ($SEVERITY) to $RELEASES_REPO"
