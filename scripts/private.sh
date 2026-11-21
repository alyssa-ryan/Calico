#!/bin/sh
# The private material that shares this folder: scripts/private.sh status | save
set -eu
cd "$(dirname "$0")/.."
if [ ! -d .private.git ] || [ ! -f .private/paths.txt ]; then
  echo "No private material in this checkout."
  exit 0
fi
pg() { git --git-dir=.private.git --work-tree=. "$@"; }
set -f
specs=$(grep -v -e '^#' -e '^[[:space:]]*$' .private/paths.txt)
skip=':(exclude,glob)**/__pycache__/** :(exclude,glob)**/.DS_Store'
case "${1:-status}" in
  status)
    # shellcheck disable=SC2086
    pg status --short --untracked-files=all --ignored=traditional -- $specs $skip | sed 's/^!! /?? /'
    ;;
  save)
    # shellcheck disable=SC2086
    pg add --all --force -- $specs $skip
    if pg diff --cached --quiet; then
      echo "Nothing new in the private material."
    else
      pg commit -q -m "Private material, $(date +%Y-%m-%d)"
      echo "Committed: $(pg log -1 --format='%h %s')"
    fi
    pg push -q calico-private main
    echo "Pushed to calico-private."
    ;;
  *)
    echo "usage: scripts/private.sh status|save" >&2
    exit 2
    ;;
esac
