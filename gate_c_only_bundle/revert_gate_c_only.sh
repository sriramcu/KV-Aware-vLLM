#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${1:-$(git rev-parse --show-toplevel)}"
PATCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCH="$PATCH_DIR/gate_c_only.patch"
TARGET="$REPO_ROOT/third_party/LMCache/lmcache/v1/lookup_client/lmcache_async_lookup_client.py"
EXPECTED_BEFORE="dcf274dfe4cc428dd92be6a8472706d82e45e41b7edcbe1679dc707e3fc31f2c"
EXPECTED_AFTER="5e1dcd17e2fbd18a0f25d1d6270dc802f3093ff8bcedaee3a4954267e10e3d31"

actual="$(sha256sum "$TARGET" | awk '{print $1}')"
if [[ "$actual" == "$EXPECTED_BEFORE" ]]; then
  echo "Gate C is already reverted: $TARGET"
  exit 0
fi
if [[ "$actual" != "$EXPECTED_AFTER" ]]; then
  echo "Refusing to revert: unexpected input hash for $TARGET" >&2
  echo "expected: $EXPECTED_AFTER" >&2
  echo "actual:   $actual" >&2
  exit 1
fi

cd "$REPO_ROOT"
git apply --check -R "$PATCH"
git apply -R "$PATCH"
python -m py_compile "$TARGET"

actual="$(sha256sum "$TARGET" | awk '{print $1}')"
if [[ "$actual" != "$EXPECTED_BEFORE" ]]; then
  echo "Reverted file hash mismatch" >&2
  echo "expected: $EXPECTED_BEFORE" >&2
  echo "actual:   $actual" >&2
  exit 1
fi

echo "Reverted Gate C only."
