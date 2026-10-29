#!/bin/sh
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Bryan C. Everly
#
# Build Ubuntu packages for proselign.
#
#   packaging/ubuntu/build.sh binary [VERSION]          -> build/deb/proselign_VERSION_ARCH.deb
#   packaging/ubuntu/build.sh source VERSION SERIES     -> unsigned source package for the PPA
#
# The binary package is built for the host architecture (amd64 or arm64, the
# platforms the local LLM tooling supports) and installs on every Ubuntu LTS
# from 22.04 onward.  Source packages are versioned per series as
# VERSION+ppa1~SERIES1 so Launchpad builds one per release, and are signed
# separately (debsign) by the release workflow.
set -eu

MODE=${1:?usage: build.sh binary|source [VERSION] [SERIES]}
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
VERSION=${2:-$(sed -n 's/^__version__ = "\(.*\)"$/\1/p' "$ROOT/src/proselign/__init__.py")}
SERIES=${3:-}
DEBFULLNAME=${DEBFULLNAME:-Bryan C. Everly}
DEBEMAIL=${DEBEMAIL:-bryan@theeverlys.com}

case "$MODE" in
  binary) DEBVER=$VERSION; DIST=${SERIES:-unstable} ;;
  source)
    [ -n "$SERIES" ] || { echo "build.sh: source builds need a SERIES (jammy, noble, resolute)" >&2; exit 2; }
    DEBVER="${VERSION}+ppa1~${SERIES}1"; DIST=$SERIES ;;
  *) echo "build.sh: unknown mode $MODE" >&2; exit 2 ;;
esac

OUT="$ROOT/build/deb"
STAGE="$OUT/proselign-$DEBVER"
rm -rf "$STAGE"
mkdir -p "$STAGE/packaging/ubuntu"

# Stage only what the package needs: no VCS metadata, venvs or caches.
cp -r "$ROOT/src" "$ROOT/tests" "$ROOT/scripts" "$ROOT/man" "$STAGE/"
cp "$ROOT/pyproject.toml" "$ROOT/README.md" "$ROOT/LICENSE" "$STAGE/"
cp "$ROOT/packaging/ubuntu/proselign.launcher" "$STAGE/packaging/ubuntu/"
cp -r "$ROOT/packaging/ubuntu/debian" "$STAGE/debian"
find "$STAGE" \( -name '__pycache__' -o -name '*.egg-info' -o -name '.pytest_cache' \) \
  -prune -exec rm -rf {} +

cat > "$STAGE/debian/changelog" <<EOF
proselign ($DEBVER) $DIST; urgency=medium

  * New upstream release $VERSION.

 -- $DEBFULLNAME <$DEBEMAIL>  $(date -R)
EOF

cd "$STAGE"
if [ "$MODE" = binary ]; then
  dpkg-buildpackage -b -us -uc
  echo "Built: $(ls "$OUT"/proselign_"${DEBVER}"_*.deb)"
else
  dpkg-buildpackage -S -sa -d -us -uc
  echo "Built: $(ls "$OUT"/proselign_"${DEBVER}"_source.changes)"
fi
