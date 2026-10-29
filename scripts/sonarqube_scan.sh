#!/bin/sh
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Bryan C. Everly
#
# SonarQube / SonarCloud analysis for `make sonarqube-scan`.
#
# Server, in order of preference:
#   1. SONAR_TOKEN set (and no SONAR_HOST_URL) -> SonarCloud (same project as CI)
#   2. SONAR_HOST_URL set                      -> that server (with SONAR_TOKEN if set)
#   3. a SonarQube already up on localhost:9000
#   4. Docker available                        -> start a throwaway Community
#      server ("sonarqube-proselign") and mint a local analysis token
#
# Scanner: sonar-scanner on PATH, else the official scanner Docker image.
# `--install-scanner` installs the CLI into ~/.local instead.
set -eu

ROOT=$(cd "$(dirname "$0")/.." && pwd)
CONTAINER=sonarqube-proselign
LOCAL_URL=http://localhost:9000
SCANNER_VERSION=7.2.0.5079

install_scanner() {
  case "$(uname -s)-$(uname -m)" in
    Linux-x86_64) platform=linux-x64 ;;
    Linux-aarch64) platform=linux-aarch64 ;;
    Darwin-*) command -v brew >/dev/null && exec brew install sonar-scanner ;;
    *) echo "Install sonar-scanner manually: https://docs.sonarsource.com/sonarqube-server/analyzing-source-code/scanners/sonarscanner/" >&2; exit 1 ;;
  esac
  dest="$HOME/.local/opt"
  mkdir -p "$dest" "$HOME/.local/bin"
  zip="/tmp/sonar-scanner-$$.zip"
  curl -fsSL -o "$zip" "https://binaries.sonarsource.com/Distribution/sonar-scanner-cli/sonar-scanner-cli-$SCANNER_VERSION-$platform.zip"
  rm -rf "$dest/sonar-scanner"
  unzip -q -o "$zip" -d "$dest"
  mv "$dest/sonar-scanner-$SCANNER_VERSION-$platform" "$dest/sonar-scanner"
  ln -sf "$dest/sonar-scanner/bin/sonar-scanner" "$HOME/.local/bin/sonar-scanner"
  rm -f "$zip"
  echo "[OK] sonar-scanner installed to $HOME/.local/bin (make sure it is on PATH)"
}

server_up() {
  curl -fs --connect-timeout 2 "$1/api/system/status" 2>/dev/null | grep -q '"status":"UP"'
}

scan() {  # $1 = host URL, $2 = token (may be empty)
  set -- "-Dsonar.host.url=$1" ${2:+"-Dsonar.token=$2"}
  if command -v sonar-scanner >/dev/null 2>&1; then
    (cd "$ROOT" && sonar-scanner "$@")
  elif command -v docker >/dev/null 2>&1; then
    echo "[INFO] sonar-scanner not installed; using the sonarsource/sonar-scanner-cli image"
    docker run --rm --network host -v "$ROOT:/usr/src" -w /usr/src \
      sonarsource/sonar-scanner-cli "$@"
  else
    echo "ERROR: no sonar-scanner and no Docker. Run: make install-sonar-scanner" >&2
    exit 1
  fi
}

if [ "${1:-}" = "--install-scanner" ]; then
  install_scanner
  exit 0
fi

[ -f "$ROOT/coverage.xml" ] || echo "[WARN] no coverage.xml; run 'make test' first for coverage figures"

if [ -n "${SONAR_TOKEN:-}" ] && [ -z "${SONAR_HOST_URL:-}" ]; then
  echo "Using SonarCloud (SONAR_TOKEN set)..."
  scan https://sonarcloud.io "$SONAR_TOKEN"
elif [ -n "${SONAR_HOST_URL:-}" ]; then
  echo "Using SonarQube at $SONAR_HOST_URL..."
  scan "$SONAR_HOST_URL" "${SONAR_TOKEN:-}"
else
  if ! server_up "$LOCAL_URL"; then
    command -v docker >/dev/null 2>&1 || {
      echo "ERROR: no SonarQube available. Options:" >&2
      echo "  - export SONAR_TOKEN=<sonarcloud token>   (scan into SonarCloud)" >&2
      echo "  - export SONAR_HOST_URL=<your server>" >&2
      echo "  - install Docker for an automatic local server" >&2
      exit 1
    }
    echo "No SonarQube on $LOCAL_URL; starting Docker container '$CONTAINER' (1-2 minutes)..."
    docker start "$CONTAINER" >/dev/null 2>&1 \
      || docker run -d --name "$CONTAINER" -p 9000:9000 sonarqube:community >/dev/null
    i=0
    until server_up "$LOCAL_URL"; do
      i=$((i + 1))
      [ "$i" -le 90 ] || { echo "ERROR: SonarQube did not start; see: docker logs $CONTAINER" >&2; exit 1; }
      sleep 2
    done
  fi
  token=${SONAR_TOKEN:-}
  if [ -z "$token" ]; then
    # A fresh server still has admin/admin; mint a short-lived analysis token.
    token=$(curl -fs -u "${SONAR_LOCAL_USER:-admin}:${SONAR_LOCAL_PASSWORD:-admin}" -X POST \
      "$LOCAL_URL/api/user_tokens/generate" -d "name=proselign-$(date +%s)" \
      | sed -n 's/.*"token":"\([^"]*\)".*/\1/p') || true
    [ -n "$token" ] || echo "[WARN] could not mint a token; set SONAR_TOKEN for your local server"
  fi
  echo "Using local SonarQube at $LOCAL_URL..."
  scan "$LOCAL_URL" "$token"
  echo "Results: $LOCAL_URL/dashboard?id=bceverly_proselign"
  echo "Container '$CONTAINER' left running; stop with: docker stop $CONTAINER"
fi
echo "[OK] SonarQube scan completed"
