# Contributing to Proselign

Thanks for helping. Proselign aims to do one thing well, so the bar for new
features is "does this make the filter better at its one job?"

## License of contributions

Proselign is BSD 3-Clause licensed. By submitting a contribution you agree
that it is provided under the same license. Every Python source file starts
with:

```python
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Bryan C. Everly
```

(`make lint` checks this.)

## Development setup

```sh
make install-dev          # .venv with everything, from hash-locked requirements
make install-dev SKIP_LLM=1   # skip compiling llama.cpp if you don't need a local model
```

Python 3.10 or newer is required (3.10 is Ubuntu 22.04's interpreter, the
oldest we support).

## The rules

- `make lint` passes: black, isort, pylint at 10.00/10, license headers,
  no file over 1000 lines, version marker valid, shellcheck.
- `make test` passes, and **line coverage never drops below 80%** (the
  `--cov-fail-under` floor). New code comes with tests. The floor is only
  ever raised.
- `make security` passes: MITRE Lucky 13, Bandit, Semgrep, pip-audit,
  secrets scan.
- stdout of the filter carries transformed text and nothing else. Anything
  else goes to stderr.
- The core tool stays standard-library only.

## Dependencies are pinned by hash

Edit the `requirements-*.in` file, then:

```sh
make lock            # re-resolve requirements-*.txt with SHA-256 hashes
make lock-upgrade    # same, moving everything to the newest allowed versions
make install-dev && make security
```

Never hand-edit a `requirements-*.txt`. If pip-audit reports a vulnerability
that cannot be fixed (an upstream pin with no patched release), add a scoped,
expiring entry to `.pip-audit-ignore` with the reason; expired entries fail
the build.

## Releasing

```sh
make release                    # last part + 1 (0.1.0.0 -> 0.1.0.1)
make release VERSION=1.0.0.0    # explicit version; DRY_RUN=1 to preview
```

This bumps `src/proselign/__init__.py`, records the source digest (so
`make install` can tell release builds from `-dev` builds), runs lint and
tests, asks `Release vW.X.Y.Z? [y/N]`, then commits, tags and pushes. The tag
triggers `.github/workflows/build-and-release.yml`, which verifies, builds
the wheel and `.deb` packages, publishes a GitHub release and uploads source
packages to `ppa:bceverly/proselign`.

## Maintainer setup (one time)

### GitHub repository settings

1. **Settings → Code security**: enable *Dependency graph*, *Dependabot
   alerts*, *Dependabot security updates*, *Grouped security updates*,
   *Secret scanning* + *Push protection*, and *Private vulnerability
   reporting*. Under *Code scanning* leave the default setup **off**
   (`codeql.yml` is the advanced setup). Version updates are driven by
   `.github/dependabot.yml`.
2. **Settings → Pages**: *Source* = **GitHub Actions**; custom domain
   `proselign.org`, then *Enforce HTTPS* once DNS resolves. DNS: apex `A`
   records to 185.199.108.153, 185.199.109.153, 185.199.110.153,
   185.199.111.153 (and `AAAA` 2606:50c0:8000::153 … 8003::153), and `www`
   `CNAME` to `bceverly.github.io`.
3. **Settings → Branches**: protect `main`; require the CI, Security and
   CodeQL checks.

### Secrets (Settings → Secrets and variables → Actions)

| Secret | Used by | How to get it |
|---|---|---|
| `SONAR_TOKEN` | `sonarcloud.yml` | sonarcloud.io → import `bceverly/proselign` into organization `bceverly` (project key `bceverly_proselign`) → *My Account → Security → Generate token*. Also in the project: *Administration → Analysis Method* → turn **off** Automatic Analysis. |
| `LAUNCHPAD_GPG_KEY` | `build-and-release.yml` | `gpg --armor --export-secret-keys KEYID` (armored or base64). The key must be registered on your Launchpad account and one of its UIDs must use `bceverly@gmail.com`, the address in `debian/control` and the changelog. |
| `LAUNCHPAD_GPG_PASSPHRASE` | `build-and-release.yml` | The passphrase for that key. |
| `LAUNCHPAD_SSH_KEY` | `build-and-release.yml` | A dedicated private SSH key (`ssh-keygen -t ed25519 -f lp_upload`) whose public half is added at launchpad.net → *SSH keys*. Uploads use SFTP. |
| `SAFETY_API_KEY` | `security.yml` (optional) | platform.safetycli.com → API keys. Without it the Safety step is skipped. |

`GITHUB_TOKEN` is provided automatically. No other secrets are needed:
Gitleaks is free for personal accounts, and CodeQL, dependency review and
TruffleHog need none.

### Launchpad PPA

1. Create the PPA at launchpad.net/~bceverly → *Create a new PPA*, name
   `proselign`.
2. On the PPA page → *Change details* → *Processors*: enable **amd64**,
   **arm64**, **ppc64el** and **s390x** (only amd64 is on by default). The
   package is restricted to the architectures the local LLM tooling
   supports: amd64/arm64 fully, ppc64el/s390x as a CPU-only tier.
3. Uploads target `jammy`, `noble` and `resolute` as
   `W.X.Y.Z+ppa1~SERIES1`.
