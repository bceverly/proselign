<div align="center">
  <img src="website/assets/logo.svg" alt="Proselign" width="96"/>
</div>

# Proselign

[![CI](https://github.com/bceverly/proselign/actions/workflows/ci.yml/badge.svg)](https://github.com/bceverly/proselign/actions/workflows/ci.yml)
[![Security](https://github.com/bceverly/proselign/actions/workflows/security.yml/badge.svg)](https://github.com/bceverly/proselign/actions/workflows/security.yml)
[![CodeQL](https://github.com/bceverly/proselign/actions/workflows/codeql.yml/badge.svg)](https://github.com/bceverly/proselign/actions/workflows/codeql.yml)
[![Build and Release](https://github.com/bceverly/proselign/actions/workflows/build-and-release.yml/badge.svg)](https://github.com/bceverly/proselign/actions/workflows/build-and-release.yml)
[![Python Version](https://img.shields.io/badge/python-3.10--3.14-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-BSD--3--Clause-blue.svg)](LICENSE)
[![PPA](https://img.shields.io/badge/ppa-bceverly%2Fproselign-E95420.svg)](https://launchpad.net/~bceverly/+archive/ubuntu/proselign)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Linting](https://img.shields.io/badge/pylint-10.00/10-brightgreen.svg)](https://github.com/PyCQA/pylint)
[![Coverage floor](https://img.shields.io/badge/coverage%20floor-80%25-green.svg)](Makefile)
[![Security: bandit](https://img.shields.io/badge/bandit-passing-brightgreen.svg)](https://github.com/PyCQA/bandit) [![Security: semgrep](https://img.shields.io/badge/semgrep-scan-brightgreen.svg)](https://semgrep.dev/) [![Security: pip-audit](https://img.shields.io/badge/pip--audit-passing-brightgreen.svg)](https://github.com/pypa/pip-audit) [![Security: gitleaks](https://img.shields.io/badge/gitleaks-clean-brightgreen.svg)](https://github.com/gitleaks/gitleaks) [![MITRE Lucky 13](https://img.shields.io/badge/MITRE%20Lucky%2013-tested-brightgreen.svg)](tests/lucky13/conftest.py)
[![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=bceverly_proselign&metric=alert_status)](https://sonarcloud.io/summary/new_code?id=bceverly_proselign) [![Coverage](https://sonarcloud.io/api/project_badges/measure?project=bceverly_proselign&metric=coverage)](https://sonarcloud.io/summary/new_code?id=bceverly_proselign) [![Bugs](https://sonarcloud.io/api/project_badges/measure?project=bceverly_proselign&metric=bugs)](https://sonarcloud.io/summary/new_code?id=bceverly_proselign) [![Vulnerabilities](https://sonarcloud.io/api/project_badges/measure?project=bceverly_proselign&metric=vulnerabilities)](https://sonarcloud.io/summary/new_code?id=bceverly_proselign) [![Code Smells](https://sonarcloud.io/api/project_badges/measure?project=bceverly_proselign&metric=code_smells)](https://sonarcloud.io/summary/new_code?id=bceverly_proselign) [![Security Rating](https://sonarcloud.io/api/project_badges/measure?project=bceverly_proselign&metric=security_rating)](https://sonarcloud.io/summary/new_code?id=bceverly_proselign)

**Learn your writing style. Align everything to it.**

Proselign is a local-first Unix filter. It learns an individual writer's
characteristic prose style from a corpus of their own documents, then rewrites
new text in that style while preserving its facts, figures, names, citations,
quotations and structure.

```sh
proselign train --voice Bryan ~/Documents/Writing/
cat foo.txt | proselign --voice Bryan > bar.txt
```

Website and documentation: **[proselign.org](https://proselign.org)**

## Why

Documents written by several people read like it. Proselign gives them one
consistent authorial voice without changing what they say: it is a style
transfer tool with a hard rule that meaning comes first.

## Design in brief

- **Unix-first.** stdin to stdout, diagnostics to stderr, documented exit
  codes. stdout carries the transformed text and nothing else.
- **Local-first.** No telemetry and no network calls by default. Rewriting runs
  against a model *you* serve locally through any OpenAI-compatible endpoint
  (llama.cpp, vLLM, Ollama, LM Studio). `--offline` enforces loopback only.
- **A voice is data, not a model.** A small, portable, versioned JSON profile:
  measured stylometric features kept separate from interpretive style hints,
  the author's vocabulary habits, and typical passages of their own writing
  that are shown to the model as examples (kept local, never exported by
  default).
  No per-author fine-tuning is required (LoRA adapters may come later, if
  benchmarks justify them).
- **Meaning first.** Protected tokens (numbers, URLs, quotations, citations,
  inline code) are extracted before each rewrite and verified after it; failures
  are retried with feedback, and `--strict` turns any remaining failure into a
  non-zero exit instead of a silent change. Code blocks, tables and front matter
  never reach the model.
- **Your house style on top.** `~/.proselignrc` sets a default voice and
  backend and switches style rules on or off: passive voice, en and em dashes,
  exclamation marks, semicolons, first person plural ("we" instead of "I"),
  and a filter for common AI artifacts (preambles, sign-offs, stock phrasing). See
  [`examples/proselignrc`](examples/proselignrc).
- **No runtime dependencies** beyond the Python standard library.

| Command | Purpose |
|---|---|
| `proselign --voice NAME [FILE...]` | Rewrite stdin (or files) in a voice (`--voice` may come from `~/.proselignrc`) |
| `proselign --voice NAME notes.md -o new.md` | Rewrite Markdown, keeping headings, code, tables, links and front matter as they are |
| `proselign --voice NAME report.docx -o new.docx` | Rewrite a Word document's prose, keeping its formatting, styles, headers, footers, tables and images |
| `proselign train --voice NAME PATH...` | Learn a voice from `.txt`, `.md`, `.docx`, `.doc` and `.pdf` documents and `.mbox` sent-mail mailboxes (`--append` to extend) |
| `proselign voices` | List voices |
| `proselign inspect NAME` | Print a voice profile as JSON |
| `proselign eval --voice NAME` | Measure how closely rewrites reproduce the voice (compare models and settings) |
| `proselign export NAME -o FILE` / `import FILE` | Portable `.proselign` archives |
| `proselign delete NAME` | Remove a voice |

Exit status: `0` ok, `1` unexpected error, `2` usage, `3` voice, `4` fidelity
validation (`--strict`), `5` inference backend. See `man proselign`.

## Install

**Ubuntu 22.04 / 24.04 / 26.04 LTS**

```sh
sudo add-apt-repository ppa:bceverly/proselign
sudo apt update && sudo apt install proselign
```

| Architecture | Tier | Local inference |
|---|---|---|
| amd64, arm64 | Full | llama.cpp (CPU or GPU), Ollama, vLLM |
| ppc64el, s390x | CPU-only | llama.cpp or the vLLM CPU backend; no Ollama, no GPU. s390x is big-endian and needs big-endian GGUF models |

**From a release:** download the `.deb` or wheel from
[Releases](https://github.com/bceverly/proselign/releases) and verify it
against `SHA256SUMS`.

**From source:** `sudo make install` (into `/usr/local`; `PREFIX=` and
`DESTDIR=` are honored). `proselign --version` reports `W.X.Y.Z-dev` when the
installed source differs from the last release.

## Development

```sh
make install-dev     # .venv with hash-locked dev, test, security and local-LLM tooling
make llm-model       # optional: download a default GGUF model (~4.7 GB)
make llm-serve       # serve it at http://127.0.0.1:8080/v1, proselign's default endpoint
make lint            # black, isort, pylint, license headers, file length, version, shellcheck
make test            # pytest; fails below the 80% line-coverage floor
make security        # MITRE Lucky 13, Bandit, Semgrep, pip-audit, Safety, secrets scan
make sonarqube-scan  # SonarCloud with SONAR_TOKEN, otherwise a local/Docker SonarQube
make release VERSION=W.X.Y.Z # bump, lint+test, y/N prompt, commit, tag, push
```

All Python dependencies are **pinned by hash**: `requirements-*.in` lists
what we want, `make lock` resolves it into `requirements-*.txt` with a SHA-256
for every artifact, and everything installs with `pip --require-hashes`. GitHub
Actions are likewise pinned to commit SHAs. See [CONTRIBUTING.md](CONTRIBUTING.md).

### Security testing

We test explicitly for the thirteen weakness classes in Steve Christey's
[*Unforgivable Vulnerabilities*](https://cwe.mitre.org/documents/unforgivable_vulns/unforgivable.pdf)
(MITRE, Black Hat USA 2007). The checks live in `tests/lucky13/`, each test
names its number and CWE, and CI runs them as their own named step.

| # | Class (CWE) | How proselign is checked |
|---|---|---|
| 1 | Buffer overflow (120) | Multi-megabyte input, 1 MB lines and 100 KB voice names: intact output or a clean usage error |
| 2 | Cross-site scripting (79) | No HTML generation in the tool; no HTML/eval sinks or inline handlers on the website |
| 3 | Directory traversal (23) | `../`, absolute and NUL-laden voice names; traversal members in voice archives |
| 4 | Remote file inclusion (98) | No `eval`/`exec`/dynamic import; no pickle-family loaders; archives are JSON only |
| 5 | SQL injection (89) | Every `execute()` must take a literal query (gate for the upcoming SQLite store) |
| 6 | World-writable files (276) | Under `umask 000` the voice store is 0700, manifests 0600, exports 0600 |
| 7 | Direct request (425) | Proselign never binds or listens; it only connects to the endpoint you configure |
| 8 | `authenticated=1` (472) | Forged prompt delimiters in input stay data (per-request random nonces); archive manifests cannot choose their own location |
| 9 | Home-grown crypto (327) | SHA-2 via hashlib and `secrets` only; no MD5/SHA-1, `random` or XOR |
| 10 | Privilege escalation via Help (271) | No subprocess, shell, pager or browser launches anywhere in the tool |
| 11 | Symlink following (61) | Corpus walks skip links; manifest writes use `O_NOFOLLOW`/`O_EXCL`; linked voice dirs and archive links are refused |
| 12 | Hard-coded password (259) | No credential-named literals; no hard-coded `Authorization` headers |
| 13 | Integer overflow (190) | NaN/inf strength, huge or negative chunk sizes and seeds are refused at the CLI |

Run them with `make test-lucky13` (also part of `make test` and `make security`).

## Status

Milestone **M0** (repository skeleton and CLI contract) of the
[roadmap](https://proselign.org/#roadmap). The design document is in
[`docs/`](docs/).

## Responsible use

Rewriting text in another person's voice without their consent is
impersonation. Use voices you own or are authorized to use, and disclose
assisted writing where it matters.

## License

[BSD 3-Clause](LICENSE). Model weights you download are licensed separately
by their publishers.
