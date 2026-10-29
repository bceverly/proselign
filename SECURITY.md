# Security Policy

## Reporting a vulnerability

Please report vulnerabilities privately through GitHub's
[private vulnerability reporting](https://github.com/bceverly/proselign/security/advisories/new),
not in a public issue. Include the version (`proselign --version`), your
platform, and steps to reproduce.

You can expect an acknowledgement within a week. Fixes are released as a new
version, with a GitHub security advisory crediting the reporter unless you
ask otherwise.

## Supported versions

Only the latest release receives security fixes while the project is pre-1.0.

## Scope and threat model

Proselign is a local command-line filter. It handles three kinds of untrusted
input, and treats each one as data, never as instructions:

- **Text being rewritten and corpus documents** may contain prompt-injection
  attempts. They are delimited with per-request random markers and are never
  executed or followed.
- **Voice archives** (`.proselign`) from other people are validated: one
  JSON member, size-capped, no links, nothing executable or unpickled.
- **Responses from the inference endpoint** are checked for preserved
  content before they reach stdout.

Out of scope: the security of the model server you run, and the content
of model weights.

## Assurance

Every change runs the MITRE "Lucky 13" suite (`tests/lucky13/`), Bandit,
Semgrep, CodeQL, pip-audit over hash-locked dependencies, dependency review,
and Gitleaks/TruffleHog secret scanning. Reviewed exceptions to the dependency
audit live in `.pip-audit-ignore`, each with a reason and an expiry date.
