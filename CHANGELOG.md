# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Versions have four
parts, `MAJOR.MINOR.PATCH.BUILD`, as in the SysManage projects.

## [Unreleased]

### Added
- M0 command-line contract: `proselign --voice NAME` filter plus `train`,
  `voices`, `inspect`, `export`, `import` and `delete`.
- Stylometric voice profiles (stdlib only) stored privately under
  `$XDG_DATA_HOME/proselign`.
- Markdown-aware segmentation; protected-token fidelity checks with a bounded
  repair loop and `--strict`.
- Backends: OpenAI-compatible local endpoint (llama.cpp, vLLM, Ollama) and a
  deterministic mock.
- MITRE "Lucky 13" security test suite.
- Ubuntu packaging for 22.04, 24.04 and 26.04 LTS (amd64, arm64) and the
  `ppa:bceverly/proselign` release pipeline.
- Project website (proselign.org).
- `train` reads Word (`.docx`, and `.doc` from Word 6 through 2003) and PDF
  documents, extracting their text in pure Python; unreadable files are
  skipped with a warning.
- `train` reads `.mbox` mailboxes of sent mail: each message is a document,
  and quoted replies, forwarded messages, signatures, links and attachments
  are left out.

- `~/.proselignrc` settings file (sample in `examples/proselignrc`): default
  voice, strength, format and backend, plus style switches for passive voice,
  en dashes, em dashes, exclamation marks, semicolons and an AI-artifact
  filter. `--config FILE`, `--no-config` and `PROSELIGN_CONFIG` select it.

### Changed
- `proselign export` writes archives owner-only (0600) instead of 0644.
- Voices reach the model as concrete writing directives (sentence length,
  contractions, person, punctuation) instead of raw metrics, which steers
  rewrites far more reliably; style breaches are fed back for another try.
- The model is told never to add greetings, sign-offs or other content.
- A number followed by a comma ("42, then") is no longer read as "42,".
