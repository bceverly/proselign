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

### Changed
- `proselign export` writes archives owner-only (0600) instead of 0644.
