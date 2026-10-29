# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Bryan C. Everly
# Proselign Makefile
# Development environment, linting, tests, security scanning, packaging and releases.
#
#   make install-dev                 .venv with dev tooling + local LLM tooling (hash-locked)
#   make lint                        every linter and guard (what CI runs)
#   make test                        unit + CLI tests with the 80% coverage floor
#   make security                    Lucky 13, static analysis, dependency audit, secrets scan
#   make sonarqube-scan              SonarQube analysis (SonarCloud, local server, or Docker)
#   make release [VERSION=W.X.Y.Z]   bump, check, y/N prompt, commit, tag, push

.DEFAULT_GOAL := help
.PHONY: help setup-venv install-dev install-llm llm-model llm-serve lock lock-upgrade \
	lint lint-python lint-format lint-license-headers lint-license-headers-fix lint-file-length lint-version lint-shell \
	format test test-lucky13 security security-python security-semgrep security-deps \
	security-secrets sonarqube-scan install-sonar-scanner install uninstall release deb deb-source \
	site clean

VENV := .venv
PYTHON := $(VENV)/bin/python
VENV_ACTIVATE := $(VENV)/bin/activate
PYTHON3 ?= $(shell command -v python3.14 || command -v python3.13 || command -v python3.12 \
	|| command -v python3.11 || command -v python3.10 || command -v python3)
SEMGREP_VENV := .venv-semgrep
SEMGREP := $(SEMGREP_VENV)/bin/semgrep
SAFETY_VENV := .venv-safety
SAFETY := $(SAFETY_VENV)/bin/safety
SEMGREP_CONFIGS := --config=p/default --config=p/security-audit --config=p/python \
	--config=p/trailofbits --config=p/owasp-top-ten

# Line-coverage floor.  NEVER lower it; raise it when real coverage rises.
COVERAGE_MIN := 80

PY_SOURCES := src tests scripts

# Every requirements-*.txt is a hash-locked file generated from its .in by
# `make lock`, and installed with --require-hashes: pip refuses any artifact
# whose bytes differ from the reviewed lock, and every transitive dependency
# is pinned.  Universal locks (markers resolved for 3.10+, all platforms).
LOCK := $(VENV)/bin/uv pip compile --quiet --universal --python-version 3.10 --generate-hashes
PIP_LOCKED := $(PYTHON) -m pip install --quiet --require-hashes --no-deps

# Default model for local development (llm-model / llm-serve).  A placeholder
# until the benchmark milestone picks one; override on the command line.
LLM_MODEL_REPO ?= bartowski/Qwen2.5-7B-Instruct-GGUF
LLM_MODEL_FILE ?= Qwen2.5-7B-Instruct-Q4_K_M.gguf
LLM_MODEL_DIR ?= $(HOME)/.cache/proselign/models
LLM_PORT ?= 8080

help:
	@echo "Proselign development targets"
	@echo ""
	@echo "  make install-dev        .venv with dev, test, security and local-LLM tooling"
	@echo "                          (SKIP_LLM=1 skips the llama.cpp build)"
	@echo "  make install-llm        Only the LLM tooling (llama.cpp server, hf CLI, sqlite-vec)"
	@echo "  make llm-model          Download the default GGUF model (network; ~4.7 GB)"
	@echo "  make llm-serve          Serve it on http://127.0.0.1:$(LLM_PORT)/v1 (proselign's default)"
	@echo "  make lock               Regenerate hash-locked requirements-*.txt from *.in"
	@echo "  make lock-upgrade       Same, upgrading everything to the newest allowed versions"
	@echo "  make lint               Every linter and guard (black, isort, pylint, headers, ...)"
	@echo "  make format             Reformat Python sources with black + isort"
	@echo "  make lint-license-headers-fix  Add/refresh copyright headers (2026-<this year>)"
	@echo "  make test               Test suite (fails under $(COVERAGE_MIN)% line coverage)"
	@echo "  make test-lucky13       MITRE 'Lucky 13' unforgivable-vulnerability checks"
	@echo "  make security           Lucky 13, Bandit, Semgrep, pip-audit, Safety, secrets scan"
	@echo "  make sonarqube-scan     SonarQube analysis (SONAR_TOKEN -> SonarCloud, else local/Docker)"
	@echo "  make release VERSION=W.X.Y.Z Bump version, lint+test, prompt y/N, commit, tag, push"
	@echo "                          (no VERSION = last part + 1; DRY_RUN=1, YES=1, SKIP_CHECKS=1)"
	@echo "  sudo make install       Install this tree into $(PREFIX) (version gets -dev if the"
	@echo "                          source changed since the last release); PREFIX=, DESTDIR="
	@echo "  sudo make uninstall     Remove what 'make install' installed"
	@echo "  make deb                Build the Ubuntu .deb into build/deb/"
	@echo "  make deb-source SERIES=noble  Build an unsigned PPA source package"
	@echo "  make site               Serve the website locally on http://127.0.0.1:8000"
	@echo "  make clean              Remove build artifacts and caches"

# ---------------------------------------------------------------- environment

$(VENV_ACTIVATE):
	@echo "Creating virtual environment with $(PYTHON3)..."
	@rm -rf $(VENV)
	@$(PYTHON3) -m venv $(VENV)

setup-venv: $(VENV_ACTIVATE)

install-dev: setup-venv
	@echo "Installing hash-locked development dependencies..."
	@$(PIP_LOCKED) -r requirements-dev.txt
	@$(PYTHON) -m pip install --quiet --no-deps --no-build-isolation -e .
	@if [ -z "$(SKIP_LLM)" ]; then $(MAKE) --no-print-directory install-llm; \
		else echo "[SKIP] LLM tooling (SKIP_LLM set)"; fi
	@echo "[OK] Development environment ready. Activate with: . $(VENV_ACTIVATE)"

# llama-cpp-python compiles llama.cpp from source (several minutes the first
# time).  CUDA is enabled automatically when nvcc is on PATH; pass your own
# CMAKE_ARGS for Metal, Vulkan or ROCm.  Note: pip installs a source build's
# *build* requirements (scikit-build-core, cmake) without hash checking --
# a known pip limitation; the installed packages themselves are verified.
install-llm: setup-venv
	@echo "Installing hash-locked local LLM tooling (llama.cpp server, hf CLI, sqlite-vec)..."
	@if [ -z "$${CMAKE_ARGS:-}" ] && command -v nvcc >/dev/null 2>&1; then \
		echo "[INFO] nvcc found: building llama.cpp with CUDA"; \
		export CMAKE_ARGS="-DGGML_CUDA=on"; \
	fi; \
	$(PIP_LOCKED) -r requirements-llm.txt
	@case "$$(uname -m)" in ppc64le|s390x) \
		echo "[INFO] $$(uname -m): CPU-only tier -- llama.cpp without GPU; Ollama is unavailable."; \
		[ "$$(uname -m)" = s390x ] && echo "[INFO] s390x is big-endian: use big-endian GGUF models."; ;; \
	esac; true
	@for tool in ollama llama-server vllm; do \
		command -v $$tool >/dev/null 2>&1 && echo "[INFO] also found $$tool: $$(command -v $$tool)"; \
	done; true
	@echo "[OK] LLM tooling installed. Next: make llm-model && make llm-serve"

llm-model: $(VENV_ACTIVATE)
	@echo "Downloading $(LLM_MODEL_REPO)/$(LLM_MODEL_FILE) into $(LLM_MODEL_DIR)"
	@echo "(this is the only target that downloads model weights; check the model's license)"
	@mkdir -p $(LLM_MODEL_DIR)
	@$(VENV)/bin/hf download $(LLM_MODEL_REPO) $(LLM_MODEL_FILE) --local-dir $(LLM_MODEL_DIR)

llm-serve: $(VENV_ACTIVATE)
	@test -f $(LLM_MODEL_DIR)/$(LLM_MODEL_FILE) || { echo "Run 'make llm-model' first"; exit 1; }
	@echo "OpenAI-compatible server on http://127.0.0.1:$(LLM_PORT)/v1 (Ctrl-C to stop)"
	@$(PYTHON) -m llama_cpp.server --model $(LLM_MODEL_DIR)/$(LLM_MODEL_FILE) \
		--host 127.0.0.1 --port $(LLM_PORT) --n_gpu_layers -1 --n_ctx 8192

$(VENV)/bin/uv: $(VENV_ACTIVATE)
	@$(PIP_LOCKED) -r requirements-dev.txt

lock: $(VENV)/bin/uv
	@$(LOCK) requirements-dev.in -o requirements-dev.txt
	@$(LOCK) requirements-llm.in -c requirements-dev.txt -o requirements-llm.txt
	@$(LOCK) requirements-semgrep.in -o requirements-semgrep.txt
	@$(LOCK) requirements-safety.in -o requirements-safety.txt
	@echo "[OK] Locks regenerated; review the diff, then make install-dev"

lock-upgrade: $(VENV)/bin/uv
	@$(LOCK) --upgrade requirements-dev.in -o requirements-dev.txt
	@$(LOCK) --upgrade requirements-llm.in -c requirements-dev.txt -o requirements-llm.txt
	@$(LOCK) --upgrade requirements-semgrep.in -o requirements-semgrep.txt
	@$(LOCK) --upgrade requirements-safety.in -o requirements-safety.txt
	@echo "[OK] Locks upgraded; review the diff, then make install-dev && make security"

# ---------------------------------------------------------------------- lint

lint: lint-format lint-python lint-license-headers lint-file-length lint-version lint-shell
	@echo "[OK] All linting completed successfully!"

lint-format: $(VENV_ACTIVATE)
	@echo "=== black / isort (check only; run 'make format' to fix) ==="
	@$(PYTHON) -m black --check --diff --quiet $(PY_SOURCES)
	@$(PYTHON) -m isort --check-only --diff --quiet $(PY_SOURCES)

lint-python: $(VENV_ACTIVATE)
	@echo "=== pylint ==="
	@$(PYTHON) -m pylint --rcfile=.pylintrc src/proselign scripts
	@$(PYTHON) -m pylint --rcfile=tests/.pylintrc tests

# Every source file asserts BSD-3-Clause and "Copyright (c) 2026[-YYYY]
# Bryan C. Everly" through the current year; the fix target adds missing
# headers and bumps the end year (run it after New Year).
lint-license-headers: $(VENV_ACTIVATE)
	@$(PYTHON) scripts/check_license_headers.py

lint-license-headers-fix: $(VENV_ACTIVATE)
	@$(PYTHON) scripts/check_license_headers.py --fix

# No source file over 1000 lines: small modules are part of "do one thing well".
lint-file-length:
	@bad=$$(find src tests scripts -name '*.py' -exec wc -l {} + \
		| awk '$$2 != "total" && $$1 > 1000 { print "  " $$1 "  " $$2 }'); \
	if [ -n "$$bad" ]; then echo "ERROR: files over 1000 lines:"; echo "$$bad"; exit 1; fi; \
	echo "[OK] all source files within 1000 lines"

lint-version: $(VENV_ACTIVATE)
	@$(PYTHON) scripts/check_version.py

# shellcheck is optional locally (CI always has it).
lint-shell:
	@if command -v shellcheck >/dev/null 2>&1; then \
		shellcheck packaging/ubuntu/build.sh scripts/sonarqube_scan.sh && echo "[OK] shellcheck"; \
	else \
		echo "[SKIP] shellcheck not installed"; \
	fi

format: $(VENV_ACTIVATE)
	@$(PYTHON) -m black $(PY_SOURCES)
	@$(PYTHON) -m isort $(PY_SOURCES)

# ---------------------------------------------------------------------- test

test: $(VENV_ACTIVATE)
	@echo "=== Running tests (coverage floor $(COVERAGE_MIN)%) ==="
	@$(PYTHON) -m pytest --cov=proselign --cov-report=term-missing --cov-report=xml \
		--cov-report=html --cov-fail-under=$(COVERAGE_MIN)
	@echo "[OK] Tests passed"

# MITRE "Lucky 13" (Christey 2007).  Also part of `make test` and
# `make security`; its own target so a regression is named on its own.
test-lucky13: $(VENV_ACTIVATE)
	@echo "=== MITRE 'Lucky 13' unforgivable vulnerabilities ==="
	@$(PYTHON) -m pytest tests/lucky13 -m lucky13 -q --tb=short -p no:cacheprovider

# ------------------------------------------------------------------ security

security: test-lucky13 security-python security-semgrep security-deps security-secrets
	@echo "[OK] All security checks completed"

security-python: $(VENV_ACTIVATE)
	@echo "=== Bandit static security analysis ==="
	@$(PYTHON) -m bandit -c pyproject.toml -r src scripts --severity-level low -q

# Semgrep gets its OWN venv (requirements-semgrep.txt) so its pinned
# dependencies never hold back packages in the development venv.
$(SEMGREP): requirements-semgrep.txt
	@echo "Creating the semgrep tool venv ($(SEMGREP_VENV))..."
	@$(PYTHON3) -m venv $(SEMGREP_VENV)
	@$(SEMGREP_VENV)/bin/pip install --quiet --require-hashes --no-deps -r requirements-semgrep.txt
	@touch $@

$(SAFETY): requirements-safety.txt
	@echo "Creating the safety tool venv ($(SAFETY_VENV))..."
	@$(PYTHON3) -m venv $(SAFETY_VENV)
	@$(SAFETY_VENV)/bin/pip install --quiet --require-hashes --no-deps -r requirements-safety.txt
	@touch $@

security-semgrep: $(SEMGREP)
	@echo "=== Semgrep static analysis (gating) ==="
	@$(SEMGREP) scan --error --quiet --metrics=off $(SEMGREP_CONFIGS) src scripts packaging

# pip-audit gates on known vulnerabilities in every lock file (the runtime
# has no third-party dependencies at all); reviewed, expiring exceptions
# live in .pip-audit-ignore.  Safety needs an account
# (SAFETY_API_KEY or `safety auth login`), so it is advisory when absent.
security-deps: $(VENV_ACTIVATE) $(SAFETY)
	@echo "=== pip-audit (dependency vulnerabilities, all lock files) ==="
	@$(PYTHON) scripts/audit_deps.py
	@echo "=== Safety ==="
	@if [ -n "$${SAFETY_API_KEY:-}" ]; then \
		$(SAFETY) scan --output screen < /dev/null; \
	else \
		$(SAFETY) scan --output screen < /dev/null 2>/dev/null \
			|| echo "[SKIP] Safety not authenticated; set SAFETY_API_KEY or run 'safety auth login'"; \
	fi

# gitleaks (if installed) scans the working tree and gates; the pattern scan
# is a dependency-free fallback that always runs.
security-secrets:
	@echo "=== Secrets detection ==="
	@if command -v gitleaks >/dev/null 2>&1; then \
		gitleaks dir . --no-banner --redact; \
	else \
		echo "[INFO] gitleaks not installed; using pattern scan only"; \
	fi
	@if grep -rInE --exclude-dir=.venv --exclude-dir=.venv-semgrep --exclude-dir=build \
		--exclude-dir=htmlcov --exclude-dir=.git --exclude-dir=__pycache__ --exclude-dir=.venv-safety \
		--exclude-dir=.scannerwork --exclude='*.docx' \
		"(AKIA[0-9A-Z]{16}|-----BEGIN [A-Z ]*PRIVATE KEY-----|(api_?key|secret|token|password)[[:space:]]*[:=][[:space:]]*['\"][A-Za-z0-9+/=_-]{16,}['\"])" \
		. ; then \
		echo "ERROR: possible secrets found above"; exit 1; \
	fi
	@echo "[OK] No secrets detected"

# ----------------------------------------------------------------- sonarqube

# Coverage first, so the scan reports real numbers.  See the script for the
# server resolution order (SonarCloud, SONAR_HOST_URL, localhost, Docker).
sonarqube-scan: test
	@scripts/sonarqube_scan.sh

install-sonar-scanner:
	@scripts/sonarqube_scan.sh --install-scanner

# ------------------------------------------------------------------- install

# A self-contained install for development builds: the package goes in its
# own directory (never into the system's dist-packages, where the .deb puts
# it), with a launcher that points the system python3 at it.  The installed
# version reads W.X.Y.Z-dev when the source differs from the last `make
# release` (see scripts/source_digest.py).  Runtime needs only python3.
PREFIX ?= /usr/local
INSTALL_LIB := $(DESTDIR)$(PREFIX)/lib/proselign
INSTALL_BIN := $(DESTDIR)$(PREFIX)/bin
INSTALL_MAN := $(DESTDIR)$(PREFIX)/share/man/man1

install:
	@suffix=$$($(PYTHON3) scripts/source_digest.py --suffix); \
	version=$$(sed -n 's/^__version__ = "\(.*\)"$$/\1/p' src/proselign/__init__.py)$$suffix; \
	echo "Installing proselign $$version into $(DESTDIR)$(PREFIX)"; \
	rm -rf $(INSTALL_LIB); \
	install -d -m 0755 $(INSTALL_LIB) $(INSTALL_BIN) $(INSTALL_MAN); \
	cp -R src/proselign $(INSTALL_LIB)/; \
	find $(INSTALL_LIB) -name __pycache__ -prune -exec rm -rf {} +; \
	rm -f $(INSTALL_LIB)/proselign/RELEASE_DIGEST; \
	printf '# Generated by make install\nBUILD_SUFFIX = "%s"\n' "$$suffix" \
		> $(INSTALL_LIB)/proselign/_build.py; \
	chmod -R u=rwX,go=rX $(INSTALL_LIB); \
	printf '#!%s\nimport sys\nsys.path.insert(0, "%s")\nfrom proselign.cli import main\nsys.exit(main())\n' \
		"$$(command -v python3)" "$(PREFIX)/lib/proselign" > $(INSTALL_BIN)/proselign; \
	chmod 0755 $(INSTALL_BIN)/proselign; \
	gzip -9n < man/proselign.1 > $(INSTALL_MAN)/proselign.1.gz; \
	chmod 0644 $(INSTALL_MAN)/proselign.1.gz; \
	echo "[OK] $(PREFIX)/bin/proselign -> $$version"

uninstall:
	rm -rf $(INSTALL_LIB)
	rm -f $(INSTALL_BIN)/proselign $(INSTALL_MAN)/proselign.1.gz
	@echo "[OK] proselign removed from $(DESTDIR)$(PREFIX)"

# ------------------------------------------------------------------- release

release:
	@$(PYTHON3) scripts/release.py $(if $(VERSION),--version "$(VERSION)") \
		$(if $(MSG),--message "$(MSG)") $(if $(DRY_RUN),--dry-run) $(if $(YES),--yes) \
		$(if $(SKIP_CHECKS),--skip-checks)

# ----------------------------------------------------------------- packaging

deb:
	@packaging/ubuntu/build.sh binary $(VERSION)

deb-source:
	@test -n "$(SERIES)" || { echo "usage: make deb-source SERIES=jammy|noble|resolute"; exit 2; }
	@packaging/ubuntu/build.sh source "$${VERSION:-$$(sed -n 's/^__version__ = "\(.*\)"$$/\1/p' src/proselign/__init__.py)}" $(SERIES)

site:
	@echo "Serving website/ on http://127.0.0.1:8000 (Ctrl-C to stop)"
	@cd website && $(PYTHON3) -m http.server 8000 --bind 127.0.0.1

clean:
	rm -rf build dist htmlcov coverage.xml .coverage .pytest_cache .scannerwork src/*.egg-info
	find . -name __pycache__ -not -path './.venv*' -prune -exec rm -rf {} +
