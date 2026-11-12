# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Bryan C. Everly

"""Command-line interface.

Contract: stdout carries the transformed text and nothing else; every
diagnostic goes to stderr; the exit status says what happened (see errors.py).
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import TextIO

from proselign import config as config_mod
from proselign import full_version
from proselign import progress as progress_mod
from proselign.errors import EXIT_ERROR, EXIT_OK, ProselignError, UsageError, VoiceError
from proselign.evaluate import NEUTRAL_FILE, Plan, evaluate, load_neutral, report_text, save_neutral
from proselign.inference import BACKENDS, make_backend
from proselign.ingest.train import build_voice
from proselign.ingest.web import WEB_DIR, Fetcher, is_url, save_posts
from proselign.paths import write_private_file
from proselign.rewrite.docx import MAX_PACKAGE_BYTES, rewrite_docx
from proselign.rewrite.engine import CHUNK_TOKENS, Options, Rewriter
from proselign.tune import command as tune_command
from proselign.voice.package import ARCHIVE_SUFFIX, export_voice, import_voice
from proselign.voice.registry import Registry

SUBCOMMANDS = (
    "rewrite",
    "eval",
    "train",
    "tune",
    "voices",
    "inspect",
    "export",
    "import",
    "delete",
)

EPILOG = """examples:
  proselign train --voice Bryan ~/Documents/Writing/
  proselign train --voice Bryan --append https://example.com/
  cat draft.md | proselign --voice Bryan > final.md
  proselign voices
  proselign export Bryan --output Bryan.proselign

With no subcommand, proselign acts as a filter (the 'rewrite' command)."""


class _Parser(argparse.ArgumentParser):
    """argparse that raises instead of exiting, so main() owns exit codes."""

    def error(self, message: str):  # type: ignore[override]
        raise UsageError(f"{self.prog}: {message}")


def _add_config_args(parser: argparse.ArgumentParser) -> None:
    """``--config`` and ``--no-config``, for every command that reads settings."""
    parser.add_argument(
        "--config", type=Path, help="settings file (default: $PROSELIGN_CONFIG or ~/.proselignrc)"
    )
    parser.add_argument("--no-config", action="store_true", help="ignore the settings file")


def _add_model_args(parser: argparse.ArgumentParser) -> None:
    """Options shared by every command that calls the model."""
    parser.add_argument(
        "--voice", "-v", help="voice to write in (default: 'voice' in ~/.proselignrc)"
    )
    parser.add_argument("--strength", type=float, help="degree of change, 0.0-1.0 (default 0.5)")
    parser.add_argument("--backend", choices=BACKENDS, help="inference backend")
    parser.add_argument("--endpoint", help="OpenAI-compatible base URL (default local)")
    parser.add_argument("--model", help="model name to request from the endpoint")
    parser.add_argument("--offline", action="store_true", help="refuse any non-loopback endpoint")
    parser.add_argument("--seed", type=int, help="sampling seed, for reproducible runs")
    parser.add_argument("--max-chunk-tokens", type=int, default=CHUNK_TOKENS)
    parser.add_argument(
        "--parallel",
        type=int,
        help="chunks sent to the model at once, 1-16 (default 4); the server must "
        "allow it (Ollama: OLLAMA_NUM_PARALLEL)",
    )
    parser.add_argument(
        "--examples",
        type=int,
        help="passages of the voice's own writing shown per chunk, 0-8 (default 3)",
    )
    parser.add_argument(
        "--candidates",
        type=int,
        help="rewrites per chunk; the closest to the voice is kept, 1-5 (default 1)",
    )
    _add_config_args(parser)
    parser.add_argument(
        "--no-tuned",
        action="store_true",
        help="ignore the model fine-tuned on the voice (proselign tune), if it has one",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="hide the progress line (shown by default on a terminal)",
    )


def _add_rewrite_args(parser: argparse.ArgumentParser) -> None:
    _add_model_args(parser)
    parser.add_argument(
        "files", nargs="*", type=Path, help="input files (default: stdin; '-' also means stdin)"
    )
    parser.add_argument("--format", dest="text_format", choices=("markdown", "text"))
    parser.add_argument(
        "--strict",
        action="store_true",
        help="fail (exit 4) instead of keeping original text when fidelity checks fail",
    )
    parser.add_argument(
        "--json-diagnostics", action="store_true", help="emit a JSON report on stderr"
    )
    parser.add_argument(
        "--no-review",
        action="store_true",
        help="keep each rewritten chunk whole instead of checking it sentence by sentence",
    )
    parser.add_argument(
        "--every-sentence",
        action="store_true",
        help="rewrite every sentence, not only the ones that need work",
    )
    parser.add_argument(
        "--judge-model",
        help="model that checks each change keeps the meaning (default: the base model)",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="list every warning instead of a summary"
    )
    parser.add_argument(
        "--output", "-o", type=Path, help="write here instead of stdout (required for .docx)"
    )


def build_parser() -> argparse.ArgumentParser:
    """The full argument parser (exposed for docs and tests)."""
    parser = _Parser(
        prog="proselign",
        description="Learn a writer's prose style and align text to it, locally.",
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {full_version()}")
    sub = parser.add_subparsers(dest="command", metavar="COMMAND", parser_class=_Parser)

    _add_rewrite_args(sub.add_parser("rewrite", help="rewrite stdin in a voice (default)"))

    scorer = sub.add_parser("eval", help="measure how closely rewrites match a voice")
    _add_model_args(scorer)
    scorer.add_argument(
        "--samples", type=int, default=8, help="passages to round-trip, 1-100 (default 8)"
    )
    scorer.add_argument("--json", action="store_true", help="print the result as JSON")
    scorer.add_argument(
        "--fresh",
        action="store_true",
        help="write new neutral versions with this model instead of reusing kept ones",
    )
    scorer.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="rewrite every passage this many times (1-10) and report the average "
        "and spread, to tell a real difference from noise",
    )
    scorer.add_argument(
        "--ablate",
        action="store_true",
        help="also rewrite without each part of the voice in turn, to show which parts "
        "help and which hurt (one more rewrite of every passage per part)",
    )
    scorer.add_argument(
        "--no-control",
        action="store_true",
        help="skip the control rewrite with no voice (about a third faster)",
    )

    train = sub.add_parser("train", help="learn or extend a voice from documents")
    train.add_argument("--voice", "-v", required=True)
    train.add_argument(
        "paths",
        nargs="+",
        metavar="PATH|URL",
        help="files or directories (.txt, .md, .docx, .doc, .pdf, .mbox), or web pages: "
        "a blog's front page reads every post, a feed URL every entry in it, any other "
        "page that one article",
    )
    train.add_argument("--append", action="store_true", help="add to an existing voice")
    train.add_argument("--offline", action="store_true", help="refuse any non-loopback URL")
    _add_config_args(train)
    train.add_argument(
        "--min-words",
        type=int,
        help="leave out documents (and e-mail messages) shorter than this, to learn "
        "from considered writing rather than quick notes (default 0)",
    )

    tune = sub.add_parser(
        "tune", help="fine-tune a model on a voice (needs a CUDA GPU and the training extras)"
    )
    _add_model_args(tune)
    tune.add_argument(
        "--base",
        default=tune_command.DEFAULT_BASE,
        help="Hugging Face id (or directory) of the weights behind --model "
        f"(default {tune_command.DEFAULT_BASE})",
    )
    tune.add_argument("--name", help="name of the tuned Ollama model (default proselign-VOICE)")
    tune.add_argument("--epochs", type=int, default=2, help="passes over the pairs (default 2)")
    tune.add_argument("--rank", type=int, default=16, help="LoRA rank: 4-64 (default 16)")
    tune.add_argument(
        "--pairs-only",
        action="store_true",
        help="only write the training pairs (no GPU needed); a later run reuses them",
    )

    sub.add_parser("voices", help="list stored voices")

    inspect = sub.add_parser("inspect", help="print a voice's profile as JSON")
    inspect.add_argument("voice")

    export = sub.add_parser("export", help="write a voice to a portable archive")
    export.add_argument("voice")
    export.add_argument("--output", "-o", type=Path)
    export.add_argument(
        "--include-sources", action="store_true", help="keep corpus file paths and hashes"
    )

    imp = sub.add_parser("import", help="install a voice from an archive")
    imp.add_argument("archive", type=Path)
    imp.add_argument("--force", action="store_true", help="replace an existing voice")

    delete = sub.add_parser("delete", help="permanently remove a voice")
    delete.add_argument("voice")
    return parser


def _read_input(files: list[Path], stdin: TextIO) -> str:
    if not files:
        return stdin.read()
    parts = []
    for path in files:
        if str(path) == "-":
            parts.append(stdin.read())
            continue
        try:
            parts.append(path.read_text("utf-8"))
        except OSError as exc:
            raise UsageError(f"cannot read {path}: {exc.strerror}") from exc
    return "".join(parts)


def _settings(args: argparse.Namespace, io: dict) -> config_mod.Config:
    if args.no_config:
        return config_mod.Config()
    settings = config_mod.load(args.config)
    for warning in settings.warnings:
        print(f"proselign: warning: {warning}", file=io["stderr"])
    return settings


def _setup(args: argparse.Namespace, registry: Registry, io: dict):
    """Settings, voice, options and backend for a command that calls the model."""
    settings = _settings(args, io)
    name = args.voice or settings.voice
    if not name:
        raise UsageError("no voice given: use --voice NAME or set 'voice' in ~/.proselignrc")
    voice = registry.get(name)
    options = Options(
        strength=_first(args.strength, settings.strength, 0.5),
        strict=getattr(args, "strict", False) or settings.strict,
        max_chunk_tokens=args.max_chunk_tokens,
        text_format=_first(getattr(args, "text_format", None), settings.text_format, "markdown"),
        rules=settings.rules,
        candidates=_first(args.candidates, settings.candidates, 1),
        examples=_first(args.examples, settings.examples, 3),
        parallel=_first(args.parallel, settings.parallel, 4),
        review=settings.review and not getattr(args, "no_review", True),
    )
    options.targeted = options.review and not (
        getattr(args, "every_sentence", False) or settings.every_sentence
    )
    backend = make_backend(
        args.backend,
        args.endpoint,
        args.model,
        args.offline or settings.offline,
        args.seed,
        defaults={
            "backend": settings.backend,
            "endpoint": settings.endpoint,
            "model": settings.model,
        },
    )
    if voice.tuned.get("model") and not args.no_tuned and not args.model:
        if args.command != "tune" and hasattr(backend, "model"):
            backend.model = voice.tuned["model"]
            options.tuned = True
    return settings, voice, options, backend


def _judge(args: argparse.Namespace, settings: config_mod.Config, voice, options, backend):
    """The backend that checks each change, or None: the same server, with
    the untuned base model (it judges more strictly than the fine-tune)."""
    if not options.review or not hasattr(backend, "model"):
        return None
    judge = copy.copy(backend)
    base = voice.tuned.get("base_model") if options.tuned else None
    judge.model = args.judge_model or settings.judge_model or base or backend.model
    return judge


def _progress(args: argparse.Namespace, settings: config_mod.Config, io: dict):
    if settings.progress and not args.no_progress and progress_mod.wanted(io["stderr"]):
        return progress_mod.Progress(io["stderr"])
    return progress_mod.NoProgress()


def _cmd_eval(args: argparse.Namespace, registry: Registry, io: dict) -> int:
    settings, voice, options, backend = _setup(args, registry, io)
    progress = _progress(args, settings, io)
    kept = registry.path_for(voice.display_name) / NEUTRAL_FILE
    neutral = {} if args.fresh else load_neutral(kept)
    try:
        plan = Plan(
            args.samples,
            args.seed or 0,
            neutral,
            not args.no_control,
            args.repeats,
            args.ablate,
        )
        result = evaluate(voice, backend, options, plan, progress)
    finally:
        progress.close()
        if neutral:
            save_neutral(kept, {**load_neutral(kept), **neutral})
    if args.json:
        io["stdout"].write(json.dumps(result, indent=2) + "\n")
    else:
        io["stdout"].write(report_text(result, getattr(backend, "model", backend.name)))
    progress.finish()
    return EXIT_OK


def _cmd_rewrite(args: argparse.Namespace, registry: Registry, io: dict) -> int:
    settings, voice, options, backend = _setup(args, registry, io)
    rewriter = Rewriter(voice, backend, options, _judge(args, settings, voice, options, backend))
    rewriter.progress = _progress(args, settings, io)
    try:
        if any(path.suffix.lower() == ".docx" for path in args.files):
            return _rewrite_docx(args, rewriter, io, settings)
        if args.output and any(
            str(f) != "-" and f.resolve() == args.output.resolve() for f in args.files
        ):
            raise UsageError("--output must not be an input file; the original is kept")
        result = rewriter.rewrite(_read_input(args.files, io["stdin"]))
    finally:
        rewriter.progress.close()
        report = rewriter.report
        if args.json_diagnostics:
            print(json.dumps(report.__dict__, sort_keys=True), file=io["stderr"])
        else:
            lines = report.warnings if args.verbose else report.summary()
            for line in lines:
                print(f"proselign: warning: {line}", file=io["stderr"])
            if lines and not args.verbose:
                print("proselign: run with --verbose for details", file=io["stderr"])
    if args.output:
        write_private_file(args.output, result.encode("utf-8"))
    else:
        io["stdout"].write(result)
        io["stdout"].flush()
    rewriter.progress.finish()
    return EXIT_OK


def _rewrite_docx(
    args: argparse.Namespace, rewriter: Rewriter, io: dict, settings: config_mod.Config
) -> int:
    """Rewrite one Word document into ``--output``, keeping its formatting."""
    if len(args.files) != 1:
        raise UsageError("rewrite one .docx file at a time")
    source = args.files[0]
    if args.output is None:
        raise UsageError("a .docx input needs --output FILE.docx")
    if args.output.resolve() == source.resolve():
        raise UsageError("--output must not be the input file; the original is kept")
    try:
        if source.stat().st_size > MAX_PACKAGE_BYTES:
            raise UsageError(f"{source} is too large to rewrite")
        data = source.read_bytes()
    except OSError as exc:
        raise UsageError(f"cannot read {source}: {exc.strerror}") from exc
    result, report = rewrite_docx(data, rewriter.rewrite_units, settings.keep_title_page)
    rewriter.progress.close()
    write_private_file(args.output, result)
    kept = ", ".join(f"{count} {why}" for why, count in sorted(report.kept.items()))
    print(
        f"proselign: rewrote {report.rewritten} paragraph(s) -> {args.output}"
        + (f"; left as they were: {kept}" if kept else ""),
        file=io["stderr"],
    )
    rewriter.progress.finish()
    return EXIT_OK


def _first(*values):
    """The first value that is not ``None``."""
    return next(v for v in values if v is not None)


def _cmd_train(args: argparse.Namespace, registry: Registry, io: dict) -> int:
    base = None
    if registry.exists(args.voice):
        if not args.append:
            raise UsageError(
                f"voice {args.voice!r} already exists; use --append to add documents "
                f"or 'proselign delete {args.voice}' first"
            )
        base = registry.get(args.voice)

    def skipped(path: Path, reason: str) -> None:
        print(f"proselign: warning: skipped {path}: {reason}", file=io["stderr"])

    paths = [Path(p) for p in args.paths if not is_url(p)]
    origins = _fetch_sources(args, registry, io)
    paths.extend(origins)
    voice = build_voice(
        args.voice, paths, base, on_skip=skipped, min_words=args.min_words, origins=origins
    )
    path = registry.save(voice)
    summary = voice.corpus_summary
    print(
        f"proselign: voice {voice.display_name!r} learned from {summary['documents']} "
        f"document(s), {summary['words']} words -> {path}",
        file=io["stderr"],
    )
    return EXIT_OK


def _fetch_sources(args: argparse.Namespace, registry: Registry, io: dict) -> dict[Path, str]:
    """Fetch the URLs among ``train``'s sources into the voice's directory."""
    urls = [p for p in args.paths if is_url(p)]
    if not urls:
        return {}
    store = registry.path_for(args.voice)
    if store.is_symlink():
        raise VoiceError(f"refusing to write into {store}: it is a symbolic link")
    fetcher = Fetcher(offline=args.offline or _settings(args, io).offline)
    origins: dict[Path, str] = {}
    for url in urls:
        posts = fetcher.posts(url)
        if posts:
            origins.update(save_posts(posts, store / WEB_DIR))
            print(f"proselign: read {len(posts)} post(s) from {url}", file=io["stderr"])
        else:
            print(f"proselign: warning: no writing found at {url}", file=io["stderr"])
    for note in fetcher.notes:
        print(f"proselign: note: {note}", file=io["stderr"])
    return origins


def _cmd_voices(_args: argparse.Namespace, registry: Registry, io: dict) -> int:
    for voice in registry.list():
        docs = voice.corpus_summary.get("documents", 0)
        io["stdout"].write(f"{voice.voice_id}\t{voice.display_name}\t{docs} docs\n")
    return EXIT_OK


def _cmd_inspect(args: argparse.Namespace, registry: Registry, io: dict) -> int:
    data = registry.get(args.voice).to_dict()
    # The passages are the author's own words: show how many, not the text.
    data["examples"] = f"{len(data['examples'])} passages (stored privately)"
    io["stdout"].write(json.dumps(data, indent=2) + "\n")
    return EXIT_OK


def _cmd_export(args: argparse.Namespace, registry: Registry, io: dict) -> int:
    voice = registry.get(args.voice)
    output = args.output or Path(f"{voice.display_name}{ARCHIVE_SUFFIX}")
    export_voice(registry, args.voice, output, args.include_sources)
    note = "" if args.include_sources else " (corpus paths omitted)"
    print(f"proselign: exported {voice.voice_id!r} to {output}{note}", file=io["stderr"])
    return EXIT_OK


def _cmd_import(args: argparse.Namespace, registry: Registry, io: dict) -> int:
    voice = import_voice(registry, args.archive, args.force)
    print(f"proselign: imported voice {voice.voice_id!r}", file=io["stderr"])
    return EXIT_OK


def _cmd_tune(args: argparse.Namespace, registry: Registry, io: dict) -> int:
    settings, voice, options, backend = _setup(args, registry, io)
    progress = _progress(args, settings, io)
    tune_command.run(
        args,
        registry,
        voice,
        options,
        backend,
        lambda text: print(f"proselign: {text}", file=io["stderr"]),
        progress,
    )
    progress.finish()
    return EXIT_OK


def _cmd_delete(args: argparse.Namespace, registry: Registry, io: dict) -> int:
    registry.delete(args.voice)
    print(f"proselign: deleted voice {args.voice!r}", file=io["stderr"])
    return EXIT_OK


COMMANDS = {
    "rewrite": _cmd_rewrite,
    "eval": _cmd_eval,
    "train": _cmd_train,
    "tune": _cmd_tune,
    "voices": _cmd_voices,
    "inspect": _cmd_inspect,
    "export": _cmd_export,
    "import": _cmd_import,
    "delete": _cmd_delete,
}


def _subcommand_first(argv: list[str]) -> list[str]:
    """Move a subcommand named after leading settings options to the front.

    ``proselign --no-config train ...`` (or ``--config FILE train``) means
    ``proselign train --no-config ...``, not a rewrite of a file called train.
    """
    i = 0
    while i < len(argv):
        if argv[i] == "--no-config" or argv[i].startswith("--config="):
            i += 1
        elif argv[i] == "--config":
            i += 2
        else:
            break
    if 0 < i < len(argv) and argv[i] in SUBCOMMANDS:
        return [argv[i], *argv[:i], *argv[i + 1 :]]
    return argv


def main(
    argv: list[str] | None = None,
    stdin: TextIO | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
    registry: Registry | None = None,
) -> int:
    """Entry point; returns the process exit status."""
    argv = list(sys.argv[1:] if argv is None else argv)
    io = {
        "stdin": stdin or sys.stdin,
        "stdout": stdout or sys.stdout,
        "stderr": stderr or sys.stderr,
    }
    argv = _subcommand_first(argv)
    # Bare `proselign --voice X` is the filter: route it to `rewrite`.
    if argv and argv[0] not in SUBCOMMANDS and argv[0] not in ("-h", "--help", "--version"):
        argv.insert(0, "rewrite")

    try:
        args = build_parser().parse_args(argv)
        if args.command is None:
            build_parser().print_help(io["stderr"])
            return UsageError.exit_code
        return COMMANDS[args.command](args, registry or Registry(), io)
    except ProselignError as exc:
        print(f"proselign: error: {exc}", file=io["stderr"])
        return exc.exit_code
    except BrokenPipeError:  # pragma: no cover - e.g. `proselign ... | head`
        return EXIT_OK
    except KeyboardInterrupt:  # pragma: no cover
        return EXIT_ERROR
