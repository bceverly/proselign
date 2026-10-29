# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Bryan C. Everly

"""Command-line interface.

Contract: stdout carries the transformed text and nothing else; every
diagnostic goes to stderr; the exit status says what happened (see errors.py).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TextIO

from proselign import config as config_mod
from proselign import full_version
from proselign import progress as progress_mod
from proselign.errors import EXIT_ERROR, EXIT_OK, ProselignError, UsageError
from proselign.inference import BACKENDS, make_backend
from proselign.ingest.train import build_voice
from proselign.paths import write_private_file
from proselign.rewrite.docx import MAX_PACKAGE_BYTES, rewrite_docx
from proselign.rewrite.engine import Options, Rewriter
from proselign.voice.package import ARCHIVE_SUFFIX, export_voice, import_voice
from proselign.voice.registry import Registry

SUBCOMMANDS = ("rewrite", "train", "voices", "inspect", "export", "import", "delete")

EPILOG = """examples:
  proselign train --voice Bryan ~/Documents/Writing/
  cat draft.md | proselign --voice Bryan > final.md
  proselign voices
  proselign export Bryan --output Bryan.proselign

With no subcommand, proselign acts as a filter (the 'rewrite' command)."""


class _Parser(argparse.ArgumentParser):
    """argparse that raises instead of exiting, so main() owns exit codes."""

    def error(self, message: str):  # type: ignore[override]
        raise UsageError(f"{self.prog}: {message}")


def _add_rewrite_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--voice", "-v", help="voice to write in (default: 'voice' in ~/.proselignrc)"
    )
    parser.add_argument(
        "files", nargs="*", type=Path, help="input files (default: stdin; '-' also means stdin)"
    )
    parser.add_argument("--strength", type=float, help="degree of change, 0.0-1.0 (default 0.5)")
    parser.add_argument("--format", dest="text_format", choices=("markdown", "text"))
    parser.add_argument(
        "--strict",
        action="store_true",
        help="fail (exit 4) instead of keeping original text when fidelity checks fail",
    )
    parser.add_argument("--backend", choices=BACKENDS, help="inference backend")
    parser.add_argument("--endpoint", help="OpenAI-compatible base URL (default local)")
    parser.add_argument("--model", help="model name to request from the endpoint")
    parser.add_argument("--offline", action="store_true", help="refuse any non-loopback endpoint")
    parser.add_argument("--seed", type=int, help="sampling seed, for reproducible runs")
    parser.add_argument("--max-chunk-tokens", type=int, default=1500)
    parser.add_argument(
        "--json-diagnostics", action="store_true", help="emit a JSON report on stderr"
    )
    parser.add_argument(
        "--config", type=Path, help="settings file (default: $PROSELIGN_CONFIG or ~/.proselignrc)"
    )
    parser.add_argument("--no-config", action="store_true", help="ignore the settings file")
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="hide the progress line (shown by default on a terminal)",
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

    train = sub.add_parser("train", help="learn or extend a voice from documents")
    train.add_argument("--voice", "-v", required=True)
    train.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="files or directories (.txt, .md, .docx, .doc, .pdf, .mbox)",
    )
    train.add_argument("--append", action="store_true", help="add to an existing voice")

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


def _cmd_rewrite(args: argparse.Namespace, registry: Registry, io: dict) -> int:
    settings = _settings(args, io)
    name = args.voice or settings.voice
    if not name:
        raise UsageError("no voice given: use --voice NAME or set 'voice' in ~/.proselignrc")
    voice = registry.get(name)
    options = Options(
        strength=_first(args.strength, settings.strength, 0.5),
        strict=args.strict or settings.strict,
        max_chunk_tokens=args.max_chunk_tokens,
        text_format=_first(args.text_format, settings.text_format, "markdown"),
        rules=settings.rules,
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
    rewriter = Rewriter(voice, backend, options)
    if settings.progress and not args.no_progress and progress_mod.wanted(io["stderr"]):
        rewriter.progress = progress_mod.Progress(io["stderr"])
    try:
        if any(path.suffix.lower() == ".docx" for path in args.files):
            return _rewrite_docx(args, rewriter, io)
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
            for warning in report.warnings:
                print(f"proselign: warning: {warning}", file=io["stderr"])
    if args.output:
        write_private_file(args.output, result.encode("utf-8"))
    else:
        io["stdout"].write(result)
        io["stdout"].flush()
    return EXIT_OK


def _rewrite_docx(args: argparse.Namespace, rewriter: Rewriter, io: dict) -> int:
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
    result, report = rewrite_docx(data, rewriter.rewrite_units)
    rewriter.progress.close()
    write_private_file(args.output, result)
    kept = ", ".join(f"{count} {why}" for why, count in sorted(report.kept.items()))
    print(
        f"proselign: rewrote {report.rewritten} paragraph(s) -> {args.output}"
        + (f"; left as they were: {kept}" if kept else ""),
        file=io["stderr"],
    )
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

    voice = build_voice(args.voice, args.paths, base, on_skip=skipped)
    path = registry.save(voice)
    summary = voice.corpus_summary
    print(
        f"proselign: voice {voice.display_name!r} learned from {summary['documents']} "
        f"document(s), {summary['words']} words -> {path}",
        file=io["stderr"],
    )
    return EXIT_OK


def _cmd_voices(_args: argparse.Namespace, registry: Registry, io: dict) -> int:
    for voice in registry.list():
        docs = voice.corpus_summary.get("documents", 0)
        io["stdout"].write(f"{voice.voice_id}\t{voice.display_name}\t{docs} docs\n")
    return EXIT_OK


def _cmd_inspect(args: argparse.Namespace, registry: Registry, io: dict) -> int:
    io["stdout"].write(json.dumps(registry.get(args.voice).to_dict(), indent=2) + "\n")
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


def _cmd_delete(args: argparse.Namespace, registry: Registry, io: dict) -> int:
    registry.delete(args.voice)
    print(f"proselign: deleted voice {args.voice!r}", file=io["stderr"])
    return EXIT_OK


COMMANDS = {
    "rewrite": _cmd_rewrite,
    "train": _cmd_train,
    "voices": _cmd_voices,
    "inspect": _cmd_inspect,
    "export": _cmd_export,
    "import": _cmd_import,
    "delete": _cmd_delete,
}


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
