"""Command line interface: ``bgremover remove|download|models|serve``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from .models import DEFAULT_MODEL, MODELS, get_spec

_EPILOG = """\
examples:
  bgremover remove photo.jpg
  bgremover remove photo.jpg -o out.png --refine --trim
  bgremover remove *.jpg -o cutouts/ -m isnet-general-use
  cat photo.jpg | bgremover remove - > out.png
  bgremover remove photo.jpg --bg white -o flat.jpg
  bgremover download u2net isnet-general-use
  bgremover serve --port 3001
"""


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bgremover",
        description="Remove image backgrounds with ONNX segmentation models.",
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"bgremover {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # remove ----------------------------------------------------------------
    remove_parser = subparsers.add_parser(
        "remove",
        help="remove the background from one or more images",
        description="Remove the background from one or more images.",
    )
    remove_parser.add_argument("inputs", nargs="+", help="image files, or - for stdin")
    remove_parser.add_argument(
        "-o",
        "--output",
        help="output file, output directory, or - for stdout "
        "(default: <name>_nobg.png next to each input)",
    )
    remove_parser.add_argument(
        "-m", "--model", default=DEFAULT_MODEL, help=f"model to use (default: {DEFAULT_MODEL})"
    )
    remove_parser.add_argument(
        "--only-mask", action="store_true", help="write the grayscale mask instead of the cutout"
    )
    remove_parser.add_argument(
        "--refine", action="store_true", help="harden and feather the mask edges"
    )
    remove_parser.add_argument("--foreground-threshold", type=int, default=240, metavar="N")
    remove_parser.add_argument("--background-threshold", type=int, default=15, metavar="N")
    remove_parser.add_argument(
        "--erode", type=int, default=0, metavar="N", help="shrink the mask edge by N pixels"
    )
    remove_parser.add_argument(
        "--blur", type=float, default=1.0, metavar="R", help="feather radius used with --refine"
    )
    remove_parser.add_argument(
        "--bg",
        "--background",
        dest="background",
        metavar="COLOR",
        help="composite onto a colour (name, #rrggbb) instead of transparency",
    )
    remove_parser.add_argument(
        "--trim", action="store_true", help="crop fully transparent borders away"
    )
    remove_parser.add_argument(
        "-f", "--format", dest="image_format", help="output format (default: from the suffix, PNG)"
    )
    remove_parser.add_argument("-q", "--quality", type=int, help="JPEG/WEBP quality")
    remove_parser.set_defaults(func=_cmd_remove)

    # download --------------------------------------------------------------
    download_parser = subparsers.add_parser(
        "download", help="pre-fetch model weights into the cache"
    )
    download_parser.add_argument("models", nargs="*", help="model names (default: the default one)")
    download_parser.add_argument("--all", action="store_true", help="download every known model")
    download_parser.add_argument(
        "--check", action="store_true", help="re-hash cached files and report mismatches"
    )
    download_parser.set_defaults(func=_cmd_download)

    # models ----------------------------------------------------------------
    models_parser = subparsers.add_parser("models", help="list the available models")
    models_parser.add_argument("--json", action="store_true", help="machine readable output")
    models_parser.set_defaults(func=_cmd_models)

    # serve -----------------------------------------------------------------
    serve_parser = subparsers.add_parser("serve", help="run the HTTP API")
    serve_parser.add_argument("--host", default="0.0.0.0", help="bind address (default: 0.0.0.0)")
    serve_parser.add_argument("--port", type=int, default=3001, help="port (default: 3001)")
    serve_parser.add_argument("--reload", action="store_true", help="reload on code changes")
    serve_parser.add_argument("--workers", type=int, default=1, help="worker processes")
    serve_parser.set_defaults(func=_cmd_serve)

    return parser


def _output_path(source: Path, output: str | None, suffix: str, many: bool) -> Path:
    if output is None:
        return source.with_name(f"{source.stem}_nobg{suffix}")
    target = Path(output)
    if many or target.is_dir() or output.endswith(("/", "\\")):
        target.mkdir(parents=True, exist_ok=True)
        return target / f"{source.stem}_nobg{suffix}"
    if target.parent != Path(""):
        target.parent.mkdir(parents=True, exist_ok=True)
    return target


def _cmd_remove(args: argparse.Namespace) -> int:
    from .images import encode_image, format_for_path, load_image
    from .session import get_remover

    remover = get_remover(args.model)
    options = dict(
        only_mask=args.only_mask,
        refine=args.refine,
        foreground_threshold=args.foreground_threshold,
        background_threshold=args.background_threshold,
        erode_size=args.erode,
        blur_radius=args.blur,
        background=args.background,
        trim_border=args.trim,
    )

    inputs = list(args.inputs)
    many = len(inputs) > 1
    to_stdout = args.output == "-"
    failures = 0

    for name in inputs:
        try:
            if name == "-":
                data = sys.stdin.buffer.read()
                image = load_image(data)
                source = Path("stdin.png")
            else:
                source = Path(name)
                image = load_image(source)

            result = remover.remove(image, **options)

            if name == "-" and args.output is None:
                to_stdout = True
            image_format = (args.image_format or "PNG").upper()
            if not to_stdout:
                suffix = f".{(args.image_format or 'png').lower().replace('jpeg', 'jpg')}"
                destination = _output_path(source, args.output, suffix, many)
                if args.image_format is None:
                    image_format = format_for_path(destination)

            payload = encode_image(result, image_format, quality=args.quality)
            if to_stdout:
                sys.stdout.buffer.write(payload)
            else:
                destination.write_bytes(payload)
                print(f"{name} -> {destination}", file=sys.stderr)
        except (ValueError, OSError) as exc:
            print(f"error: {name}: {exc}", file=sys.stderr)
            failures += 1

    return 1 if failures else 0


def _cmd_download(args: argparse.Namespace) -> int:
    from .download import DownloadError, download_model, is_cached, model_path, verify_model

    names = list(MODELS) if args.all else (args.models or [DEFAULT_MODEL])
    failures = 0

    for name in names:
        try:
            spec = get_spec(name)
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            failures += 1
            continue

        if args.check:
            if not model_path(spec).is_file():
                print(f"{spec.name}: not cached")
                continue
            ok = verify_model(spec)
            print(f"{spec.name}: {'ok' if ok else 'CORRUPT - re-download with bgremover download'}")
            failures += 0 if ok else 1
            continue

        if is_cached(spec):
            print(f"{spec.name}: already cached at {model_path(spec)}")
            continue

        print(f"{spec.name}: downloading {spec.size_mb:.1f} MB from {spec.url}", file=sys.stderr)
        try:
            path = download_model(spec, progress=True, force=True)
            print(f"{spec.name}: saved to {path}")
        except DownloadError as exc:
            print(f"error: {exc}", file=sys.stderr)
            failures += 1

    return 1 if failures else 0


def _cmd_models(args: argparse.Namespace) -> int:
    from .download import is_cached, model_path

    specs = list(MODELS.values())
    if args.json:
        import json

        print(
            json.dumps(
                [
                    {
                        "name": s.name,
                        "filename": s.filename,
                        "url": s.url,
                        "sha256": s.sha256,
                        "size_bytes": s.size_bytes,
                        "input_size": list(s.input_size),
                        "license": s.license,
                        "description": s.description,
                        "cached": is_cached(s),
                        "path": str(model_path(s)),
                    }
                    for s in specs
                ],
                indent=2,
            )
        )
        return 0

    width = max(len(s.name) for s in specs)
    print(f"{'MODEL'.ljust(width)}  {'SIZE':>8}  {'INPUT':>9}  {'CACHED':>6}  DESCRIPTION")
    for spec in specs:
        size = f"{spec.size_mb:.0f} MB"
        shape = f"{spec.input_size[0]}x{spec.input_size[1]}"
        cached = "yes" if is_cached(spec) else "no"
        default = " (default)" if spec.name == DEFAULT_MODEL else ""
        print(
            f"{spec.name.ljust(width)}  {size:>8}  {shape:>9}  {cached:>6}  "
            f"{spec.description}{default}"
        )
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    try:
        import uvicorn
    except ImportError:
        print(
            'error: the API needs the extra dependencies: pip install "bgremover[api]"',
            file=sys.stderr,
        )
        return 1

    uvicorn.run(
        "bgremover.server.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers if not args.reload else None,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns a process exit code."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
