from __future__ import annotations

import argparse
import sys

from . import __version__


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="alma3", description="ALMA3 diagnostic inference runtime")
    parser.add_argument("--version", action="version", version=f"alma3 {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("infer", help="run ALMA3-Dx inference", add_help=False)
    sub.add_parser("demo", help="run the packaged example", add_help=False)
    sub.add_parser("download", help="download and verify the ALMA3 3.0.0 model", add_help=False)
    sub.add_parser("verify-release", help="verify an ALMA3 release artifact", add_help=False)
    sub.add_parser(
        "export-bedmethyl-target",
        help="export the verified ALMA3 3.0.0 GRCh38 Modkit target",
        add_help=False,
    )
    args, rest = parser.parse_known_args(argv)
    try:
        if args.command == "infer":
            from .infer import main as infer_main

            return infer_main(rest)
        if args.command == "demo":
            from .infer import demo_main

            return demo_main(rest)
        if args.command == "download":
            from .download import main as download_main

            return download_main(rest)
        if args.command == "verify-release":
            from .release import main as verify_release_main

            return verify_release_main(rest)
        if args.command == "export-bedmethyl-target":
            from .bedmethyl_target import main as export_bedmethyl_target_main

            return export_bedmethyl_target_main(rest)
    except (OSError, RuntimeError, ValueError) as error:
        print(f"alma3: error: {error}", file=sys.stderr)
        return 2
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
