"""Command-line entry points for dataset inspection and API-backed analysis."""

from __future__ import annotations

import argparse
import json
import sys

from .config import Settings
from .dataset import describe_dataframe, load_csv
from .engine import AnalysisEngine


def main() -> int:
    parser = argparse.ArgumentParser(description="DataTalk multi-provider data analysis backend")
    parser.add_argument("--env-file", default=".env", help="Configuration file (default: .env)")
    subparsers = parser.add_subparsers(dest="command", required=True)
    inspect_parser = subparsers.add_parser("inspect", help="Inspect CSV locally without API calls")
    analyze_parser = subparsers.add_parser("analyze", help="Analyze CSV with the configured model")
    for subparser in (inspect_parser, analyze_parser):
        subparser.add_argument("--csv", required=True, help="Input CSV path")
        subparser.add_argument("--encoding", default="utf-8-sig")
    analyze_parser.add_argument("--question", required=True)
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    try:
        settings = Settings.from_env(args.env_file)
        df = load_csv(args.csv, settings, args.encoding)
        if args.command == "inspect":
            print(
                json.dumps(
                    describe_dataframe(df, settings.sample_rows), ensure_ascii=False, indent=2
                )
            )
            return 0
        with AnalysisEngine(settings) as engine:
            result = engine.analyze(args.question, df)
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
        return 0 if result.success else 1
    except (ValueError, OSError, UnicodeError) as exc:
        print(f"DataTalk: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
