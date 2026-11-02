#!/usr/bin/env python3
"""
What the pipeline record says, read straight from the media database.

    python dev/insights.py                    # the last 30 days
    python dev/insights.py --days 7
    python dev/insights.py --trace <job id>   # every step of one generation
    python dev/insights.py --export run.jsonl # every generation and event, for analysis

Reads DATA_DIR from the environment (default ./data), so it also works on a copy of the
database from the server.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from media import db, insights, telemetry  # noqa: E402


def table(rows, columns):
    if not rows:
        print("  (nothing yet)")
        return
    cells = [[str(row.get(c, "") if row.get(c) is not None else "") for c in columns] for row in rows]
    widths = [max(len(c), *(len(r[i]) for r in cells)) for i, c in enumerate(columns)]
    print("  " + "  ".join(c.ljust(w) for c, w in zip(columns, widths)))
    for row in cells:
        print("  " + "  ".join(v.ljust(w) for v, w in zip(row, widths)))


def show(report):
    print(f"Last {report['days']} days: {json.dumps(report['totals'])}\n")
    print("What to look at")
    for note in report["suggestions"] or ["Nothing stands out yet."]:
        print(f"  - {note}")
    print("\nRoutes")
    table(report["routes"], ["kind", "provider", "model", "mode", "runs", "done", "failed", "median_seconds", "avg_usd", "avg_credits", "rating", "rated"])
    print("\nSteps")
    table(report["steps"], ["stage", "event", "count", "failure_rate", "median_ms", "p90_ms", "max_ms"])
    for name, rows in report["tables"].items():
        print(f"\nRatings by {name}")
        table(rows, ["value", "runs", "rated", "rating"])
    for section in ("director", "voice", "groups", "client"):
        print(f"\n{section.title()}: {json.dumps(report[section])}")
    for item in report["comments"]:
        print(f"  [{item['rating']}] {item['model']}: {item['comment']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--trace")
    parser.add_argument("--export", metavar="FILE")
    args = parser.parse_args()
    db.connect()
    if args.trace:
        print(json.dumps(telemetry.trace(args.trace), indent=2, ensure_ascii=False))
    elif args.export:
        with open(args.export, "w", encoding="utf-8") as out:
            for line in insights.export_lines(args.days):
                out.write(line + "\n")
        print(f"Wrote {args.export}")
    else:
        show(insights.report(args.days))


if __name__ == "__main__":
    main()
