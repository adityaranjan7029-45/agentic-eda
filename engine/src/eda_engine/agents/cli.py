"""`eda-analyze`: run a full analysis on a CSV from the terminal.

    eda-analyze data/samples/credit_applications.csv --target default_flag --env-file .env

Provider keys are read from the environment or, with --env-file, from a .env
file. Only GROQ_API_KEY, CEREBRAS_API_KEY and OPENROUTER_API_KEY are read;
nothing else in the file is touched, and nothing is written back. Without
any key the report is template-narrated (no LLM).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd

from eda_engine.agents.graph import run_analysis
from eda_engine.llm import KEY_ENV_VARS, ApiKey, KeyRing, LLMClient, TokenBudget


def read_env_keys(path: Path) -> dict[str, str]:
    """Provider keys from a .env file (KEY=value lines; quotes and comments allowed)."""
    wanted = {v: p for p, v in KEY_ENV_VARS.items()}
    keys = {}
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = (part.strip() for part in line.split("=", 1))
        name = name.removeprefix("export ").strip()
        value = value.split(" #", 1)[0].strip().strip('"').strip("'")
        if name in wanted and value:
            keys[wanted[name]] = value
    return keys


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Analyse a CSV and print a verified EDA report.")
    parser.add_argument("csv", type=Path)
    parser.add_argument("--target", default=None, help="outcome column (e.g. default_flag)")
    parser.add_argument("--env-file", type=Path, default=None, help="read provider keys from this .env file")
    parser.add_argument("--budget", type=int, default=10_000, help="token budget for the run")
    parser.add_argument("--no-llm", action="store_true", help="template report, no LLM calls")
    parser.add_argument("--out", type=Path, default=None, help="write the full report as JSON")
    args = parser.parse_args(argv)

    server = {p: os.environ[v] for p, v in KEY_ENV_VARS.items() if os.environ.get(v, "").strip()}
    if args.env_file:
        server.update(read_env_keys(args.env_file))
    llm = None
    if server and not args.no_llm:
        llm = LLMClient(KeyRing(server={p: ApiKey(k) for p, k in server.items()}),
                        budget=TokenBudget(args.budget))

    df = pd.read_csv(args.csv)
    report = run_analysis(df, args.target, llm=llm,
                          on_event=lambda stage, detail: print(f"  [{stage}] {detail}"))
    n = report.narrative
    print(f"\n{n['headline']}\n")
    for s in n["sections"]:
        print(f"## {s['title']}\n{s['text']}  ({', '.join(s['findings'])})\n")
    for c in n["caveats"]:
        print(f"- {c}")
    v, t = report.verification, report.tokens
    print(f"\nverification: {v['status']} (revisions: {v['revisions']}) | narrative: {report.narrative_source} "
          f"| tokens: {t['used']:,}/{t['limit']:,} | {report.seconds:.1f}s")
    for note in report.notes:
        print(f"note: {note}")
    if args.out:
        args.out.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
        print(f"full report written to {args.out}")
    if llm is not None:
        llm.close()
