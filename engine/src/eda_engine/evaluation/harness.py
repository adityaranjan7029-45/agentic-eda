"""Evaluation harness: scores the whole engine against the synthetic answer key.

Every synthetic dataset ships with a manifest of what was planted in it
(drivers, noise, leakage, a confounder, a proxy, duplicates, messy values).
The harness runs the full pipeline (schema -> preprocessing -> profile ->
findings -> fintech pack) on many datasets of different sizes and seeds and
checks each stage against that manifest.

It answers the question every reviewer asks: "how do you know it works?"
with numbers, and it doubles as a regression gate: a change that lowers
recall or lets noise through shows up as a failed check.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

from eda_engine.findings import FindingsConfig, discover_findings
from eda_engine.packs import run_fintech_pack
from eda_engine.preprocess import run_preprocessing
from eda_engine.profile import build_compact_profile
from eda_engine.schema import ColumnRole, Flag, infer_schema
from eda_engine.synth.credit import GeneratedDataset, generate_credit_applications

PROFILE_BUDGET = 2000   # tokens for the compact profile (step 4)


@dataclass(frozen=True)
class Check:
    stage: str      # schema | preprocess | profile | findings | pack
    name: str
    passed: bool
    detail: str = ""


@dataclass
class RunScore:
    """One dataset: hard checks (must always pass) plus measured quality metrics."""
    n_rows: int
    seed: int
    checks: list[Check]
    recall_by_strength: dict[str, float]   # strength -> share of planted drivers found
    planted_counts: dict[str, int]         # strength -> number of planted drivers
    n_findings: int
    n_high_confidence: int
    noise_reported: dict[str, str]         # noise column -> confidence it was reported with
    profile_tokens: int
    seconds: float

    @property
    def recall(self) -> float:
        found = sum(self.recall_by_strength[s] * n for s, n in self.planted_counts.items())
        return found / max(1, sum(self.planted_counts.values()))

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)

    def failed(self) -> list[Check]:
        return [c for c in self.checks if not c.passed]

    def to_dict(self) -> dict:
        return {
            "n_rows": self.n_rows, "seed": self.seed, "passed": self.passed,
            "recall": round(self.recall, 4), "recall_by_strength": self.recall_by_strength,
            "n_findings": self.n_findings, "n_high_confidence": self.n_high_confidence,
            "noise_reported": self.noise_reported,
            "profile_tokens": self.profile_tokens, "seconds": round(self.seconds, 2),
            "checks": [c.__dict__ for c in self.checks],
        }


# ---------------------------------------------------------------------------
# Scoring one dataset
# ---------------------------------------------------------------------------

def _schema_checks(schema, manifest) -> list[Check]:
    role = {c.name: c.role for c in schema.columns}
    flagged = {f: set(schema.with_flag(f)) for f in Flag}
    checks = [
        Check("schema", "id columns detected",
              all(role[c] == ColumnRole.ID for c in manifest["id_columns"]), str(manifest["id_columns"])),
        Check("schema", "constant columns detected",
              all(role[c] == ColumnRole.CONSTANT for c in manifest["constant_columns"])),
        Check("schema", "high-cardinality columns detected",
              all(role[c] == ColumnRole.HIGH_CARDINALITY for c in manifest["high_cardinality_columns"])),
        Check("schema", "date columns detected",
              all(role[c] == ColumnRole.DATETIME for c in manifest["date_columns"])),
        Check("schema", "mostly-missing columns flagged",
              set(manifest["mostly_missing_columns"]) <= flagged[Flag.MOSTLY_MISSING]),
        Check("schema", "leakage columns flagged, nothing else",
              flagged[Flag.LEAKAGE_SUSPECT] == set(manifest["leakage_columns"]),
              f"flagged {sorted(flagged[Flag.LEAKAGE_SUSPECT])}"),
    ]
    issues = manifest["data_quality_issues"]
    if manifest.get("messy"):
        checks += [
            Check("schema", "hidden null tokens flagged",
                  set(issues["hidden_null_tokens"]) <= flagged[Flag.HIDDEN_NULLS]),
            Check("schema", "numbers stored as text flagged",
                  set(issues["currency_strings"]) <= flagged[Flag.STORED_AS_TEXT]),
            Check("schema", "inconsistent spellings flagged",
                  set(issues["inconsistent_category_spellings"]) <= flagged[Flag.INCONSISTENT_SPELLING]),
        ]
    return checks


def _preprocess_checks(prep, manifest) -> list[Check]:
    unusable = (manifest["id_columns"] + manifest["constant_columns"]
                + manifest["high_cardinality_columns"] + manifest["leakage_columns"])
    dupes = manifest["data_quality_issues"].get("duplicate_rows", 0)
    removed = next((s.detail for s in prep.steps if s.name == "deduplicate"), "")
    return [
        Check("preprocess", "unusable columns excluded with a reason",
              all(c in prep.excluded for c in unusable), f"excluded {sorted(prep.excluded)}"),
        Check("preprocess", "duplicate rows removed", f"removed {dupes} " in removed + " ", removed),
        Check("preprocess", "target encoded as 0/1",
              set(prep.data[manifest["target"]].unique()) <= {0, 1}),
        Check("preprocess", "currency strings parsed to numbers",
              all(str(prep.data[c].dtype).startswith(("float", "int"))
                  for c in manifest["data_quality_issues"].get("currency_strings", []))),
    ]


def score_dataset(dataset: GeneratedDataset, findings_config: FindingsConfig | None = None) -> RunScore:
    m = dataset.manifest
    started = time.perf_counter()
    schema = infer_schema(dataset.data, target=m["target"])
    prep = run_preprocessing(dataset.data, schema)
    profile = build_compact_profile(dataset.data, schema, max_tokens=PROFILE_BUDGET)
    report = discover_findings(prep, findings_config)
    pack = run_fintech_pack(prep, report)
    seconds = time.perf_counter() - started

    final = pack.findings
    by_column = {f.column: f for f in final.findings}
    drivers = {f.column for f in final.drivers()}

    counts: dict[str, int] = {}
    hits: dict[str, int] = {}
    wrong_direction = []
    for d in m["planted_drivers"]:
        counts[d["strength"]] = counts.get(d["strength"], 0) + 1
        if d["column"] in drivers:
            hits[d["strength"]] = hits.get(d["strength"], 0) + 1
            if by_column[d["column"]].direction != d["direction"]:
                wrong_direction.append(d["column"])
    recall = {s: round(hits.get(s, 0) / n, 4) for s, n in counts.items()}
    strong = [d["column"] for d in m["planted_drivers"] if d["strength"] == "strong"]

    noise = {c: by_column[c].confidence for c in m["noise_columns"] if c in by_column}
    never = (set(m["leakage_columns"]) | set(m["null_effect_columns"])) & set(by_column)
    confounder = m["confounded_columns"][0]
    missing_col = m["informative_missingness"][0]["column"] + "__missing"
    proxies = {(p["feature"], p["protected"]) for p in pack.proxies}
    redundant = {frozenset(p["columns"]) for p in final.redundant_pairs}
    expected_pairs = [frozenset(p) for p in m["trivial_derived_pairs"] + m["redundant_pairs"]]

    checks = _schema_checks(schema, m) + _preprocess_checks(prep, m) + [
        Check("profile", f"profile fits {PROFILE_BUDGET} tokens", profile.token_estimate <= PROFILE_BUDGET,
              f"{profile.token_estimate} tokens, detail={profile.detail_level}"),
        Check("findings", "strong planted drivers found", all(c in drivers for c in strong)),
        Check("findings", "found drivers point the planted direction", not wrong_direction, str(wrong_direction)),
        Check("findings", "leakage and null-effect columns never reported", not never, str(sorted(never))),
        Check("findings", "informative missingness found",
              missing_col in by_column and by_column[missing_col].direction == "-"),
        Check("findings", "near-duplicate pairs reported as structure",
              all(p in redundant for p in expected_pairs)),
        Check("pack", "confounder labelled as lender decision",
              confounder["column"] not in by_column or by_column[confounder["column"]].kind == "lender_decision"),
        Check("pack", "applicant driver behind the confounder promoted",
              confounder["via"] in drivers and by_column[confounder["via"]].related_to is None),
        Check("pack", "planted proxy detected",
              all((p["column"], p["proxies_for"]) in proxies for p in m["proxy_features"]), str(sorted(proxies))),
    ]
    n_high = sum(f.confidence == "high" for f in final.findings)
    return RunScore(len(dataset.data), m["seed"], checks, recall, counts, len(final.findings), n_high,
                    noise, profile.token_estimate, seconds)


# ---------------------------------------------------------------------------
# Benchmark across sizes and seeds
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkReport:
    runs: list[RunScore]

    def by_size(self) -> dict[int, list[RunScore]]:
        out: dict[int, list[RunScore]] = {}
        for r in self.runs:
            out.setdefault(r.n_rows, []).append(r)
        return dict(sorted(out.items()))

    def summary(self) -> list[dict]:
        rows = []
        for n, runs in self.by_size().items():
            strengths = runs[0].planted_counts
            findings = sum(r.n_findings for r in runs)
            noise = [conf for r in runs for conf in r.noise_reported.values()]
            rows.append({
                "n_rows": n, "runs": len(runs), "passed": sum(r.passed for r in runs),
                "recall": round(mean(r.recall for r in runs), 4),
                **{f"recall_{s}": round(mean(r.recall_by_strength[s] for r in runs), 4) for s in strengths},
                "findings": findings,
                "noise_reported": len(noise),
                "noise_high_confidence": noise.count("high"),
                "false_discovery_share": round(len(noise) / max(1, findings), 4),
                "max_profile_tokens": max(r.profile_tokens for r in runs),
                "mean_seconds": round(mean(r.seconds for r in runs), 2),
            })
        return rows

    def false_discovery_share(self, high_confidence_only: bool = False) -> float:
        """Share of reported findings that are planted noise (an estimate of the real FDR)."""
        if high_confidence_only:
            findings = sum(r.n_high_confidence for r in self.runs)
            noise = sum(list(r.noise_reported.values()).count("high") for r in self.runs)
        else:
            findings = sum(r.n_findings for r in self.runs)
            noise = sum(len(r.noise_reported) for r in self.runs)
        return noise / max(1, findings)

    def gates(self, max_fdr: float = 0.05, max_high_confidence_fdr: float = 0.01,
              full_recall_from: int = 5000) -> list[Check]:
        """Benchmark-level targets. Rates can only be judged across many runs, not one."""
        large = [r for r in self.runs if r.n_rows >= full_recall_from]
        return [
            Check("benchmark", f"false-discovery share <= {max_fdr:.0%}",
                  self.false_discovery_share() <= max_fdr, f"{self.false_discovery_share():.2%}"),
            Check("benchmark", f"high-confidence false-discovery share <= {max_high_confidence_fdr:.0%}",
                  self.false_discovery_share(True) <= max_high_confidence_fdr,
                  f"{self.false_discovery_share(True):.2%}"),
            Check("benchmark", f"every planted driver found at >= {full_recall_from:,} rows",
                  all(r.recall == 1.0 for r in large), f"{len(large)} runs"),
            Check("benchmark", "every run passes its hard checks", all(r.passed for r in self.runs),
                  f"{sum(r.passed for r in self.runs)}/{len(self.runs)}"),
        ]

    def failure_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for r in self.runs:
            for c in r.failed():
                key = f"[{r.n_rows:,} rows] {c.stage}: {c.name}"
                counts[key] = counts.get(key, 0) + 1
        return dict(sorted(counts.items()))

    def to_dict(self) -> dict:
        return {"summary": self.summary(), "gates": [g.__dict__ for g in self.gates()],
                "false_discovery_share": round(self.false_discovery_share(), 4),
                "high_confidence_false_discovery_share": round(self.false_discovery_share(True), 4),
                "failures": self.failure_counts(), "runs": [r.to_dict() for r in self.runs]}

    def to_markdown(self) -> str:
        rows = self.summary()
        strengths = [k for k in rows[0] if k.startswith("recall_")] if rows else []
        head = (["Rows", "Runs passing all checks", "Recall"]
                + [s.removeprefix("recall_").title() for s in strengths]
                + ["Findings", "Noise reported (high conf.)", "False-discovery share", "Max profile tokens", "Sec/run"])
        lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
        for r in rows:
            cells = ([f"{r['n_rows']:,}", f"{r['passed']}/{r['runs']}", f"{r['recall']:.0%}"]
                     + [f"{r[s]:.0%}" for s in strengths]
                     + [str(r["findings"]), f"{r['noise_reported']} ({r['noise_high_confidence']})",
                        f"{r['false_discovery_share']:.1%}", str(r["max_profile_tokens"]), f"{r['mean_seconds']:.1f}"])
            lines.append("| " + " | ".join(cells) + " |")
        failures = self.failure_counts()
        lines += ["", "| Target | Result | Value |", "|---|---|---|"]
        lines += [f"| {g.name} | {'pass' if g.passed else 'FAIL'} | {g.detail} |" for g in self.gates()]
        lines += ["", "Failed per-run checks: " + ("none" if not failures else "")]
        lines += [f"- {k} ({v}x)" for k, v in failures.items()]
        return "\n".join(lines) + "\n"


def run_benchmark(sizes=(1000, 2500, 5000, 10000), seeds=range(5),
                  findings_config: FindingsConfig | None = None) -> BenchmarkReport:
    runs = [score_dataset(generate_credit_applications(n_rows=n, seed=s), findings_config)
            for n in sizes for s in seeds]
    return BenchmarkReport(runs)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Score the EDA engine against synthetic datasets with known answers.")
    parser.add_argument("--sizes", type=int, nargs="+", default=[1000, 2500, 5000, 10000])
    parser.add_argument("--seeds", type=int, default=5, help="number of seeds per size")
    parser.add_argument("--first-seed", type=int, default=0,
                        help="use seeds not seen while tuning (e.g. 100) for an honest held-out score")
    parser.add_argument("--out", type=Path, default=Path("reports"))
    args = parser.parse_args(argv)

    report = run_benchmark(args.sizes, range(args.first_seed, args.first_seed + args.seeds))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "benchmark.json").write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    (args.out / "benchmark.md").write_text(report.to_markdown(), encoding="utf-8")
    print(report.to_markdown())
    if not all(g.passed for g in report.gates()):
        raise SystemExit(1)
