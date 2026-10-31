"""The Critic: checks the narrative mechanically against the facts sheet.

No LLM is involved: an LLM judging another LLM can be fooled the same way.
Rules:

  1. Every number must appear on the facts sheet, allowing only for rounding
     (23.7% may be written 24%, never 25%). Percentages are matched against
     percentages and ratios ("8.9x") against ratios. Numbers are checked in
     scope: a sentence that names a variable (or cites a finding) may only use
     that finding's numbers plus the dataset totals, so a true number can't be
     attached to the wrong variable.
  2. Every cited finding ID must exist.
  3. No causal language ("causes", "leads to"...): the engine finds
     associations, and the report must not claim more. (Caveats are exempt:
     that's where the report says what is *not* causal.)
  4. A section citing a tentative finding must say "tentative".

Failed sentences go back to the narrator with the reason; if revisions run
out, `repair()` deletes the failing sentences, so what the user reads is
always verified.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from eda_engine.agents.facts import Facts

NUMBER = re.compile(
    r"(?<![\w.])(?P<value>-?\d[\d,]*(?:\.\d+)?)(?:\s?(?P<unit>%|percent\b|x(?![a-z])|×|times\b))?(?![\w])"
)
CITATION = re.compile(r"\[?\b(F\d+)\b\]?")
CAUSAL = re.compile(r"\b(causes?|caused|causing|leads? to|led to|results? in|because of|due to)\b", re.I)
SENTENCE = re.compile(r"(?<=[.!?])\s+")
ALWAYS_ALLOWED_PLAIN = {20.0, 80.0, 100.0}       # "top 20%" and similar framing numbers


@dataclass(frozen=True)
class Number:
    value: float
    decimals: int
    unit: str            # "pct" | "ratio" | "plain"
    text: str


@dataclass(frozen=True)
class Issue:
    kind: str            # "unverified_number" | "unknown_citation" | "causal_language" | "missing_tentative"
    location: str        # "headline" | "section 2" | "caveat 1"
    sentence: str
    detail: str

    def to_dict(self) -> dict:
        return self.__dict__.copy()


@dataclass(frozen=True)
class Verdict:
    issues: list[Issue]

    @property
    def passed(self) -> bool:
        return not self.issues

    def feedback(self) -> str:
        return "\n".join(f"- {i.location}: {i.detail} -> \"{i.sentence}\"" for i in self.issues)


def extract_numbers(text: str) -> list[Number]:
    out = []
    for m in NUMBER.finditer(text):
        raw = m.group("value").replace(",", "")
        unit = m.group("unit") or ""
        kind = "pct" if unit in ("%", "percent") else "ratio" if unit in ("x", "×", "times") else "plain"
        decimals = len(raw.split(".")[1]) if "." in raw else 0
        out.append(Number(float(raw), decimals, kind, m.group(0)))
    return out


def _matches(n: Number, allowed: list[Number]) -> bool:
    tolerance = 0.5 * 10 ** (-n.decimals) + 1e-9
    for a in allowed:
        if n.unit != "plain" and a.unit != n.unit:
            continue
        if abs(n.value - a.value) <= tolerance:
            return True
    return n.unit == "plain" and n.value in ALWAYS_ALLOWED_PLAIN


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in SENTENCE.split(text.strip()) if s.strip()]


def _parts(narrative: dict):
    yield "headline", narrative.get("headline", ""), []
    for i, section in enumerate(narrative.get("sections", []), 1):
        cited = list(section.get("findings", []))
        yield f"section {i}", f"{section.get('title', '')}. {section.get('text', '')}", cited
    for i, caveat in enumerate(narrative.get("caveats", []), 1):
        yield f"caveat {i}", caveat, []


def _mentioned(sentence: str, facts: Facts) -> set[str]:
    """Finding IDs a sentence refers to: inline citations, or variable names (also with spaces)."""
    ids = set(CITATION.findall(sentence)) & set(facts.findings)
    lower = sentence.lower()
    for fid, f in facts.findings.items():
        names = {f.column.lower(), f.column.lower().replace("_", " ")}
        if any(re.search(r"(?<![a-z0-9_])" + re.escape(n), lower) for n in names):
            ids.add(fid)
    return ids


def _allowed_for(ids: set[str], facts: Facts, everything: list[Number]) -> list[Number]:
    if not ids:
        return everything
    scoped = [facts.header, *(facts.scoped.get(fid, "") for fid in ids)]
    return extract_numbers("\n".join(scoped))


def critique(narrative: dict, facts: Facts) -> Verdict:
    everything = extract_numbers(facts.text)
    issues: list[Issue] = []
    for location, text, cited in _parts(narrative):
        ids = set(cited) | set(CITATION.findall(text))
        for fid in sorted(ids - set(facts.findings)):
            issues.append(Issue("unknown_citation", location, fid, f"{fid} is not on the facts sheet"))
        for sentence in _sentences(text):
            clean = CITATION.sub("", sentence)
            scope = _mentioned(sentence, facts) or (set(cited) & set(facts.findings))
            allowed = _allowed_for(scope, facts, everything)
            for n in extract_numbers(clean):
                if not _matches(n, allowed):
                    where = f"the facts for {', '.join(sorted(scope))}" if scope else "the facts sheet"
                    issues.append(Issue("unverified_number", location, sentence,
                                        f"{n.text} does not match any number in {where}"))
            if not location.startswith("caveat") and CAUSAL.search(sentence):
                issues.append(Issue("causal_language", location, sentence,
                                    f"'{CAUSAL.search(sentence).group(0)}' claims cause; say 'is associated with'"))
        tentative = sorted(ids & facts.tentative)
        if tentative and "tentative" not in text.lower():
            issues.append(Issue("missing_tentative", location, ", ".join(tentative),
                                f"cites tentative finding(s) {', '.join(tentative)} without saying 'tentative'"))
    return Verdict(issues)


def repair(narrative: dict, verdict: Verdict, facts: Facts) -> dict:
    """Remove what failed: bad sentences, unknown citations; label tentative findings."""
    bad_sentences = {(i.location, i.sentence) for i in verdict.issues
                     if i.kind in ("unverified_number", "causal_language")}

    def keep(location: str, text: str) -> str:
        return " ".join(s for s in _sentences(text) if (location, s) not in bad_sentences)

    fixed = {"headline": keep("headline", narrative.get("headline", "")), "sections": [], "caveats": []}
    for i, section in enumerate(narrative.get("sections", []), 1):
        location = f"section {i}"
        title_ok = (location, f"{section.get('title', '')}.") not in bad_sentences
        text = keep(location, section.get("text", ""))
        text = CITATION.sub(lambda m: m.group(0) if m.group(1) in facts.findings else "", text).strip()
        cited = [f for f in section.get("findings", []) if f in facts.findings]
        if not text:
            continue
        tentative = sorted(set(cited) & facts.tentative)
        if tentative and "tentative" not in text.lower():
            text += f" Evidence for {', '.join(tentative)} is tentative."
        fixed["sections"].append({**section, "title": section.get("title", "") if title_ok else "Finding",
                                  "text": text, "findings": cited})
    fixed["caveats"] = [c for i, c in enumerate(narrative.get("caveats", []), 1) if keep(f"caveat {i}", c)]
    if not fixed["headline"]:
        fixed["headline"] = "Key findings"
    return fixed
