"""Tests for the agents: facts sheet, charts, Critic, and the LangGraph run.

The LLM is the real LLMClient talking to a fake HTTP server, so the token
budget and fallback logic are exercised for real while replies are scripted.
"""

import json

import httpx
import pytest

from eda_engine.agents import (
    AgentConfig,
    build_charts,
    build_facts,
    critique,
    extract_numbers,
    repair,
    run_analysis,
    template_narrative,
    validate_plan,
)
from eda_engine.findings import discover_findings
from eda_engine.llm import ApiKey, KeyRing, LLMClient, TokenBudget
from eda_engine.packs import run_fintech_pack
from eda_engine.preprocess import run_preprocessing
from eda_engine.schema import infer_schema
from eda_engine.synth.credit import generate_credit_applications


@pytest.fixture(scope="module")
def dataset():
    return generate_credit_applications(n_rows=5000, seed=42)


@pytest.fixture(scope="module")
def analysed(dataset):
    prep = run_preprocessing(dataset.data, infer_schema(dataset.data, target="default_flag"))
    pack = run_fintech_pack(prep, discover_findings(prep))
    facts = build_facts(prep, pack)
    return prep, pack, facts, build_charts(facts, pack)


def fid_of(facts, column):
    return next(fid for fid, f in facts.findings.items() if f.column == column)


def good_narrative(facts):
    cs, dti, ir = fid_of(facts, "credit_score"), fid_of(facts, "debt_to_income"), fid_of(facts, "interest_rate")
    return {
        "headline": "Credit score and debt-to-income are the strongest applicant signals.",
        "sections": [
            {"title": "Credit score", "findings": [cs], "charts": ["C2"],
             "text": f"Applicants in the bottom 20% of credit_score default at 26.2%, versus 3.0% in the top 20% "
                     f"(about 8.9x) [{cs}]."},
            {"title": "Debt load", "findings": [dti], "charts": [],
             "text": "Default is associated with higher debt_to_income: 24% in the top 20% vs 5.5% in the bottom 20%."},
            {"title": "Interest rate is a lender decision", "findings": [ir], "charts": [],
             "text": "interest_rate reflects the lender's view of risk, not the applicant."},
        ],
        "caveats": ["These are associations, not proof of cause."],
    }


# ---------------------------------------------------------------------------
# Facts sheet
# ---------------------------------------------------------------------------

def test_facts_sheet_lists_findings_with_ids_and_stays_small(analysed):
    _, pack, facts, _ = analysed
    assert list(facts.findings) == [f"F{i}" for i in range(1, len(facts.findings) + 1)]
    assert all(f"{fid} [" in facts.text for fid in facts.findings)
    assert facts.token_estimate < 2000          # the narrator call fits the 10k run budget with room to spare
    assert "postal_area may be a proxy for age" in facts.text
    assert "days_past_due_current" in facts.text   # leakage is reported
    assert facts.lender_decisions == {fid_of(facts, "interest_rate")}


def test_small_datasets_get_a_caveat():
    data = generate_credit_applications(n_rows=1000, seed=1).data
    prep = run_preprocessing(data, infer_schema(data, target="default_flag"))
    facts = build_facts(prep, run_fintech_pack(prep, discover_findings(prep)))
    assert any("only 1,0" in c for c in facts.caveats)


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

def test_charts_are_vega_lite_built_from_evidence(analysed):
    _, pack, facts, charts = analysed
    assert [c.id for c in charts] == [f"C{i}" for i in range(1, len(charts) + 1)]
    assert all(c.spec["$schema"].startswith("https://vega.github.io/schema/vega-lite/") for c in charts)
    cs = next(c for c in charts if c.finding_id == fid_of(facts, "credit_score"))
    woe = next(v for v in pack.variable_strength if v.column == "credit_score").woe
    assert [r["rate"] for r in cs.spec["data"]["values"]] == [w["bad_rate"] for w in woe]
    assert cs.spec["layer"][1]["encoding"]["y"]["datum"] == pack.findings.target_rate
    json.dumps([c.to_dict() for c in charts])


def test_categorical_finding_gets_a_rate_by_group_chart(analysed):
    _, pack, facts, charts = analysed
    assert len(charts) == 9 and charts[-1].kind == "variable_strength"   # 8 finding charts + IV overview
    every = build_charts(facts, pack, max_charts=20)
    term = next(c for c in every if c.finding_id == fid_of(facts, "loan_term_months"))
    assert term.kind == "rate_by_group"                                   # 36 vs 60 months: 2 groups
    assert {r["group"] for r in term.spec["data"]["values"]} == {"36", "60"}


# ---------------------------------------------------------------------------
# Critic
# ---------------------------------------------------------------------------

def test_extract_numbers_ignores_identifiers_and_reads_units():
    nums = extract_numbers("bureau_attr_051 in F3 over 3m: 26.2% vs 3% (8.9x), 5,000 rows, -0.12")
    assert [(n.value, n.unit) for n in nums] == [(26.2, "pct"), (3.0, "pct"), (8.9, "ratio"),
                                                 (5000.0, "plain"), (-0.12, "plain")]


def test_a_faithful_narrative_passes(analysed):
    *_, facts, _ = analysed
    verdict = critique(good_narrative(facts), facts)
    assert verdict.passed, verdict.feedback()


@pytest.mark.parametrize("claim, ok", [
    ("26.2%", True), ("26%", True), ("26.24%", False), ("31%", False),
    ("27%", False),          # 27.2% is on the sheet, but for interest_rate, not credit score
    ("8.9x", True), ("9x", True), ("10x", False), ("26.2x", False),
])
def test_numbers_must_match_the_sheet_up_to_rounding(analysed, claim, ok):
    *_, facts, _ = analysed
    narrative = {"headline": f"Low credit scores show {claim}.", "sections": [], "caveats": []}
    assert critique(narrative, facts).passed is ok


def test_true_numbers_attached_to_the_wrong_variable_are_caught(analysed):
    *_, facts, _ = analysed
    swapped = {"headline": "Report", "caveats": [], "sections": [{
        "title": "Debt", "findings": [], "charts": [],
        "text": "debt_to_income: 26.2% in the top 20% vs 3.0% in the bottom 20%."}]}   # credit_score's numbers
    issues = critique(swapped, facts).issues
    assert {i.kind for i in issues} == {"unverified_number"} and len(issues) == 2
    assert "facts for " + fid_of(facts, "debt_to_income") in issues[0].detail


def test_section_citations_scope_sentences_that_name_no_variable(analysed):
    *_, facts, _ = analysed
    cs = fid_of(facts, "credit_score")
    section = {"title": "Scores", "findings": [cs], "charts": [], "text": "The worst fifth defaults at 26.2%."}
    assert critique({"headline": "R", "sections": [section], "caveats": []}, facts).passed
    section["text"] = "The worst fifth defaults at 23.7%."          # debt_to_income's number
    assert not critique({"headline": "R", "sections": [section], "caveats": []}, facts).passed


def test_invented_numbers_and_unknown_citations_are_caught(analysed):
    *_, facts, _ = analysed
    bad = good_narrative(facts)
    bad["sections"][1]["text"] += " Over 40% of high-DTI applicants default."
    bad["sections"][0]["findings"].append("F99")
    kinds = {(i.kind, i.location) for i in critique(bad, facts).issues}
    assert ("unverified_number", "section 2") in kinds
    assert ("unknown_citation", "section 1") in kinds


def test_causal_language_is_caught_outside_caveats(analysed):
    *_, facts, _ = analysed
    bad = good_narrative(facts)
    bad["sections"][0]["text"] = "A low credit_score causes default."
    issues = critique(bad, facts).issues
    assert [i.kind for i in issues] == ["causal_language"]


def test_tentative_findings_must_be_labelled():
    data = generate_credit_applications(n_rows=1000, seed=0).data
    prep = run_preprocessing(data, infer_schema(data, target="default_flag"))
    facts = build_facts(prep, run_fintech_pack(prep, discover_findings(prep)))
    tentative = sorted(facts.tentative)[0]
    section = {"title": "A weak signal", "text": "This variable is associated with default.", "findings": [tentative]}
    narrative = {"headline": "Report", "sections": [section], "caveats": []}
    assert [i.kind for i in critique(narrative, facts).issues] == ["missing_tentative"]
    section["text"] += " The evidence is tentative."
    assert critique(narrative, facts).passed


def test_repair_removes_only_failing_sentences(analysed):
    *_, facts, _ = analysed
    bad = good_narrative(facts)
    bad["sections"][1]["text"] += " Over 40% of high-DTI applicants default."
    bad["sections"][0]["findings"].append("F99")
    fixed = repair(bad, critique(bad, facts), facts)
    assert critique(fixed, facts).passed
    assert "40%" not in json.dumps(fixed)
    assert "5.5%" in fixed["sections"][1]["text"]            # the true sentence survives
    assert "F99" not in fixed["sections"][0]["findings"]


def test_template_narrative_always_passes_the_critic(analysed):
    *_, facts, charts = analysed
    template = template_narrative(facts, charts)
    assert critique(template, facts).passed, critique(template, facts).feedback()
    assert template["sections"][0]["findings"] == [fid_of(facts, "credit_score")]
    assert any("lender" in s["title"] for s in template["sections"])


def test_planner_output_is_restricted_to_the_fixed_vocabulary():
    plan, rejected = validate_plan({"steps": ["cap_outliers", "drop_column_age", "missing_indicators", 7]})
    assert plan == ["missing_indicators", "cap_outliers"]
    assert rejected == ["drop_column_age"]
    assert validate_plan("not a dict") == ([], [])


# ---------------------------------------------------------------------------
# The graph, end to end
# ---------------------------------------------------------------------------

def reply(content, usage=(1500, 600)):
    return httpx.Response(200, json={"choices": [{"message": {"content": content}}],
                                     "usage": {"prompt_tokens": usage[0], "completion_tokens": usage[1]}})


class FakeLLMServer:
    """Answers planner prompts with `plan` and narrator prompts from a script."""

    def __init__(self, narratives, plan=None, fail=False):
        self.narratives = list(narratives)
        self.plan = plan or {"steps": ["missing_indicators"], "reason": "informative missingness"}
        self.fail = fail
        self.calls: list[str] = []

    def __call__(self, request):
        if self.fail:
            return httpx.Response(503)
        system = json.loads(request.content)["messages"][0]["content"]
        if system.startswith("You choose optional preprocessing"):
            self.calls.append("plan")
            return reply(json.dumps(self.plan), usage=(700, 120))
        self.calls.append("narrate")
        item = self.narratives.pop(0) if len(self.narratives) > 1 else self.narratives[0]
        return reply(item if isinstance(item, str) else json.dumps(item))


def llm_for(server, budget=10_000):
    keys = KeyRing(server={"groq": ApiKey("gsk_fake_key_for_tests")})
    return LLMClient(keys, budget=TokenBudget(budget), transport=httpx.MockTransport(server), sleep=lambda s: None)


def test_without_an_llm_the_report_is_template_and_verified(dataset):
    events = []
    report = run_analysis(dataset.data, "default_flag", on_event=lambda s, d: events.append(s))
    assert report.narrative_source == "template" and report.plan_source == "default"
    assert report.verification["status"] == "passed" and report.verification["remaining_issues"] == []
    assert events == ["plan", "analyse", "narrate", "critic", "finalize"]
    restored = json.loads(json.dumps(report.to_dict()))
    assert restored["charts"] and restored["findings"]["target"] == "default_flag"


def test_good_draft_passes_first_time_within_budget(dataset, analysed):
    *_, facts, _ = analysed
    server = FakeLLMServer([good_narrative(facts)])
    llm = llm_for(server)
    report = run_analysis(dataset.data, "default_flag", llm=llm)
    assert server.calls == ["plan", "narrate"]
    assert report.plan == ["missing_indicators"] and report.plan_source == "llm"
    assert report.narrative_source == "llm" and report.verification["status"] == "passed"
    assert report.tokens["used"] == 700 + 120 + 1500 + 600 <= 10_000


def test_bad_draft_is_revised_with_critic_feedback(dataset, analysed):
    *_, facts, _ = analysed
    bad = good_narrative(facts)
    bad["sections"][1]["text"] = "High debt_to_income causes default in 41% of cases."
    server = FakeLLMServer([bad, good_narrative(facts)])
    report = run_analysis(dataset.data, "default_flag", llm=llm_for(server))
    assert server.calls == ["plan", "narrate", "narrate"]
    assert report.verification["status"] == "passed_after_revision" and report.verification["revisions"] == 1
    first_round = report.verification["rounds"][0]["issues"]
    assert {i["kind"] for i in first_round} == {"unverified_number", "causal_language"}


def test_revision_prompt_contains_the_critic_feedback(dataset, analysed):
    *_, facts, _ = analysed
    bad = good_narrative(facts)
    bad["headline"] = "Defaults run at 99%."
    seen = []

    class Recording(FakeLLMServer):
        def __call__(self, request):
            seen.append(json.loads(request.content)["messages"])
            return super().__call__(request)

    run_analysis(dataset.data, "default_flag", llm=llm_for(Recording([bad, good_narrative(facts)])),
                 config=AgentConfig(llm_planner=False))
    revision = seen[1]
    assert "99%" in revision[-1]["content"] and "does not match" in revision[-1]["content"]
    assert revision[-2]["role"] == "assistant"


def test_stubborn_model_is_repaired_after_max_revisions(dataset, analysed):
    *_, facts, _ = analysed
    bad = good_narrative(facts)
    bad["sections"][1]["text"] += " Default reaches 41% here."
    server = FakeLLMServer([bad], plan={"steps": []})
    report = run_analysis(dataset.data, "default_flag", llm=llm_for(server, budget=20_000),
                          config=AgentConfig(max_revisions=2))
    assert server.calls.count("narrate") == 3
    assert report.verification["status"] == "repaired"
    assert "41%" not in json.dumps(report.narrative) and "5.5%" in json.dumps(report.narrative)


def test_budget_limits_revisions_before_they_are_attempted(dataset, analysed):
    *_, facts, _ = analysed
    bad = good_narrative(facts)
    bad["headline"] = "Defaults run at 99%."
    server = FakeLLMServer([bad])
    report = run_analysis(dataset.data, "default_flag", llm=llm_for(server, budget=10_000),
                          config=AgentConfig(llm_planner=False, max_revisions=5))
    # each narrate call costs 2,100 tokens and needs ~3,500 headroom: the run stops while still in budget
    assert report.tokens["used"] <= 10_000
    assert report.verification["status"] == "repaired"
    assert server.calls.count("narrate") < 6


def test_all_providers_down_still_gives_a_verified_report(dataset):
    report = run_analysis(dataset.data, "default_flag", llm=llm_for(FakeLLMServer([{}], fail=True)))
    assert report.narrative_source == "template" and report.plan_source == "default"
    assert report.verification["status"] == "passed"
    assert any("narrator unavailable" in n for n in report.notes)
    assert report.tokens["used"] == 0


def test_garbage_model_output_falls_back_to_template(dataset):
    report = run_analysis(dataset.data, "default_flag", llm=llm_for(FakeLLMServer(["I cannot do that."])),
                          config=AgentConfig(llm_planner=False))
    assert report.narrative_source == "template"
    assert report.verification["status"] == "passed"


def test_planner_cannot_smuggle_in_unknown_steps(dataset, analysed):
    *_, facts, _ = analysed
    server = FakeLLMServer([good_narrative(facts)], plan={"steps": ["drop_column:age", "cap_outliers"]})
    report = run_analysis(dataset.data, "default_flag", llm=llm_for(server))
    assert report.plan == ["cap_outliers"]
    assert any("drop_column:age" in n for n in report.notes)


# ---------------------------------------------------------------------------
# eda-analyze command
# ---------------------------------------------------------------------------

def test_env_reader_takes_only_provider_keys(tmp_path):
    from eda_engine.agents.cli import read_env_keys
    env = tmp_path / ".env"
    env.write_text('# comment\nGROQ_API_KEY="gsk_abc"\nexport OPENROUTER_API_KEY=sk-or-x  # mine\n'
                   "OPENAI_API_KEY=should_not_be_read\nCEREBRAS_API_KEY=\nDATABASE_URL=postgres://secret\n",
                   encoding="utf-8")
    assert read_env_keys(env) == {"groq": "gsk_abc", "openrouter": "sk-or-x"}


def test_cli_writes_a_json_report_without_llm(tmp_path, dataset, capsys, monkeypatch):
    from eda_engine.agents.cli import main
    csv, out = tmp_path / "data.csv", tmp_path / "report.json"
    dataset.data.to_csv(csv, index=False)
    for var in ("GROQ_API_KEY", "CEREBRAS_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    main([str(csv), "--target", "default_flag", "--out", str(out)])
    printed = capsys.readouterr().out
    assert "verification: passed" in printed and "narrative: template" in printed
    assert json.loads(out.read_text(encoding="utf-8"))["verification"]["status"] == "passed"
