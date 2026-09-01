import importlib.util
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("write_results", ROOT / "scripts" / "write_results.py")
wr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(wr)


def _rstar(rows):
    base = {"epochs": 10.0, "n_eval": 1000, "max_steps": -1, "min_seeds": 1, "min_rank_tested": 1}
    return pd.DataFrame([{**base, **r} for r in rows])


def test_threshold_at_smallest_rank_is_a_ceiling():
    # The real calibration data: ranks 4 and 64 only, r* = 4 at both N. The first version called this "fails".
    both = _rstar([
        {"task": "facts", "n": 1000, "arm": "lora", "r_star": 4, "r_star_ci": 4, "max_rank_tested": 64, "min_rank_tested": 4},
        {"task": "facts", "n": 4000, "arm": "lora", "r_star": 4, "r_star_ci": 4, "max_rank_tested": 64, "min_rank_tested": 4},
    ])
    verdict, evidence = wr.h1_facts(both)
    assert verdict == "inconclusive" and "never binds" in evidence

    lo_only = _rstar([
        {"task": "facts", "n": 250, "arm": "lora", "r_star": 1, "r_star_ci": 1, "max_rank_tested": 256},
        {"task": "facts", "n": 4000, "arm": "lora", "r_star": 4, "r_star_ci": 4, "max_rank_tested": 256},
    ])
    verdict, evidence = wr.h1_facts(lo_only)  # growth >= 4x is only a lower bound; prediction needs >= 8x
    assert verdict == "inconclusive" and "≤ 1" in evidence and "≥ 4×" in evidence

    sql = _rstar([{"task": "sql", "n": 2000, "arm": "lora", "r_star": 1, "r_star_ci": 1, "max_rank_tested": 128}])
    assert wr.h1_sql(sql) == ("supported", "r* ≤ 1 (prediction: ≤ 8)")


def test_h1_sql_verdicts():
    ok = _rstar([{"task": "sql", "n": 10000, "arm": "lora", "r_star": 4, "r_star_ci": 4, "max_rank_tested": 128}])
    assert wr.h1_sql(ok)[0] == "supported"
    bad = _rstar([{"task": "sql", "n": 10000, "arm": "lora", "r_star": 32, "r_star_ci": 32, "max_rank_tested": 128}])
    assert wr.h1_sql(bad)[0] == "fails"
    never = _rstar([{"task": "sql", "n": 10000, "arm": "lora", "r_star": None, "r_star_ci": None, "max_rank_tested": 128}])
    assert wr.h1_sql(never)[0] == "fails" and "never reached" in wr.h1_sql(never)[1]


def test_h1_facts_growth_scaled_to_n_ratio():
    rows = lambda r_hi: _rstar([
        {"task": "facts", "n": 1000, "arm": "lora", "r_star": 4, "r_star_ci": 4, "max_rank_tested": 256},
        {"task": "facts", "n": 8000, "arm": "lora", "r_star": r_hi, "r_star_ci": r_hi, "max_rank_tested": 256},
    ])
    assert wr.h1_facts(rows(16))[0] == "supported"   # 4x growth for 8x facts; needs >= 4x
    assert wr.h1_facts(rows(8))[0] == "fails"        # 2x growth
    verdict, evidence = wr.h1_facts(rows(None))      # not reached by 256 -> growth >= 512/4 = 128x
    assert verdict == "supported" and "not reached" in evidence


def test_h1_facts_needs_two_sizes():
    one = _rstar([{"task": "facts", "n": 1000, "arm": "lora", "r_star": 4, "r_star_ci": 4, "max_rank_tested": 256}])
    assert wr.h1_facts(one)[0] == "not tested"


def test_h2a_within_two_x():
    rstar = _rstar([
        {"task": "facts", "n": 1000, "arm": "lora", "r_star": 16, "r_star_ci": 16, "max_rank_tested": 256},
        {"task": "sql", "n": 10000, "arm": "lora", "r_star": 4, "r_star_ci": 4, "max_rank_tested": 128},
    ])
    energy = {("facts", 1000, 10.0, 1000, -1): {"energy_rank_90": 24.0},
              ("sql", 10000, 10.0, 1000, -1): {"energy_rank_90": 40.0}}
    verdict, evidence = wr.h2a(rstar, energy)
    assert verdict == "partly supported" and evidence.startswith("1/2")


def test_h2b_classifies_direction():
    def curves(lora_g, oracle_g):
        rows = []
        for arm, gs in (("lora", lora_g), ("oracle", oracle_g)):
            rows += [{"task": "sql", "n": 2000, "epochs": 2.0, "n_eval": 500, "max_steps": -1, "arm": arm,
                      "rank": r, "G": g} for r, g in zip((1, 4, 16), gs)]
        return pd.DataFrame(rows)

    assert "LoRA ≫ oracle" in wr.h2b(curves((0.6, 0.9, 1.0), (0.1, 0.5, 0.95)))[0][1]
    assert "oracle ≫ LoRA" in wr.h2b(curves((0.1, 0.5, 0.9), (0.6, 0.9, 1.0)))[0][1]
    assert "capacity" in wr.h2b(curves((0.5, 0.9, 1.0), (0.52, 0.88, 1.0)))[0][1]


def _curves(diff_sign: float):
    rows = []
    for arm, gs in (("lora", (0.6, 0.9, 1.0)), ("oracle", tuple(g - diff_sign for g in (0.6, 0.9, 1.0)))):
        rows += [{"task": "facts", "n": 4000, "epochs": 10.0, "n_eval": 1000, "max_steps": -1, "arm": arm,
                  "rank": r, "G": g} for r, g in zip((1, 4, 16), gs)]
    return pd.DataFrame(rows)


def _bullet_rstar(sql_r, facts_r, facts_min=1):
    return _rstar([
        {"task": "sql", "n": 2000, "arm": "lora", "r_star": sql_r, "r_star_ci": sql_r, "max_rank_tested": 128},
        {"task": "facts", "n": 250, "arm": "lora", "r_star": 1, "r_star_ci": 1, "max_rank_tested": 256},
        {"task": "facts", "n": 4000, "arm": "lora", "r_star": facts_r, "r_star_ci": facts_r, "max_rank_tested": 256,
         "min_rank_tested": facts_min},
    ])


def test_resume_bullet_uses_measured_values_and_largest_n():
    text = wr.resume_bullet(_bullet_rstar(4, 16), _curves(0.3), n_runs=55)
    assert "55-run" in text and "rank 4 on text-to-SQL" in text and "rank 16 for 4,000 injected facts" in text
    assert "beats SVD-truncated full-FT updates" in text and "0.30" in text
    assert "250" not in text  # only the largest N is quoted


def test_resume_bullet_states_ceilings_and_misses_honestly():
    text = wr.resume_bullet(_bullet_rstar(1, None), _curves(0.0), n_runs=None)
    assert "rank 1 (the smallest tested) on text-to-SQL" in text
    assert "no tested rank (up to 256) for 4,000 injected facts" in text
    assert "consistent with a capacity limit" in text and "-run" not in text


def test_resume_bullet_optimization_gap_direction():
    text = wr.resume_bullet(_bullet_rstar(4, 16), _curves(-0.2), n_runs=55)
    assert "optimization gap" in text and "0.20" in text


def test_replace_section_is_idempotent():
    readme = f"# T\n\nintro\n\n{wr.START}\nold\n{wr.END}\n\n## Tail\n"
    once = wr.replace_section(readme, "## Results\nnew\n")
    assert wr.replace_section(once, "## Results\nnew\n") == once
    assert "old" not in once and "## Tail" in once and "intro" in once
    with pytest.raises(SystemExit):
        wr.replace_section("# no markers", "x")
