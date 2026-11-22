#!/usr/bin/env python3
"""Checks for G3 budget allocation (must pass before GPU)."""

from __future__ import annotations

import sys
from pathlib import Path

_PAPER = Path(__file__).resolve().parents[1]
if str(_PAPER) not in sys.path:
    sys.path.insert(0, str(_PAPER))

from g_chung.budget import assert_plan_windows, equal_share_budget, target_frac_budget


def test_target_frac_windows():
    local_ns = {f"M{i:02d}": 38355 for i in range(1, 11)}
    # Vary one meter to ensure we use target's n_local, not average.
    local_ns["M01"] = 40000
    for p in (0.25, 0.5, 0.75):
        plan = target_frac_budget(local_ns, target="M01", p=p)
        assert_plan_windows(plan)
        assert plan.total == 40000
        assert plan.budgets["M01"] == round(p * 40000)
        others = sum(v for k, v in plan.budgets.items() if k != "M01")
        assert others == 40000 - plan.budgets["M01"]


def test_equal_share_matches_matched_budget_formula():
    # Numbers from a prior measurement on real meters (h48, calendar on): avg≈38355.1
    local_ns = {
        "M01": 40018,
        "M02": 40192,
        "M03": 38716,
        "M04": 37270,
        "M05": 39006,
        "M06": 36744,
        "M07": 38219,
        "M08": 39682,
        "M09": 37270,
        "M10": 36434,
    }
    plan = equal_share_budget(local_ns)
    assert_plan_windows(plan)
    avg = sum(local_ns.values()) / len(local_ns)
    per = round(avg / len(local_ns))
    assert per == 3836
    assert all(v == per for v in plan.budgets.values())
    assert plan.total == per * len(local_ns) == 38360
    # p corresponding to 1/10
    assert abs(plan.p - 0.1) < 1e-12


def test_p_one_tenth_on_target_approx():
    """Document: target_frac p=0.1 on equal-length meters equals equal share."""
    n = 10000
    local_ns = {f"M{i:02d}": n for i in range(1, 11)}
    eq = equal_share_budget(local_ns)
    tf = target_frac_budget(local_ns, target="M01", p=0.1)
    assert eq.budgets["M01"] == tf.budgets["M01"] == round(0.1 * n)
    assert eq.total == tf.total == n


def main() -> int:
    test_target_frac_windows()
    test_equal_share_matches_matched_budget_formula()
    test_p_one_tenth_on_target_approx()
    print("PASS test_budget: p windows + equal_share matches run_matched_budget.py formula")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
