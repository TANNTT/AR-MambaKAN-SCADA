"""Matched-budget window allocation for G3 (R1.3).

Upstream reference: experiments/matched_budget/run_matched_budget.py:44-60 and
experiments/pooled/common_pooled.py:136-143 @ 8582ecc.

Original equal-share protocol:
  per_meter_budget = round(avg_local_n / n_meters)
  each meter contributes that many train windows (rng.choice without replacement).

G3 target-fraction protocol (new):
  total budget = n_local of the TARGET meter
  target gets round(p * n_local)
  remaining windows split across the other meters so the sum equals n_local.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from torch.utils.data import Subset


@dataclass(frozen=True)
class BudgetPlan:
    """Per-meter train-window budgets. Keys are meter labels."""

    budgets: dict[str, int]
    n_local_target: int
    target: str
    p: float | None
    mode: str  # "target_frac" | "equal_share"

    @property
    def total(self) -> int:
        return int(sum(self.budgets.values()))


def equal_share_budget(local_ns: dict[str, int]) -> BudgetPlan:
    """Mirror run_matched_budget.py:47-49 (p ≈ 1/n_meters equal share)."""
    codes = list(local_ns.keys())
    if not codes:
        raise ValueError("empty local_ns")
    avg_local_n = sum(local_ns.values()) / len(codes)
    per_meter = round(avg_local_n / len(codes))
    budgets = {c: int(per_meter) for c in codes}
    # target unused in equal-share; keep first for metadata
    return BudgetPlan(
        budgets=budgets,
        n_local_target=int(round(avg_local_n)),
        target=codes[0],
        p=1.0 / len(codes),
        mode="equal_share",
    )


def target_frac_budget(
    local_ns: dict[str, int],
    target: str,
    p: float,
) -> BudgetPlan:
    """Total = n_local(target); target gets round(p * n_local); rest split among others."""
    if target not in local_ns:
        raise KeyError(f"target {target} not in local_ns")
    if not (0.0 < p <= 1.0):
        raise ValueError(f"p must be in (0, 1], got {p}")
    others = [c for c in local_ns if c != target]
    n_local = int(local_ns[target])
    n_target = int(round(p * n_local))
    if n_target < 0 or n_target > n_local:
        raise RuntimeError(f"n_target={n_target} out of range for n_local={n_local}")
    remaining = n_local - n_target
    budgets = {target: n_target}
    if not others:
        if remaining != 0:
            raise RuntimeError("no other meters but remaining budget")
        return BudgetPlan(budgets=budgets, n_local_target=n_local, target=target, p=p, mode="target_frac")

    base = remaining // len(others)
    extra = remaining % len(others)
    # Stable order: sort labels so remainder assignment is deterministic.
    for i, c in enumerate(sorted(others)):
        budgets[c] = base + (1 if i < extra else 0)
    if sum(budgets.values()) != n_local:
        raise RuntimeError(
            f"budget sum {sum(budgets.values())} != n_local {n_local} (p={p}, target={target})"
        )
    if budgets[target] != n_target:
        raise RuntimeError("target budget mismatch")
    return BudgetPlan(budgets=budgets, n_local_target=n_local, target=target, p=p, mode="target_frac")


def subsample_train_datasets(
    train_datasets: dict[str, object],
    budgets: dict[str, int],
    seed: int,
) -> tuple[list, int]:
    """Apply per-meter Subset sampling like common_pooled.py:136-143.

    Returns (list_of_subsets_in_label_order, pooled_n).
    """
    rng = np.random.default_rng(seed)
    subsets = []
    for label, ds in train_datasets.items():
        budget = budgets[label]
        n = len(ds)
        if budget > n:
            raise RuntimeError(f"{label}: budget {budget} > available train windows {n}")
        if budget < n:
            chosen = rng.choice(n, size=budget, replace=False)
            subsets.append(Subset(ds, chosen.tolist()))
        else:
            subsets.append(ds)
    pooled_n = sum(len(s) for s in subsets)
    return subsets, pooled_n


def assert_plan_windows(plan: BudgetPlan) -> None:
    """Hard checks required by the revision plan."""
    if plan.mode == "target_frac":
        if plan.total != plan.n_local_target:
            raise AssertionError(f"total {plan.total} != n_local {plan.n_local_target}")
        expected_target = int(round(plan.p * plan.n_local_target))
        if plan.budgets[plan.target] != expected_target:
            raise AssertionError(
                f"target budget {plan.budgets[plan.target]} != round(p*n_local)={expected_target}"
            )
    elif plan.mode == "equal_share":
        vals = list(plan.budgets.values())
        if len(set(vals)) != 1:
            raise AssertionError(f"equal_share budgets not equal: {plan.budgets}")
    else:
        raise AssertionError(f"unknown mode {plan.mode}")
