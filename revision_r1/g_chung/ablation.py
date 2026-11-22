"""Component ablation codes A1–A6 for G2b (Table 18 protocol).

Source of truth (read-only upstream):
- upstream ablation script (variant codes)
  ABLATIONS (lines 14-20): A1_mlp, A2_lstm, A3_bilstm, A4_armambakan
  COMPONENT_ABLATIONS (lines 26-35): A5_no_stem, A6_no_kan
- model_factory.py @ 8582ecc lines 65-68: armambakan_nostem / armambakan_nokan switches
- PAPER_CLAIMS.md Task #71: 10 meters × 3 seeds × 4 variants (A3/A5/A6/A4) =
  120 local runs, H=48, no-calendar. Summary CSV columns match AR-MambaKAN-SCADA
  results/component_ablation_summary.csv.

BiLSTM readout affects only A3, A4, A5, A6. A1 (MLP) and A2 (uni-LSTM) have no
bidirectional state, so --readout full does not change their architecture.
"""

from __future__ import annotations

from typing import TypedDict


class AblationSpec(TypedDict):
    code: str
    model: str
    note: str
    has_bilstm: bool
    use_stem: bool | None
    use_kan: bool | None


# Codes as reported in Table 18 / component_ablation_summary.csv naming.
ABLATIONS: dict[str, AblationSpec] = {
    "A1": {
        "code": "A1_mlp_no_temporal",
        "model": "mlp",
        "note": "no recurrent temporal memory",
        "has_bilstm": False,
        "use_stem": None,
        "use_kan": None,
    },
    "A2": {
        "code": "A2_lstm_unidirectional",
        "model": "lstm",
        "note": "remove bidirectional encoder",
        "has_bilstm": False,
        "use_stem": None,
        "use_kan": None,
    },
    "A3": {
        "code": "A3_bilstm",
        "model": "bilstm",
        "note": "bidirectional temporal encoder",
        "has_bilstm": True,
        "use_stem": None,
        "use_kan": None,
    },
    "A4": {
        "code": "A4_full",
        "model": "armambakan",
        "note": "full AR-MambaKAN (stem + BiLSTM + KAN)",
        "has_bilstm": True,
        "use_stem": True,
        "use_kan": True,
    },
    "A5": {
        "code": "A5_no_stem",
        "model": "armambakan_nostem",
        "note": "AR-MambaKAN without gated conv stem",
        "has_bilstm": True,
        "use_stem": False,
        "use_kan": True,
    },
    "A6": {
        "code": "A6_no_kan",
        "model": "armambakan_nokan",
        "note": "AR-MambaKAN without KAN fusion branch",
        "has_bilstm": True,
        "use_stem": True,
        "use_kan": False,
    },
}

# Variants whose BiLSTM readout changes under --readout full (G2b GPU scope).
BILSTM_ABLATION_KEYS = tuple(k for k, v in ABLATIONS.items() if v["has_bilstm"])
# Table 18 primary columns (A1/A2 not in the 10-meter summary CSV).
TABLE18_KEYS = ("A3", "A5", "A6", "A4")


def resolve_ablation(key: str) -> AblationSpec:
    k = key.strip().upper()
    if k.startswith("A") and k[1:].isdigit():
        k = f"A{int(k[1:])}"
    if k not in ABLATIONS:
        # Also accept long codes
        for short, spec in ABLATIONS.items():
            if key == spec["code"] or key == spec["model"]:
                return spec
        raise KeyError(f"unknown ablation {key!r}; expected one of {list(ABLATIONS)}")
    return ABLATIONS[k]
