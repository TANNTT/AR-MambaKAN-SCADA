import re
from pathlib import Path
from typing import List

import pandas as pd


INSTANT_COLUMNS = [
    "pc_time",
    "meter_no",
    "freq_hz",
    "pt_kw",
    "qt_kvar",
    "cos_phi",
    "ua_v",
    "ub_v",
    "uc_v",
    "ia_a",
    "ib_a",
    "ic_a",
    "phi_a",
    "phi_b",
    "phi_c",
]

ENERGY_COLUMNS = [
    "sitecode",
    "pc_time",
    "p0_plus",
    "p1_plus",
    "p2_plus",
    "p3_plus",
    "p0_minus",
    "p1_minus",
    "p2_minus",
    "p3_minus",
    "q0_plus",
    "q0_minus",
]

BILLING_COLUMNS = [
    "meter_time",
    "meter_no",
    "et_exp_kwh",
    "e1_exp_kwh",
    "e2_exp_kwh",
    "e3_exp_kwh",
    "et_imp_kwh",
    "e1_imp_kwh",
    "e2_imp_kwh",
    "e3_imp_kwh",
    "qt_exp",
    "qt_imp",
]


def parse_html_xls(file_path: str, kind: str) -> pd.DataFrame:
    """Parse EVN .xls files exported as one-line HTML tables."""
    text = Path(file_path).read_text(encoding="utf-8", errors="ignore")
    rows = re.findall(r"<TR>(.*?)</TR>", text)
    records: List[List[str]] = []
    for row in rows:
        cells = re.findall(r"<T[dD]>(.*?)</T[dD]>", row)
        if not cells:
            continue
        cleaned = [re.sub(r"<.*?>", "", c).strip() for c in cells]
        if any(c.lower() in {"pc time", "meter time", "sitecode"} for c in cleaned):
            continue
        records.append(cleaned)

    if kind == "instant":
        columns = INSTANT_COLUMNS
        time_col = "pc_time"
    elif kind == "energy":
        columns = ENERGY_COLUMNS
        time_col = "pc_time"
    elif kind == "billing":
        columns = BILLING_COLUMNS
        time_col = "meter_time"
    else:
        raise ValueError(f"Unknown kind: {kind}")

    df = pd.DataFrame(records, columns=columns[: len(records[0])] if records else columns)
    if not df.empty:
        df[time_col] = pd.to_datetime(df[time_col], format="%d/%m/%Y %H:%M", errors="coerce")
        for c in df.columns:
            if c != time_col:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=[time_col]).sort_values(time_col).reset_index(drop=True)
    return df

