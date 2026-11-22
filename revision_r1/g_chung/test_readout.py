#!/usr/bin/env python3
"""Unit checks for BiLSTM / AR-MambaKAN readout last vs full.

(a) full: backward half equals h_n of the reverse LSTM direction.
(b) last: model output matches original out[:, -1, :] path on identical weights.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

_PAPER = Path(__file__).resolve().parents[1]
if str(_PAPER) not in sys.path:
    sys.path.insert(0, str(_PAPER))

from g_chung.models.armambakan import ARMambaKAN
from g_chung.models.rnn_baselines import BaselineBiLSTM
from g_chung.readout import bilstm_state_readout


def test_full_matches_hn_bilstm():
    torch.manual_seed(0)
    B, T, F, H = 4, 32, 7, 16
    x = torch.randn(B, T, F)
    lstm = nn.LSTM(F, H, batch_first=True, bidirectional=True)
    out, (h_n, _) = lstm(x)
    # h_n: [2, B, H] -> [0]=forward final, [1]=backward final
    z_full = bilstm_state_readout(out, H, "full")
    expected = torch.cat([h_n[0], h_n[1]], dim=-1)
    assert torch.allclose(z_full, expected, atol=1e-6, rtol=1e-5), (
        float((z_full - expected).abs().max())
    )
    # Explicit sequence positions
    assert torch.allclose(out[:, -1, :H], h_n[0], atol=1e-6)
    assert torch.allclose(out[:, 0, H:], h_n[1], atol=1e-6)


def test_last_matches_original_bilstm():
    torch.manual_seed(1)
    B, T, F, H, horizon = 3, 40, 5, 12, 4
    x = torch.randn(B, T, F)
    model = BaselineBiLSTM(F, H, horizon, readout="last")
    model.eval()
    with torch.no_grad():
        out_seq, _ = model.backbone(x)
        z_orig = out_seq[:, -1, :]
        y_orig = model.head(z_orig).view(B, horizon, 2)
        y_model = model(x)
    assert torch.allclose(y_model, y_orig, atol=0.0, rtol=0.0)


def test_last_matches_original_armambakan():
    torch.manual_seed(2)
    B, T, F, horizon = 2, 48, 7, 4
    x = torch.randn(B, T, F)
    model = ARMambaKAN(in_dim=F, d_model=32, hidden=32, horizon=horizon, kan_hidden=32, readout="last")
    model.eval()
    with torch.no_grad():
        h = model.input_proj(x)
        if model.use_stem:
            h = model.stem(h)
        out_seq, _ = model.lstm(h)
        z_orig = out_seq[:, -1, :]
        y_orig = model.head(z_orig, regime_signal=None)
        y_model = model(x)
    assert torch.allclose(y_model, y_orig, atol=0.0, rtol=0.0)


def test_full_differs_from_last_generally():
    torch.manual_seed(3)
    B, T, F, H, horizon = 2, 20, 4, 8, 4
    x = torch.randn(B, T, F)
    m_last = BaselineBiLSTM(F, H, horizon, readout="last")
    m_full = BaselineBiLSTM(F, H, horizon, readout="full")
    m_full.load_state_dict(m_last.state_dict())
    m_last.eval()
    m_full.eval()
    with torch.no_grad():
        y_last = m_last(x)
        y_full = m_full(x)
    # Same weights but different readout -> outputs should differ for random x
    assert not torch.allclose(y_last, y_full, atol=1e-5)


def main() -> int:
    test_full_matches_hn_bilstm()
    test_last_matches_original_bilstm()
    test_last_matches_original_armambakan()
    test_full_differs_from_last_generally()
    print("PASS test_readout: full==h_n reverse; last==out[:, -1, :] original path")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
