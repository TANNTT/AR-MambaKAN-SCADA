"""BiLSTM readout helpers for last (paper-as-submitted) vs full (window-complete).

Source rationale: AR-MambaKAN-SCADA @ 8582ecc
  - src/models/rnn_baselines.py:18-19 uses out[:, -1, :]
  - src/models/armambakan.py:187-188 uses out[:, -1, :]
Reviewer-requested full readout: concat forward-final and backward-final states.
"""

from __future__ import annotations

import torch


def bilstm_state_readout(out: torch.Tensor, hidden: int, mode: str) -> torch.Tensor:
    """Read a bidirectional LSTM output sequence into a 2*hidden vector.

    Parameters
    ----------
    out : [B, T, 2*hidden] sequence outputs (batch_first=True)
    hidden : per-direction hidden size
    mode : "last" | "full"
      last: out[:, -1, :]  (as in 8582ecc submitted code)
      full: cat(out[:, -1, :hidden], out[:, 0, hidden:])
            = forward state after full window + backward state after full window
    """
    if mode == "last":
        return out[:, -1, :]
    if mode == "full":
        if out.size(-1) != 2 * hidden:
            raise ValueError(f"expected last dim 2*hidden={2 * hidden}, got {out.size(-1)}")
        fwd = out[:, -1, :hidden]
        bwd = out[:, 0, hidden:]
        return torch.cat([fwd, bwd], dim=-1)
    raise ValueError(f"unknown readout mode {mode!r}; expected 'last' or 'full'")
