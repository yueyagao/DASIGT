import numpy as np
import torch


def fit_time_cutpoints(times, events, n_bins=4, events_only=True):
    times = np.asarray(times, dtype=float)
    events = np.asarray(events, dtype=int)
    ev = times[events == 1] if events_only else times
    if len(ev) < n_bins:
        ev = times
    qs = np.linspace(0, 1, n_bins + 1)[1:-1]
    cuts = np.quantile(ev, qs) if len(ev) else np.linspace(0, 1, n_bins + 1)[1:-1]
    cuts = np.unique(cuts)
    if len(cuts) < n_bins - 1:
        lo = float(np.min(times)) if len(times) else 0.0
        hi = float(np.max(times)) if len(times) else 1.0
        if hi <= lo:
            hi = lo + max(1e-6, abs(lo) * 1e-6)
        extra = np.linspace(lo, hi, n_bins + 1)[1:-1]
        cuts = np.unique(np.concatenate([cuts, extra]))
    if len(cuts) < n_bins - 1:
        lo = float(np.min(times)) if len(times) else 0.0
        hi = float(np.max(times)) if len(times) else 1.0
        if hi <= lo:
            hi = lo + max(1e-6, abs(lo) * 1e-6)
        cuts = np.linspace(lo, hi, n_bins + 1)[1:-1]
    if len(cuts) > n_bins - 1:
        cuts = cuts[:n_bins - 1]
    return np.asarray(cuts, dtype=float)


def discretize_time(time, cutpoints, n_bins=4):
    return int(np.clip(np.searchsorted(np.asarray(cutpoints), float(time), side="right"), 0, n_bins - 1))


def discrete_survival_nll_per_sample(hazards, y, event):
    """Vectorized per-patient discrete-time survival NLL.

    event=1 means an observed event; event=0 means right censoring. For an event
    in interval y, the likelihood includes survival through earlier intervals and
    the event hazard in y. For censoring in y, it includes survival through y.
    Returns a tensor of shape [B] without a Python loop or per-patient device sync.
    """
    hazards = hazards.clamp(1e-7, 1.0 - 1e-7)
    if hazards.ndim == 1:
        hazards = hazards.unsqueeze(0)
    y = torch.as_tensor(y, device=hazards.device, dtype=torch.long).view(-1)
    event = torch.as_tensor(event, device=hazards.device, dtype=hazards.dtype).view(-1)
    if hazards.ndim != 2:
        raise ValueError(f"hazards must be [B,K] or [K], got {tuple(hazards.shape)}")
    if hazards.shape[0] != y.numel() or y.numel() != event.numel():
        raise ValueError(f"Batch mismatch: hazards={hazards.shape[0]}, y={y.numel()}, event={event.numel()}")
    y = y.clamp(0, hazards.shape[1] - 1)
    log_h = torch.log(hazards)
    log_1mh = torch.log1p(-hazards)
    intervals = torch.arange(hazards.shape[1], device=hazards.device).unsqueeze(0)
    before_mask = intervals < y.unsqueeze(1)
    through_mask = intervals <= y.unsqueeze(1)
    log_survival_before = (log_1mh * before_mask.to(log_1mh.dtype)).sum(dim=1)
    log_survival_through = (log_1mh * through_mask.to(log_1mh.dtype)).sum(dim=1)
    log_h_at_y = log_h.gather(1, y.unsqueeze(1)).squeeze(1)
    event_loss = -(log_survival_before + log_h_at_y)
    censor_loss = -log_survival_through
    return torch.where(event > 0.5, event_loss, censor_loss)


def discrete_survival_nll(hazards, y, event):
    """Mean discrete-time survival NLL over a batch."""
    return discrete_survival_nll_per_sample(hazards, y, event).mean()


def risk_from_survival(survival):
    return -survival.sum(dim=-1)


def concordance_index(times, events, risks):
    try:
        from sksurv.metrics import concordance_index_censored
        return float(concordance_index_censored(np.asarray(events, dtype=bool), np.asarray(times, dtype=float), np.asarray(risks, dtype=float))[0])
    except Exception:
        # Harrell C-index fallback: larger risk means shorter survival.
        t = np.asarray(times, float)
        e = np.asarray(events, int)
        r = np.asarray(risks, float)
        concordant = 0.0
        comparable = 0.0
        for i in range(len(t)):
            for j in range(i + 1, len(t)):
                if t[i] == t[j]:
                    if e[i] and e[j]:
                        comparable += 1
                        concordant += 0.5 if r[i] == r[j] else 0
                elif t[i] < t[j] and e[i]:
                    comparable += 1
                    concordant += 1 if r[i] > r[j] else (0.5 if r[i] == r[j] else 0)
                elif t[j] < t[i] and e[j]:
                    comparable += 1
                    concordant += 1 if r[j] > r[i] else (0.5 if r[i] == r[j] else 0)
        return concordant / comparable if comparable else float("nan")
