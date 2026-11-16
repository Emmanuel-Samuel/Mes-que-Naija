"""Experiment only: feature scaling and small sequence nets (GRU, TCN) for the model comparison.

Sequences are read at 5 fps (every 5th frame, any of 5 phases). Nothing here is used by the
submission pipeline.
"""

import copy
import warnings

import numpy as np
import torch
from torch import nn

LOG_FLOOR = 1e-4
STRIDE = 5


class Prep:
    """Median-fill NaNs, add a 0/1 column for each feature that has NaNs, then standardise."""

    def __init__(self, clip=10.0):
        self.clip = clip

    def fit(self, X_list, every=10):
        X = np.concatenate([x[::every] for x in X_list])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            med = np.nanmedian(X, axis=0)
        self.med = np.nan_to_num(med, nan=0.0).astype(np.float32)
        self.nan_cols = np.flatnonzero(np.isnan(X).any(axis=0))
        filled = np.where(np.isnan(X), self.med, X)
        self.mean = filled.mean(axis=0).astype(np.float32)
        self.std = np.maximum(filled.std(axis=0), 1e-6).astype(np.float32)
        return self

    def transform(self, X):
        nan = np.isnan(X)
        Z = (np.where(nan, self.med, X) - self.mean) / self.std
        np.clip(Z, -self.clip, self.clip, out=Z)
        return np.hstack([Z, nan[:, self.nan_cols].astype(np.float32)]).astype(np.float32)


class GRUNet(nn.Module):
    def __init__(self, d_in, hidden=64, layers=2, dropout=0.2):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU(), nn.Dropout(dropout))
        self.rnn = nn.GRU(hidden, hidden, num_layers=layers, bidirectional=True,
                          batch_first=True, dropout=dropout)
        self.out = nn.Linear(2 * hidden, 3)

    def forward(self, x):
        return self.out(self.rnn(self.inp(x))[0])


class TCNNet(nn.Module):
    """Non-causal dilated residual convolutions; receptive field about 25 s at 5 fps."""

    def __init__(self, d_in, channels=64, dilations=(1, 2, 4, 8, 16, 32), dropout=0.2):
        super().__init__()
        self.inp = nn.Conv1d(d_in, channels, 1)
        self.blocks = nn.ModuleList(
            nn.Sequential(nn.Conv1d(channels, channels, 3, padding=d, dilation=d), nn.ReLU(),
                          nn.Dropout(dropout), nn.Conv1d(channels, channels, 1))
            for d in dilations)
        self.out = nn.Conv1d(channels, 3, 1)

    def forward(self, x):
        h = self.inp(x.transpose(1, 2))
        for b in self.blocks:
            h = torch.relu(h + b(h))
        return self.out(h).transpose(1, 2)


def make_net(kind, d_in, cfg):
    if kind == "gru":
        return GRUNet(d_in, cfg["hidden"])
    return TCNNet(d_in, cfg["hidden"])


def _predict_offset(net, Z, phase, window, batch=64):
    """Log-probs [n,3] for frames phase, phase+5, ... of one segment, in overlapping windows."""
    sub = torch.from_numpy(np.ascontiguousarray(Z[phase::STRIDE]))
    n = len(sub)
    out = np.empty((n, 3), dtype=np.float32)
    if n <= window:
        starts = [0]
        window = n
    else:
        hop = window // 2
        starts = list(range(0, n - window + 1, hop))
        if starts[-1] + window < n:
            starts.append(n - window)
    margin = window // 4
    net.eval()
    with torch.no_grad():
        for b in range(0, len(starts), batch):
            chunk = starts[b:b + batch]
            x = torch.stack([sub[s:s + window] for s in chunk])
            lp = torch.log_softmax(net(x), dim=-1).numpy()
            for s, p in zip(chunk, lp, strict=True):
                lo = 0 if s == 0 else s + margin
                hi = n if s + window >= n else s + window - margin
                out[lo:hi] = p[lo - s:hi - s]
    return out


def predict_logp_seq(net, Z, window):
    """Log-probs [T,3] for every frame of one segment: frame f is read from phase f % 5."""
    out = np.empty((len(Z), 3), dtype=np.float32)
    for phase in range(STRIDE):
        out[phase::STRIDE] = _predict_offset(net, Z, phase, window)
    return np.maximum(out, np.log(LOG_FLOOR))


def _val_loss(net, val, window):
    tot, cnt = 0.0, 0
    for Z, y in val:
        lp = _predict_offset(net, Z, 2, window)
        yy = y[2::STRIDE]
        keep = yy <= 2
        tot -= float(lp[keep, yy[keep]].sum())
        cnt += int(keep.sum())
    return tot / max(cnt, 1)


def train_seq(kind, train, val, cfg, seed=0):
    """train / val: lists of (Z [T,D] float32, y [T] int). Returns the net with best val loss."""
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    net = make_net(kind, train[0][0].shape[1], cfg)
    opt = torch.optim.Adam(net.parameters(), lr=cfg["lr"], weight_decay=1e-4)
    w, bs = cfg["window"], cfg["batch"]
    lens = np.array([len(z) // STRIDE - 1 for z, _ in train], dtype=np.float64)
    probs = lens / lens.sum()
    best, best_state, bad = np.inf, None, 0
    for _ in range(cfg["epochs"]):
        net.train()
        for _ in range(cfg["steps"]):
            xs, ys = [], []
            for i in rng.choice(len(train), size=bs, p=probs):
                Z, y = train[i]
                n = (len(Z) - STRIDE) // STRIDE
                ww = min(w, n)
                phase = rng.randint(STRIDE)
                s = rng.randint(0, n - ww + 1)
                sl = slice(phase + STRIDE * s, phase + STRIDE * (s + ww), STRIDE)
                xs.append(Z[sl])
                ys.append(y[sl])
            x = torch.from_numpy(np.stack(xs))
            yt = torch.from_numpy(np.stack(ys).astype(np.int64))
            yt[yt > 2] = -100
            loss = nn.functional.cross_entropy(net(x).reshape(-1, 3), yt.reshape(-1),
                                               ignore_index=-100)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
        v = _val_loss(net, val, w)
        if v < best - 1e-4:
            best, best_state, bad = v, copy.deepcopy(net.state_dict()), 0
        else:
            bad += 1
            if bad >= cfg["patience"]:
                break
    net.load_state_dict(best_state)
    net.eval()
    return net
