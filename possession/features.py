"""Per-frame clues for the possession model.

Everything is built from team-level aggregates and order statistics (nearest, second nearest,
deepest, most advanced), never from a track slot's identity, so track swaps, fragments and
duplicates do not change the result. Nothing here reads `label`.
Teams are measured in their own attacking direction: home attacks +x, away attacks -x.
"""

import warnings

import numpy as np

HALF_LENGTH, HALF_WIDTH = 52.5, 34.0
SPEED_STEP = 5          # frames either side for player velocity
BALL_STEP = 3           # frames either side for ball speed
MAX_PLAYER_SPEED = 12.0  # faster than this is a track swap or a jump, so the value is dropped
MAX_BALL_SPEED = 40.0
GAP_CAP_SECONDS = 120.0
WINDOWS = ((25, "1s"), (75, "3s"))

KEY_FEATURES = ("dd", "dd_fill", "dh1", "da1", "ball_vis", "ball_speed", "ball_out", "ball_x",
                "h_speed", "a_speed", "h_fwd", "a_fwd", "all_speed",
                # Clues for a stoppage while the ball is still tracked: shapes resetting, nobody
                # contesting the ball, ball moving no faster than the player next to it.
                "h_cx", "a_cx", "h_along", "a_along", "near_any2", "ball_edge", "carried_gap")
FAST_SPEED = 2.0


def _nanstat(fn, a, axis=1):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return fn(a, axis=axis)


def _central_velocity(xy, step, fps):
    """Per-slot velocity [T,P,2] from positions `step` frames either side (NaN at the ends)."""
    vel = np.full(xy.shape, np.nan, dtype=np.float32)
    vel[step:-step] = (xy[2 * step:] - xy[:-2 * step]) * (fps / (2.0 * step))
    return vel


def _team_block(xy, sign, fps):
    """Team aggregates in the team's own attacking direction (sign = +1 home, -1 away)."""
    x = xy[:, :, 0] * sign
    y = xy[:, :, 1]
    vel = _central_velocity(xy, SPEED_STEP, fps)
    speed = np.hypot(vel[:, :, 0], vel[:, :, 1])
    speed[speed > MAX_PLAYER_SPEED] = np.nan
    fwd = vel[:, :, 0] * sign
    fwd[np.isnan(speed)] = np.nan
    return {
        "n": np.sum(~np.isnan(x), axis=1).astype(np.float32),
        "cx": _nanstat(np.nanmean, x),
        "cy": _nanstat(np.nanmean, y),
        "along": _nanstat(np.nanstd, x),
        "across": _nanstat(np.nanstd, y),
        "deep": _nanstat(np.nanmin, x),
        "adv": _nanstat(np.nanmax, x),
        "speed": _nanstat(np.nanmean, speed),
        "slow": _nanstat(np.nanmean, np.where(np.isnan(speed), np.nan, speed < 1.0)),
        "fwd": _nanstat(np.nanmean, fwd),
        "_speed": speed,
    }


def player_features(match):
    """Clues from players only (no ball). Returns (X float32 [T,F], names)."""
    fps = match.fps
    blocks = {"h": _team_block(match.home_xy, 1.0, fps), "a": _team_block(match.away_xy, -1.0, fps)}
    cols = {}
    for t, b in blocks.items():
        for k, v in b.items():
            if not k.startswith("_"):
                cols["{}_{}".format(t, k)] = v
    for k in ("n", "cx", "along", "across", "deep", "adv", "speed", "slow", "fwd"):
        cols["diff_" + k] = cols["h_" + k] - cols["a_" + k]
    all_speed = np.concatenate([blocks["h"]["_speed"], blocks["a"]["_speed"]], axis=1)
    all_x = np.concatenate([match.home_xy[:, :, 0], match.away_xy[:, :, 0]], axis=1)
    cols["all_speed"] = _nanstat(np.nanmean, all_speed)
    cols["all_slow"] = _nanstat(np.nanmean, np.where(np.isnan(all_speed), np.nan, all_speed < 1.0))
    cols["all_cx"] = _nanstat(np.nanmean, all_x)
    cols["all_along"] = _nanstat(np.nanstd, all_x)
    cols.update(_half_clock(match))
    names = list(cols)
    return np.stack([cols[n] for n in names], axis=1).astype(np.float32), names


def _half_clock(match):
    out = np.full(match.n_frames, np.nan, dtype=np.float32)
    for h in (1, 2):
        idx = np.flatnonzero(match.half == h)
        if len(idx):
            out[idx] = (idx - idx[0]) / match.fps
    return {"half_seconds": out}


def _window_mean(x, lo, hi):
    """Mean of x over frames [t+lo, t+hi], ignoring NaN (NaN when the window has none)."""
    n = len(x)
    valid = ~np.isnan(x)
    csum = np.concatenate(([0.0], np.cumsum(np.where(valid, x, 0.0), dtype=np.float64)))
    ccnt = np.concatenate(([0], np.cumsum(valid)))
    t = np.arange(n)
    a = np.clip(t + lo, 0, n)
    b = np.clip(t + hi + 1, 0, n)
    cnt = ccnt[b] - ccnt[a]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(cnt > 0, (csum[b] - csum[a]) / np.maximum(cnt, 1), np.nan).astype(np.float32)


def _nearest(xy, ball):
    """Sorted distances [T,P] (NaN last) from one team's players to the ball."""
    d = np.hypot(xy[:, :, 0] - ball[:, None, 0], xy[:, :, 1] - ball[:, None, 1])
    return np.sort(d, axis=1)


def _near_block(match, ball, suffix, with_counts):
    dh, da = _nearest(match.home_xy, ball), _nearest(match.away_xy, ball)
    cols = {"dh1" + suffix: dh[:, 0], "da1" + suffix: da[:, 0],
            "dd" + suffix: dh[:, 0] - da[:, 0]}
    if with_counts:
        seen = ~np.isnan(ball[:, 0])
        cols["dh2"], cols["da2"] = dh[:, 1], da[:, 1]
        cols["dd2"] = dh[:, 1] - da[:, 1]
        for r in (2, 5):
            for t, d in (("h", dh), ("a", da)):
                cnt = np.sum(d < r, axis=1).astype(np.float32)
                cols["{}_within{}".format(t, r)] = np.where(seen, cnt, np.nan)
            cols["within{}_diff".format(r)] = cols["h_within{}".format(r)] - cols["a_within{}".format(r)]
    return cols


def _outside(ball):
    ox = np.maximum(np.abs(ball[:, 0]) - HALF_LENGTH, 0.0)
    oy = np.maximum(np.abs(ball[:, 1]) - HALF_WIDTH, 0.0)
    return np.hypot(ox, oy)


def ball_speed(ball, fps):
    """Ball speed [T] in m/s from positions BALL_STEP frames either side; NaN where unknown."""
    n = len(ball)
    disp = np.full(n, np.nan, dtype=np.float32)
    s = BALL_STEP
    disp[s:-s] = np.hypot(*(ball[2 * s:] - ball[:-2 * s]).T) * (fps / (2.0 * s))
    disp[disp > MAX_BALL_SPEED] = np.nan
    disp[np.isnan(ball[:, 0])] = np.nan
    return disp


def outside_lines(ball):
    return _outside(ball)


def _player_speed(xy, fps):
    vel = _central_velocity(xy, SPEED_STEP, fps)
    speed = np.hypot(vel[:, :, 0], vel[:, :, 1])
    speed[speed > MAX_PLAYER_SPEED] = np.nan
    return speed


def _seconds_to_flag(flag, fps, cap=GAP_CAP_SECONDS):
    """Seconds since the last True and until the next True (0 where True), capped."""
    n = len(flag)
    t = np.arange(n)
    last = np.maximum.accumulate(np.where(flag, t, -10 ** 9))
    nxt = np.minimum.accumulate(np.where(flag, t, 10 ** 9)[::-1])[::-1]
    return (np.minimum((t - last) / fps, cap).astype(np.float32),
            np.minimum((nxt - t) / fps, cap).astype(np.float32))


def _ball_block(match, all_speed):
    fps = match.fps
    ball = match.ball_xy.astype(np.float32)
    n = len(ball)
    seen = ~np.isnan(ball[:, 0])
    t = np.arange(n)
    last = np.maximum.accumulate(np.where(seen, t, -1))
    nxt = np.minimum.accumulate(np.where(seen, t, n)[::-1])[::-1]
    has_last, has_next = last >= 0, nxt < n
    since = np.where(has_last, (t - last) / fps, GAP_CAP_SECONDS)
    until = np.where(has_next, (nxt - t) / fps, GAP_CAP_SECONDS)
    since, until = np.minimum(since, GAP_CAP_SECONDS), np.minimum(until, GAP_CAP_SECONDS)
    ball_last = np.where(has_last[:, None], ball[np.clip(last, 0, n - 1)], np.nan)
    ball_next = np.where(has_next[:, None], ball[np.clip(nxt, 0, n - 1)], np.nan)

    disp = ball_speed(ball, fps)

    cols = {
        "ball_vis": seen.astype(np.float32),
        "since_seen": since.astype(np.float32),
        "until_seen": until.astype(np.float32),
        "gap_len": np.where(seen, 0.0, since + until).astype(np.float32),
        "ball_x": ball[:, 0], "ball_y": ball[:, 1], "ball_speed": disp,
        "ball_out": _outside(ball),
        "ball_x_last": ball_last[:, 0], "ball_y_last": ball_last[:, 1],
        "ball_x_next": ball_next[:, 0], "ball_y_next": ball_next[:, 1],
        "ball_out_last": _outside(ball_last), "ball_out_next": _outside(ball_next),
    }
    cols.update(_near_block(match, ball, "", True))
    cols.update(_stoppage_clues(match, ball, disp, all_speed))
    cols.update(_near_block(match, ball_last, "_last", False))
    cols.update(_near_block(match, ball_next, "_next", False))
    # Gap-filled margin: the current one when the ball is seen, else the nearer-in-time of last/next.
    nearer_last = since <= until
    fill = np.where(nearer_last, cols["dd_last"], cols["dd_next"])
    fill = np.where(np.isnan(fill), np.where(nearer_last, cols["dd_next"], cols["dd_last"]), fill)
    cols["dd_fill"] = np.where(seen, cols["dd"], fill).astype(np.float32)
    return cols


def _stoppage_clues(match, ball, disp, all_speed):
    fps = match.fps
    seen = ~np.isnan(ball[:, 0])
    both = np.concatenate([match.home_xy, match.away_xy], axis=1)
    d = np.hypot(both[:, :, 0] - ball[:, None, 0], both[:, :, 1] - ball[:, None, 1])
    speeds = np.concatenate([_player_speed(match.home_xy, fps), _player_speed(match.away_xy, fps)], axis=1)
    order = np.argsort(d, axis=1)
    sorted_d = np.take_along_axis(d, order[:, :2], axis=1)
    near_speed = np.take_along_axis(speeds, order[:, :1], axis=1)[:, 0]
    near_speed = np.where(np.isnan(sorted_d[:, 0]), np.nan, near_speed)
    ax, ay = np.abs(ball[:, 0]), np.abs(ball[:, 1])
    edge = np.minimum(HALF_LENGTH - ax, HALF_WIDTH - ay)
    corner = np.hypot(HALF_LENGTH - ax, HALF_WIDTH - ay)
    fast = np.nan_to_num(all_speed, nan=0.0) >= FAST_SPEED
    since_fast, until_fast = _seconds_to_flag(fast, fps)
    return {
        "near_any1": sorted_d[:, 0], "near_any2": sorted_d[:, 1],
        "ball_edge": edge.astype(np.float32), "ball_corner": corner.astype(np.float32),
        "near_player_speed": near_speed.astype(np.float32),
        "carried_gap": (disp - near_speed).astype(np.float32),
        "since_fast": since_fast, "until_fast": until_fast,
    }


def build_features(match, set_pieces=None):
    """Return (X float32 [T,F], names). NaN where a clue is unknown.

    set_pieces: optional [N,5] tag array (see possession.tagging); adds the tag clues.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        px, pnames = player_features(match)
        cols = {n: px[:, i] for i, n in enumerate(pnames)}
        cols.update(_ball_block(match, cols["all_speed"]))
        for key in KEY_FEATURES:
            x = cols[key].astype(np.float32)
            for w, tag in WINDOWS:
                cols["{}_m{}".format(key, tag)] = _window_mean(x, -w, w)
                cols["{}_d{}".format(key, tag)] = (_window_mean(x, 0, w) - _window_mean(x, -w, 0))
    if set_pieces is not None:
        from possession.tagging import tag_features
        tx, tnames = tag_features(set_pieces, match.n_frames, match.fps)
        cols.update({n: tx[:, i] for i, n in enumerate(tnames)})
    names = list(cols)
    X = np.stack([np.asarray(cols[n], dtype=np.float32) for n in names], axis=1)
    return X, names
