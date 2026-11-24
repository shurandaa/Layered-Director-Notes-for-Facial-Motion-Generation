"""Encoder (paper Sec. 3.3 "Encoder", Eq. (4); Supp. B).

Clip header (neutral values, R_ref) -> states (d_tilde = G_5 * med_31(x - rest), s^(f) = g_f(d_tilde), levels ell_t
by the boundaries e^f_k, segments of at least 1 s) -> events by hysteresis on rho = x - rest - s_hat ->
tendencies.  Rule-based and deterministic; constants calibrated once on the training set.
"""
import json
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.ndimage import median_filter
from strata import decoder as _decoder
from strata import representation as _representation
from strata import tendency as _tendency
from strata import vocabulary as _vocabulary


def track_signals(track: dict, blendshape_names: list) -> tuple:
    """(x in channel space (T, K), frames valid for the neutral value, per-group validity, R_ref) of a tracked
    clip; head rotation becomes omega_t about R_ref (Sec. 3.1)."""
    f = np.asarray(track["features"], dtype=np.float64)
    n = f.shape[0]
    described = _representation.tracked_valid(track["observed"], track["quality"], track["is_gap_filled"], np.zeros(n, dtype=bool))
    r6 = f[:, _representation.channel(_representation.ROTATION_CHANNEL).slice]
    reference = _representation.reference_orientation(r6, described[:, _representation.CHANNEL_NAMES.index(_representation.ROTATION_CHANNEL)])
    bs = f[:, _representation.channel("blendshape").slice]
    keep = [k for k, name in enumerate(blendshape_names) if name not in _vocabulary.EXCLUDED]
    y = np.concatenate([bs[:, keep], _representation.omega_from_r6(r6, reference), f[:, _representation.channel("headpose_translation").slice],
                        f[:, _representation.channel("gaze").slice]], axis=1)
    ok = np.asarray(track["observed"], dtype=bool) & ~np.asarray(track["is_gap_filled"], dtype=bool)
    if ok.ndim > 1:
        ok = ok.all(axis=1)
    if ok.sum() <= _vocabulary.VOCABULARY["rest"]["min_ok_frames"]:
        ok = np.ones(n, dtype=bool)
    return y, ok, described, reference


def rest_vector(y: np.ndarray, ok: np.ndarray, names: list, rest_stat: str = None, vocabulary: dict = None) -> np.ndarray:
    """Neutral value of each channel (Supp. B "clip header"): one-sided families (smile, squint, ...) take the
    10th percentile of the validly tracked frames, all other channels the median."""
    vocabulary = vocabulary or _vocabulary.VOCABULARY
    rest_stat = rest_stat or vocabulary["rest"]["one_sided_stat"]
    q = float(vocabulary["rest"]["one_sided_stat_options"][rest_stat])
    one_sided = {s for fam in vocabulary["rest"]["one_sided_families"] for s in _vocabulary.channels_of(fam, vocabulary)}
    pct = np.array([q if name in one_sided else 50.0 for name in names])
    out = np.array([np.percentile(y[ok, k], pct[k]) for k in range(len(names))])
    return np.round(out, vocabulary["rest"]["decimals"]) + 0.0


def smooth_d_tilde(dev: np.ndarray, vocabulary: dict = None) -> np.ndarray:
    """d_tilde of Eq. (4): running median over 31 frames, then a Gaussian with sigma = 5 frames (both mode
    "nearest"), per channel; `dev` = x - rest."""
    st = (vocabulary or _vocabulary.VOCABULARY)["state"]
    m = median_filter(dev, size=(int(st["median_frames"]),) + (1,) * (dev.ndim - 1), mode="nearest")
    return gaussian_filter1d(m, float(st["gauss_sigma_frames"]), axis=0, mode="nearest")


def _runs(labels: np.ndarray) -> list:
    """[[start, stop, label]] of equal consecutive labels."""
    labels = np.asarray(labels)
    cuts = np.flatnonzero(np.diff(labels)) + 1
    starts = np.concatenate([[0], cuts])
    stops = np.concatenate([cuts, [len(labels)]])
    return [[int(a), int(b), int(labels[a])] for a, b in zip(starts, stops)]


def _true_runs(mask: np.ndarray) -> list:
    return [(a, b) for a, b, v in _runs(np.asarray(mask, dtype=int)) if v] if len(mask) else []


def g_f(d_tilde: np.ndarray, names: list, vocabulary: dict = None) -> dict:
    """s^(f)_t = g_f(d_tilde_t) for every timeline (Eq. (4); combination rule g_f in Supp. B "states"):
    one-sided families average their channels (clipped at 0), eyelids = closing minus widening, mouth tension =
    the strongest of its four kinds (plus the per-frame kind), head and gaze one timeline per axis."""
    vocabulary = vocabulary or _vocabulary.VOCABULARY
    ix = {n: k for k, n in enumerate(names)}
    mean = lambda chans: d_tilde[:, [ix[c] for c in chans]].mean(axis=1)
    out = {}
    for tl, fam, axis in _vocabulary.timelines(vocabulary):
        spec = vocabulary["families"][fam]
        if spec["type"] == "vector":
            out[tl] = d_tilde[:, ix[_vocabulary._axis_signal(spec, axis)]].copy()
        elif spec["type"] == "bs_diff":
            out[tl] = mean(spec["plus"]) - mean(spec["minus"])
        elif spec["type"] == "bs_mean":
            out[tl] = np.maximum(mean(spec["channels"]), 0.0)
        else:
            ratios = np.stack([mean(k["channels"]) / k["dev_high"] for k in spec["kinds"].values()], axis=1)
            out[tl] = np.maximum(ratios.max(axis=1), 0.0)
            out[tl + "/kind"] = ratios.argmax(axis=1)
            out[tl + "/ratios"] = ratios
    return out


def frame_levels(s_f: np.ndarray, e_f: list) -> np.ndarray:
    """ell_t of Eq. (4): sign(s^(f)_t) * #{k : e^f_k <= |s^(f)_t|}; 0 neutral, 1 slight, 2 moderate,
    3 strong."""
    level = np.searchsorted(np.asarray(e_f, dtype=np.float64), np.abs(s_f), side="right")
    return (np.sign(s_f) * level).astype(int)


def merge_runs(levels: np.ndarray, min_hold: int) -> list:
    """Segments of equal level; while a segment is shorter than min_hold (1 s), merge the shortest into the
    neighbour with the closest level (Supp. B "states")."""
    runs = _runs(levels)
    while len(runs) > 1:
        lengths = [b - a for a, b, _ in runs]
        i = min(range(len(runs)), key=lambda k: (lengths[k], k))
        if lengths[i] >= min_hold:
            break
        nb = [j for j in (i - 1, i + 1) if 0 <= j < len(runs)]
        j = min(nb, key=lambda j: (abs(runs[j][2] - runs[i][2]), -lengths[j], j))
        runs[i][2] = runs[j][2]
        merged = [runs[0]]
        for r in runs[1:]:
            if r[2] == merged[-1][2]:
                merged[-1] = [merged[-1][0], r[1], r[2]]
            else:
                merged.append(r)
        runs = merged
    return runs


def state_time_points(s: np.ndarray, runs: list, vocabulary: dict = None) -> list:
    """The four time points of each state (Supp. B "states"): for each boundary, the frames at which the
    state value has covered 10 % and 90 % of the way between the two plateaus."""
    st = (vocabulary or _vocabulary.VOCABULARY)["state"]
    v = [0.0 if lab == 0 else float(s[a:b].mean()) for a, b, lab in runs]
    out = []
    for i in range(len(runs) - 1):
        k = runs[i + 1][0]
        lo, hi = (runs[i][0] + k) // 2, (k + runs[i + 1][1]) // 2
        if v[i + 1] == v[i]:
            out.append((k, k))
            continue
        p = (s - v[i]) / (v[i + 1] - v[i])
        t0 = next((f for f in range(k, lo - 1, -1) if p[f] <= st["release_frac"]), lo)
        t1 = next((f for f in range(k, hi + 1) if f < len(s) and p[f] >= st["settled_frac"]), hi)
        out.append((t0, t1))
    return out


def encode_state(d_tilde: np.ndarray, names: list, n: int, hz: float, fitted: dict, vocabulary: dict = None) -> list:
    """The state layer S of Eq. (1) for all 14 timelines: levels by Eq. (4), segments of at least 1 s,
    four time points tau and plateau value v (the mean of s^(f) on the plateau)."""
    vocabulary = vocabulary or _vocabulary.VOCABULARY
    ix = {nm: k for k, nm in enumerate(names)}
    s_f = g_f(d_tilde, names, vocabulary)
    lv = vocabulary["state"]["levels"]
    entries = []
    for tl, fam, axis in _vocabulary.timelines(vocabulary):
        spec = vocabulary["families"][fam]
        s = s_f[tl]
        runs = merge_runs(frame_levels(s, _vocabulary.level_boundaries(tl, fitted)), int(vocabulary["state"]["min_hold_frames"]))
        cuts = state_time_points(s, runs, vocabulary)
        for i, (a, b, lab) in enumerate(runs):
            if lab == 0:
                continue
            start, settled = cuts[i - 1] if i > 0 else (0, 0)
            release, end = cuts[i] if i < len(cuts) else (n, n)
            plateau = slice(settled, release) if release > settled else slice(a, b)
            e = {"family": fam, "layer": "state", "t": [_vocabulary.t_of_frame(x, hz) for x in (start, settled, release, end)],
                 "level": lv[abs(lab) - 1], "value": _vocabulary._round_step(s[plateau].mean(), spec["state_step"])}
            if spec["type"] == "bs_tension":
                kinds = list(spec["kinds"])
                e["kind"] = kinds[int(np.argmax(s_f[tl + "/ratios"][plateau].mean(axis=0)))]
            if spec["type"] in ("vector", "bs_diff"):
                e["dir"] = (spec["dir"][axis] if axis else spec["dir"])["+" if lab > 0 else "-"]
            pair = _vocabulary._pair_of(fam, e, vocabulary)
            if pair:
                e["asym"] = _vocabulary._asym(d_tilde[plateau, ix[pair[0]]].mean(), d_tilde[plateau, ix[pair[1]]].mean(), vocabulary)
                e["side"] = _vocabulary._side(e["asym"], vocabulary)
            entries.append({k: e[k] for k in _vocabulary.STATE_KEYS if k in e})
    return entries


def _detect_spec(fam: str, vocabulary: dict) -> tuple:
    """(driver signals, per-driver (dev_low, dev_high), signed?) of a family's event detector."""
    spec = vocabulary["families"][fam]
    ev = spec["event"]
    if spec["type"] == "bs_tension":
        drivers = [(c, k["dev_low"], k["dev_high"]) for k in spec["kinds"].values() for c in k["channels"]]
        return drivers, False
    if spec["type"] == "vector":
        return [(_vocabulary._axis_signal(spec, a), ev["dev_low"], ev["dev_high"]) for a in spec["axes"]], True
    return [(c, ev["dev_low"], ev["dev_high"]) for c in ev["drivers"]], False


def _duration_frames(ev: dict, hz: float) -> tuple:
    return tuple(int(np.floor(ev[k] * hz + 0.5)) for k in ("min_s", "max_s"))


def detect_events(fam: str, rho: np.ndarray, covered: np.ndarray, names: list, hz: float, vocabulary: dict = None) -> list:
    """Event candidates of one family on the residual rho (Sec. 3.3; Supp. B, detection equation): runs with
    rho >= theta_low that reach theta_high (both signs for directed families), overlapping runs of the family's
    channels merged, duration within [d_min, d_max].  Returns [(tau0, tau3, top channel, first peak frame, tau1,
    tau2)]; the apex is the run around the first peak above 90 % of the peak value."""
    vocabulary = vocabulary or _vocabulary.VOCABULARY
    ix = {nm: k for k, nm in enumerate(names)}
    drivers, signed = _detect_spec(fam, vocabulary)
    ev = vocabulary["families"][fam]["event"]
    runs = []
    for name, theta_low, theta_high in drivers:
        r = rho[:, ix[name]]
        for sign in ((1.0, -1.0) if signed else (1.0,)):
            for a, b in _true_runs((sign * r >= theta_low) & covered):
                if (sign * r[a:b]).max() >= theta_high:
                    runs.append((a, b, name))
    runs.sort()
    clusters = []
    for a, b, name in runs:
        if clusters and a < clusters[-1][1]:
            clusters[-1] = (clusters[-1][0], max(clusters[-1][1], b), clusters[-1][2] | {name})
        else:
            clusters.append((a, b, {name}))
    d_min, d_max = _duration_frames(ev, hz)
    out = []
    for a, b, active in clusters:
        if not d_min <= b - a <= d_max:
            continue
        top = max(sorted(active), key=lambda nm: np.abs(rho[a:b, ix[nm]]).max())
        r = np.abs(rho[a:b, ix[top]])
        peak = r.max()
        f = int(np.flatnonzero(r == peak)[0])
        high = r >= vocabulary["event"]["apex_frac"] * peak
        a0 = f
        while a0 > 0 and high[a0 - 1]:
            a0 -= 1
        a1 = f + 1
        while a1 < len(r) and high[a1]:
            a1 += 1
        out.append((a, b, top, a + f, a + a0, a + a1))
    return out


def _event_core(fam: str, cand: tuple, rho: np.ndarray, names: list, vocabulary: dict) -> dict:
    """The level-free description of one candidate: kind / direction, amplitude A_e (the residual at the
    first peak frame), and the asymmetry."""
    spec = vocabulary["families"][fam]
    ix = {nm: k for k, nm in enumerate(names)}
    a, b, top, f, a0, a1 = cand
    out = {"frames": (a, a0, a1, b)}
    if spec["type"] == "vector":
        axes = spec["axes"]
        r_f = np.array([rho[f, ix[_vocabulary._axis_signal(spec, ax)]] for ax in axes])
        k = int(np.argmax(np.abs(r_f)))
        axis = axes[k]
        out.update(axis=axis, dir=spec["dir"][axis]["+" if r_f[k] > 0 else "-"],
                   value=float(r_f[k]), vector=r_f.tolist())
        if "kind" in spec["event"]:
            out["kind"] = spec["event"]["kind"][axis]
        return out
    if spec["type"] == "bs_tension":
        kinds = list(spec["kinds"])
        score = [np.mean([np.abs(rho[a0:a1, ix[c]]).mean() for c in k["channels"]]) / k["dev_high"]
                 for k in spec["kinds"].values()]
        kind = kinds[int(np.argmax(score))]
        drivers = spec["kinds"][kind]["channels"]
        out["kind"] = kind
    else:
        drivers = spec["event"]["drivers"]
        if "kind" in spec["event"]:
            out["kind"] = spec["event"]["kind"]
    g = rho[:, [ix[c] for c in drivers]].mean(axis=1)
    out.update(value=float(g[f]), g_apex=g[a0:a1], kind_key=out.get("kind", fam))
    pair = _vocabulary._pair_of(fam, out, vocabulary)
    if pair:
        out["asym"] = _vocabulary._asym(rho[a0:a1, ix[pair[0]]].mean(), rho[a0:a1, ix[pair[1]]].mean(), vocabulary)
    return out


def _event_entry(fam: str, core: dict, hz: float, fitted: dict, vocabulary: dict) -> dict:
    spec = vocabulary["families"][fam]
    step = spec["event"].get("value_step", spec["state_step"])
    key = _vocabulary.event_level_key(fam, core)
    e = {"family": fam, "layer": "event", "t": [_vocabulary.t_of_frame(x, hz) for x in core["frames"]]}
    mag = abs(_vocabulary._round_step(core["value"], step))
    e["level"] = vocabulary["event"]["levels"][int(np.searchsorted(fitted["event_edges"][key], mag, side="right"))]
    if spec["type"] == "vector" and spec["event"].get("value_vector"):
        e["value"] = [_vocabulary._round_step(x, step) for x in core["vector"]]
    else:
        e["value"] = _vocabulary._round_step(core["value"], step)
    for k in ("kind", "dir", "asym"):
        if k in core:
            e[k] = core[k]
    if "asym" in core:
        e["side"] = _vocabulary._side(core["asym"], vocabulary)
    return {k: e[k] for k in _vocabulary.EVENT_KEYS if k in e}


def _repeat_ok(units: list, gap_max: int, rp: dict, n: int) -> bool:
    """Whether a run of event units is a regular repetition (Supp. A "rules"): onset intervals and durations
    within tolerance, the first unit no longer than the period, and the last unit inside the clip."""
    if any(u2[0] - u1[3] > gap_max for u1, u2 in zip(units, units[1:])):
        return False
    iv = np.diff([u[0] for u in units])
    du = np.array([u[3] - u[0] for u in units])
    mi, md = float(np.median(iv)), float(np.median(du))
    if np.any(np.abs(iv - mi) > max(rp["interval_tol_frames"], rp["interval_tol_frac"] * mi)):
        return False
    if np.any(np.abs(du - md) > max(rp["duration_tol_frames"], rp["duration_tol_frac"] * md)):
        return False
    period = int(np.floor(mi + 0.5))
    return du[0] <= period and units[0][3] + (len(units) - 1) * period <= n


def _period(run: list, hz: float) -> int:
    """The median onset interval of a run of events, in whole frames (halves up)."""
    return int(np.floor(float(np.median(np.diff([_vocabulary.tau_frames(r["t"], hz)[0] for r in run]))) + 0.5))


def compress_repeats(events: list, n: int, hz: float, vocabulary: dict = None) -> list:
    """Repeated identical events (same kind, direction, side and level, regular spacing) become one entry
    with repeat and period (Supp. A "rules")."""
    vocabulary = vocabulary or _vocabulary.VOCABULARY
    rp = vocabulary["event"]["repeat"]
    gap_max = int(np.floor(rp["max_gap_s"] * hz + 1e-9))
    same = lambda e1, e2: all(e1.get(k) == e2.get(k) for k in ("kind", "dir", "side", "level"))
    out, i = [], 0
    while i < len(events):
        j = i
        while j + 1 < len(events) and same(events[i], events[j + 1]) and \
                _repeat_ok([_vocabulary.tau_frames(e["t"], hz) for e in events[i:j + 2]], gap_max, rp, n):
            j += 1
        # guard: the decoded last unit (first unit shifted) must end before the family's next event starts
        while j > i and j + 1 < len(events) and _vocabulary.tau_frames(events[i]["t"], hz)[3] + (j - i) * _period(events[i:j + 1], hz) \
                > _vocabulary.tau_frames(events[j + 1]["t"], hz)[0]:
            j -= 1
        if j - i + 1 >= rp["min_units"]:
            run = events[i:j + 1]
            spec = vocabulary["families"][run[0]["family"]]
            step = spec["event"].get("value_step", spec["state_step"])
            e = dict(run[0])
            v = [r["value"] for r in run]
            e["value"] = ([_vocabulary._round_step(x, step) for x in np.mean(v, axis=0)] if isinstance(v[0], list)
                          else _vocabulary._round_step(np.mean(v), step))
            if "asym" in e:
                e["asym"] = _vocabulary._round_step(np.mean([r["asym"] for r in run]), vocabulary["side"]["asym_step"])
            e["repeat"] = len(run)
            e["period"] = _vocabulary.t_of_frame(_period(run, hz), hz)
            out.append({k: e[k] for k in _vocabulary.EVENT_KEYS if k in e})
        else:
            out.append(events[i])
        i = j + 1
    return out


def _covered(fam: str, described: np.ndarray, vocabulary: dict) -> np.ndarray:
    """Frames on which every channel group holding the family's channels is validly tracked."""
    spec = vocabulary["families"][fam]
    groups = {"rotation": [_representation.ROTATION_CHANNEL], "translation": ["headpose_translation"],
              "gaze": ["gaze"] + (["blendshape"] if spec.get("members") else [])}
    schema = groups[spec["group"]] if spec["type"] == "vector" else ["blendshape"]
    return described[:, [_representation.CHANNEL_NAMES.index(s) for s in schema]].all(axis=1)


def encode_events(rho: np.ndarray, described: np.ndarray, names: list, n: int, hz: float, fitted: dict,
                  vocabulary: dict = None, collect: dict = None) -> list:
    """The event layer E of Eq. (1) from the residual rho = x - rest - s_hat (Sec. 3.3): detection, amplitude
    level by the training-set tertiles, then repeat compression."""
    vocabulary = vocabulary or _vocabulary.VOCABULARY
    out = []
    for fam, spec in vocabulary["families"].items():
        if "event" not in spec:
            continue
        cores = [_event_core(fam, c, rho, names, vocabulary)
                 for c in detect_events(fam, rho, _covered(fam, described, vocabulary), names, hz, vocabulary)]
        if collect is not None:
            collect.setdefault(fam, []).extend(cores)
            continue
        events = [_event_entry(fam, c, hz, fitted, vocabulary) for c in cores]
        if spec["event"].get("repeat"):
            events = compress_repeats(events, n, hz, vocabulary)
        out += events
    return out


def encode(track: dict, blendshape_names: list, hz: float = None, rest_stat: str = None, fitted: dict = None,
                 vocabulary: dict = None, sentences: list = None, words: list = None, tendency_q: dict = None) -> dict:
    """Encode a tracked clip into full director notes N = (h, T, S, E) (Eq. (1); Sec. 3.3 "Encoder").

    Inputs are only the tracked 63-D curves (`track`: features, observed, quality, is_gap_filled), the script timing
    (`sentences`, `words`: inclusive frame ranges) and the calibrated constants.  Order of dependency: clip header
    (rest, R_ref) -> states (Eq. (4)) -> events on rho = x - rest - s_hat -> tendencies (strata.tendency).  Without
    `sentences` the tendency layer is None."""
    vocabulary = vocabulary or _vocabulary.VOCABULARY
    rest_stat = rest_stat or vocabulary["rest"]["one_sided_stat"]
    fitted = fitted if fitted is not None else _vocabulary.calibration_for(rest_stat)
    hz = float(hz or vocabulary["hz"])
    names = _vocabulary.signal_names(blendshape_names)
    y, ok, described, reference = track_signals(track, blendshape_names)
    n = y.shape[0]
    rest = rest_vector(y, ok, names, rest_stat, vocabulary)
    d_tilde = smooth_d_tilde(y - rest, vocabulary)
    state = encode_state(d_tilde, names, n, hz, fitted, vocabulary)
    s_hat, _ = _decoder.decode_state(state, names, n, hz, fitted, vocabulary)
    _decoder._gaze_to_eyelook(s_hat, names, fitted, vocabulary)
    rho = y - rest - s_hat  # residual rho = x - rest - s_hat (Sec. 3.3)
    events = encode_events(rho, described, names, n, hz, fitted, vocabulary)
    out = {k: track[k] for k in _vocabulary.META_KEYS if k in track}
    out.update({"hz": hz, "n_frames": int(n), "rest": _vocabulary.rest_dict(rest, names),
                "reference": {_vocabulary.ROTATION_TEXT: {a: float(reference[k]) for k, a in enumerate(_vocabulary.ROTATION_AXES)}},
                "tendency": None, "state": state, "event": events})
    if sentences is not None:
        q = tendency_q if tendency_q is not None else _tendency.quantiles_for(rest_stat)
        segs = _tendency.segments_from_sentences(sentences, n)
        iw = _tendency.in_word_mask(words, n) if words else None
        vals = _tendency.tendency_values(track, blendshape_names, out, segs, iw, q["articulation_sigma"])
        out["tendency"] = _tendency.encode_tendency(vals, hz, q)
    return json.loads(json.dumps(out))
