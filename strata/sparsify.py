"""Standard and lite notes from full notes by fixed rules (paper Sec. 3.2; Supp. A "densities"): the most
salient events are kept (salience = amplitude x duration), adjacent states of a family closest in amplitude are
merged until the quota holds, all tendencies are kept; lite notes also drop values, sides and repeats.
"""
import json
import math
import numpy as np
import zlib
from fractions import Fraction
from strata import decoder as _decoder
from strata import notes as _notes
from strata import tendency as _tendency
from strata import vocabulary as _vocabulary


TIERS = ("standard", "lite")


AMPLITUDES = ("value", "dev_high")


QUOTAS = ("per_minute_floor", "per_track_round_min1", "per_track_stochastic")


STATE_CAPS = ("per_track", "per_minute_ceil_min1")


LITE_STRIP = ("value", "asym", "side", "repeat", "period")


def _no_cap(x) -> bool:
    return x is None or (isinstance(x, float) and math.isinf(x) and x > 0)


def _check_cap(x, what: str) -> None:
    if _no_cap(x):
        return
    if isinstance(x, bool) or not isinstance(x, int) or x < 0:
        raise ValueError("%s must be an integer >= 0, None or inf (got %r)" % (what, x))


def _check_mode(x, allowed: tuple, what: str) -> None:
    if x not in allowed:
        raise ValueError("%s must be one of %s" % (what, allowed))


def _written(x) -> Fraction:
    """A number exactly as written in the JSON text (the shortest repr of the float)."""
    return Fraction(repr(float(x)))


def _amplitude_terms(e: dict, fitted: dict, amplitude: str, vocabulary: dict) -> tuple:
    """(components, divisor) of the amplitude as written: |components| / divisor (event_amplitude)."""
    _check_mode(amplitude, AMPLITUDES, "amplitude")
    fam = e["family"]
    spec = vocabulary["families"][fam]
    v = e.get("value")
    if v is None:
        core = {"kind": e.get("kind")}
        if spec["type"] == "vector":
            core["axis"] = _vocabulary.entry_timeline(e, vocabulary).split(".", 1)[1]
        comps = [fitted["event_repr"][_vocabulary.event_level_key(fam, core)][vocabulary["event"]["levels"].index(e["level"])]]
    else:
        comps = v if isinstance(v, list) else [v]
    div = None
    if amplitude == "dev_high":
        div = spec["kinds"][e["kind"]]["dev_high"] if spec["type"] == "bs_tension" else spec["event"]["dev_high"]
    return comps, div


def _salience_key(e: dict, hz: float, fitted: dict, amplitude: str, vocabulary: dict) -> Fraction:
    """(salience x hz)^2, exact in the written decimals: the ranking key of event_salience (module docstring)."""
    comps, div = _amplitude_terms(e, fitted, amplitude, vocabulary)
    fr = _vocabulary.tau_frames(e["t"], hz)
    sq = sum(_written(c) ** 2 for c in comps) * (fr[3] - fr[0]) ** 2
    return sq / _written(div) ** 2 if div is not None else sq


def track_u(track_id) -> float:
    """The default quota_u of per_track_stochastic: crc32(track_id) / 2^32 in [0, 1)."""
    return zlib.crc32(str(track_id or "").encode()) / 2.0 ** 32


def block_quota(k_per_min, n_frames: int, hz: float, quota: str = "per_minute_floor", quota_u: float = None) -> list:
    """Entries kept per 60 s block from frame 0: floor(K * L / (60 hz)) for a block of L frames (the last one partial).
    quota="per_track_round_min1": one block, round(K * n_frames / (60 hz)) (halves up), at least 1 when K > 0.
    quota="per_track_stochastic": one block, floor(x) + [quota_u < frac(x)], x = K * n_frames / (60 hz) exactly."""
    _check_mode(quota, QUOTAS, "quota")
    w = int(round(60.0 * hz))
    if quota == "per_track_stochastic":
        if quota_u is None or not 0.0 <= quota_u < 1.0:
            raise ValueError("per_track_stochastic needs quota_u in [0, 1) (got %r)" % (quota_u,))
        x = Fraction(k_per_min * n_frames, w)
        base = x.numerator // x.denominator
        return [base + int(Fraction(quota_u) < x - base)]
    if quota == "per_track_round_min1":
        return [max(1, (2 * k_per_min * n_frames + w) // (2 * w)) if k_per_min > 0 else 0]
    blocks = [min(w, n_frames - a) for a in range(0, n_frames, w)]
    return [k_per_min * b // w for b in blocks]


def select_events(events: list, n_frames: int, hz: float, k_per_min, fitted: dict, amplitude: str = "value",
                  vocabulary: dict = None, quota: str = "per_minute_floor", quota_u: float = None) -> list:
    """The top-K-per-minute event entries (module docstring), in their encoded order. New list, entries shared."""
    vocabulary = vocabulary or _vocabulary.VOCABULARY
    _check_cap(k_per_min, "k_per_min")
    _check_mode(quota, QUOTAS, "quota")
    if _no_cap(k_per_min):
        return list(events)
    w = int(round(60.0 * hz))
    quota = block_quota(k_per_min, n_frames, hz, quota, quota_u)
    order = list(vocabulary["families"])
    keys = []
    for i, e in enumerate(events):
        start = _vocabulary.tau_frames(e["t"], hz)[0]
        keys.append((min(start // w, len(quota) - 1), -_salience_key(e, hz, fitted, amplitude, vocabulary), start,
                     order.index(e["family"]), i))
    kept, used = set(), [0] * len(quota)
    for b, _, _, _, i in sorted(keys):
        if used[b] < quota[b]:
            used[b] += 1
            kept.add(i)
    return [e for i, e in enumerate(events) if i in kept]


def _merge_pair(p: dict, c: dict, tl: str, hz: float, fitted: dict, vocabulary: dict) -> dict:
    """Two consecutive entries of one timeline -> one entry (module docstring)."""
    fam = p["family"]
    spec = vocabulary["families"][fam]
    axis = tl.split(".", 1)[1] if "." in tl else None
    fp, fc = _vocabulary.tau_frames(p["t"], hz), _vocabulary.tau_frames(c["t"], hz)
    wp, wc = fp[3] - fp[0], fc[3] - fc[0]
    ap, ac = _decoder._state_value(p, tl, fitted, vocabulary), _decoder._state_value(c, tl, fitted, vocabulary)
    v = _vocabulary._round_step((wp * ap + wc * ac) / (wp + wc), spec["state_step"])
    levels = vocabulary["state"]["levels"]
    mag = int(np.searchsorted(np.asarray(_vocabulary.level_boundaries(tl, fitted), dtype=np.float64), abs(v), side="right"))
    e = {"family": fam, "layer": "state", "t": [p["t"][0], p["t"][1], c["t"][2], c["t"][3]],
         "level": levels[min(max(mag, 1), len(levels)) - 1]}
    if "value" in p and "value" in c:
        e["value"] = v
    if "kind" in p:
        e["kind"] = c["kind"] if wc > wp else p["kind"]
    if "dir" in p:
        if v == 0:
            e["dir"] = p["dir"]
        else:
            e["dir"] = (spec["dir"][axis] if axis else spec["dir"])["+" if v > 0 else "-"]
    if _vocabulary._pair_of(fam, e, vocabulary):
        sides = [(w, x["asym"]) for w, x in ((wp, p), (wc, c)) if "asym" in x]
        if sides:
            e["asym"] = _vocabulary._round_step(sum(w * a for w, a in sides) / sum(w for w, _ in sides),
                                      vocabulary["side"]["asym_step"])
            e["side"] = _vocabulary._side(e["asym"], vocabulary)
    return {k: e[k] for k in _vocabulary.STATE_KEYS if k in e}


def merge_family_states(entries: list, cap, hz: float, fitted: dict, vocabulary: dict = None) -> list:
    """One family's state entries -> at most cap entries (module docstring). Returns [(position, entry)] where
    position = the index in entries of the merged group's first member (for the encoded order)."""
    vocabulary = vocabulary or _vocabulary.VOCABULARY
    tl_order = [tl for tl, _, _ in _vocabulary.timelines(vocabulary)]
    by_tl = {}
    for i, e in enumerate(entries):
        by_tl.setdefault(_vocabulary.entry_timeline(e, vocabulary), []).append((_vocabulary.tau_frames(e["t"], hz)[0], i, e))
    for tl in by_tl:
        by_tl[tl].sort(key=lambda x: (x[0], x[1]))
    count = len(entries)
    while count > cap:
        pairs, singles = [], []
        for tl, items in by_tl.items():
            amps = [_written(_decoder._state_value(e, tl, fitted, vocabulary)) for _, _, e in items]
            for j in range(len(items) - 1):
                pairs.append((abs(amps[j] - amps[j + 1]), items[j][0], tl_order.index(tl), tl, j))
            for j, (start, _, _) in enumerate(items):
                singles.append((abs(amps[j]), start, tl_order.index(tl), tl, j))
        if pairs:
            _, _, _, tl, j = min(pairs)
            items = by_tl[tl]
            (s, i, p), (_, _, c) = items[j], items[j + 1]
            items[j:j + 2] = [(s, i, _merge_pair(p, c, tl, hz, fitted, vocabulary))]
        else:
            _, _, _, tl, j = min(singles)
            del by_tl[tl][j]
        count -= 1
    return sorted((i, e) for items in by_tl.values() for _, i, e in items)


def state_cap_entries(cap: int, n_frames: int, hz: float, state_cap: str = "per_track") -> int:
    """The entries a family keeps: C (per_track), or per_minute_ceil_min1: ceil(C * n_frames / (60 hz)), at least 1
    when C > 0."""
    _check_mode(state_cap, STATE_CAPS, "state_cap")
    if state_cap == "per_track" or cap == 0:
        return cap
    w = int(round(60.0 * hz))
    return max(1, -(-cap * n_frames // w))


def merge_states(states: list, c_per_family, hz: float, fitted: dict, vocabulary: dict = None,
                 state_cap: str = "per_track", n_frames: int = None) -> list:
    """Every capped family through merge_family_states; the result in the encoded order (an entry at the place of its
    first member). New list. state_cap="per_minute_ceil_min1" needs n_frames."""
    vocabulary = vocabulary or _vocabulary.VOCABULARY
    _check_mode(state_cap, STATE_CAPS, "state_cap")
    if state_cap != "per_track" and n_frames is None:
        raise ValueError("state_cap=%r needs n_frames" % state_cap)
    if c_per_family is None:
        return list(states)
    caps = c_per_family if isinstance(c_per_family, dict) else {f: c_per_family for f in vocabulary["families"]}
    bad = sorted(set(caps) - set(vocabulary["families"]))
    if bad:
        raise ValueError("c_per_family: unknown families %s" % bad)
    for fam, cap in caps.items():
        _check_cap(cap, "c_per_family[%s]" % fam)
    placed = []
    for fam in vocabulary["families"]:
        idx = [i for i, e in enumerate(states) if e["family"] == fam]
        cap = caps.get(fam)
        if _no_cap(cap):
            placed += [(i, states[i]) for i in idx]
            continue
        cap = state_cap_entries(cap, n_frames, hz, state_cap)
        placed += [(idx[j], e) for j, e in merge_family_states([states[i] for i in idx], cap, hz, fitted, vocabulary)]
    return [e for _, e in sorted(placed, key=lambda x: x[0])]


def strip_entry(e: dict) -> dict:
    """A state / event entry without value / asym / side / repeat / period (lite: levels only)."""
    return {k: v for k, v in e.items() if k not in LITE_STRIP}


def sparsify(track: dict, k_per_min, c_per_family, tier: str, fitted: dict = None, rest_stat: str = None,
             amplitude: str = "value", vocabulary: dict = None, quota: str = "per_minute_floor",
             state_cap: str = "per_track", quota_u: float = None) -> dict:
    """Full notes (or notes of a denser tier) -> standard or lite notes (Supp. A "densities").  k_per_min:
    events kept per minute (None / inf: all); c_per_family: state segments per family and minute; fitted: the
    calibrated constants; amplitude / quota / state_cap: the selection modes (the paper uses "dev_high",
    "per_track_stochastic", "per_minute_ceil_min1").  The result is validated."""
    vocabulary = vocabulary or _vocabulary.VOCABULARY
    _check_mode(tier, TIERS, "tier")
    _check_mode(quota, QUOTAS, "quota")
    _check_mode(state_cap, STATE_CAPS, "state_cap")
    out = _notes.strip_text(track)
    _notes.validate_track(out, vocabulary)
    fitted = fitted if fitted is not None else _vocabulary.calibration_for(rest_stat)
    hz, n = float(out["hz"]), int(out["n_frames"])
    if quota == "per_track_stochastic" and quota_u is None:
        quota_u = track_u(out.get("track_id"))
    out["event"] = select_events(out["event"], n, hz, k_per_min, fitted, amplitude, vocabulary, quota, quota_u)
    out["state"] = merge_states(out["state"], c_per_family, hz, fitted, vocabulary, state_cap, n)
    if tier == "lite":
        out["state"] = [strip_entry(e) for e in out["state"]]
        out["event"] = [strip_entry(e) for e in out["event"]]
        out["tendency"] = _tendency.strip_values(out["tendency"])
    out = json.loads(json.dumps(out))
    _notes.validate_track(out, vocabulary)
    return out


SPARSIFY_VERSION = "1.1.0"
