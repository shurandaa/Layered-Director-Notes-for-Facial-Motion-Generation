"""Tendencies (paper Sec. 3.2; Supp. A "tendencies", Supp. B "encoder: tendencies"): seven statistics per clip
and per sentence segment (eyeblink frequency, head energy, head motion frequency, gaze wandering, gaze shifting
frequency, upper face energy, mouth amplitude), five levels by the training-set quintiles.
"""
import numpy as np
import os
from scipy.ndimage import gaussian_filter1d
from strata import encoder as _encoder
from strata import notes as _notes
from strata import vocabulary as _vocabulary


RULES_PATH = os.path.join(_vocabulary.SCHEMAS, "l2_v2_tendency.json")


QUANTILES_PATH = os.path.join(_vocabulary.SCHEMAS, "l2_v2_tendency_quantiles.json")


RULES = _vocabulary._load(RULES_PATH)


RULES_SHA256 = _vocabulary._sha256_file(RULES_PATH)


PARAMS = tuple(RULES["params"])


LEVELS = tuple(RULES["levels"])


BLOCK_KEYS = ("default", "segments")


PARAM_KEYS = ("level", "value")


def load_quantiles(path: str = QUANTILES_PATH) -> dict:
    """The tendency quantile file (calibrated on the training set), refused unless it was fitted under these
    rules, this vocabulary and these constants."""
    q = _vocabulary._load(path)
    want = {"tendency_rules_sha256": RULES_SHA256, "vocabulary_sha256": _vocabulary.VOCABULARY_SHA256,
            "fitted_sha256": _vocabulary._sha256_file(_vocabulary.FITTED_PATH)}
    bad = {k: (q.get(k), v) for k, v in want.items() if q.get(k) != v}
    if bad:
        raise ValueError("%s does not match the loaded files %s: refit (scripts/l2_v2.py fit-tendency)" % (path, bad))
    return q


def quantiles_for(rest_stat: str = None, quantiles: dict = None) -> dict:
    """{"edges", "representatives", "articulation_sigma"} of one rest variant (the event rates depend on it)."""
    rest_stat = rest_stat or _vocabulary.VOCABULARY["rest"]["one_sided_stat"]
    quantiles = quantiles if quantiles is not None else load_quantiles()
    if rest_stat not in quantiles.get("rest_variants", {}):
        raise ValueError("no tendency quantiles for rest_stat %r" % rest_stat)
    return dict(quantiles["rest_variants"][rest_stat], articulation_sigma=quantiles["articulation_sigma"])


def segments_from_sentences(sentences: list, n_frames: int, rules: dict = None) -> list:
    """[(start_frame, end_frame inclusive)] per sentence, in order -> [(a, b)] half-open segments covering [0, n):
    boundary between k and k+1 = floor((end_k + 1 + start_{k+1}) / 2), then the merge rule (rules segmentation)."""
    min_frames = int((rules or RULES)["segmentation"]["min_frames"])
    sentences = [(int(s), int(e)) for s, e in sentences]
    for (s0, e0), (s1, e1) in zip(sentences, sentences[1:]):
        if not (s0 <= e0 < s1 <= e1):
            raise ValueError("sentences overlap or are out of order: %s %s" % ((s0, e0), (s1, e1)))
    cuts = [(e0 + 1 + s1) // 2 for (_, e0), (s1, _) in zip(sentences, sentences[1:])]
    bounds = [0] + [c for c in cuts if 0 < c < n_frames] + [n_frames]
    segs = [[a, b] for a, b in zip(bounds, bounds[1:]) if b > a]
    while len(segs) > 1:
        lens = [b - a for a, b in segs]
        k = int(np.argmin(lens))
        if lens[k] >= min_frames:
            break
        if k == 0:
            j = 1
        elif k == len(segs) - 1:
            j = k - 1
        else:
            j = k - 1 if lens[k - 1] < lens[k + 1] else k + 1
        a, b = min(segs[k][0], segs[j][0]), max(segs[k][1], segs[j][1])
        lo = min(k, j)
        segs[lo:lo + 2] = [[a, b]]
    return [tuple(s) for s in segs]


def in_word_mask(words: list, n_frames: int) -> np.ndarray:
    """(T,) bool: frames inside a word, words = [(start_frame, end_frame inclusive)] (textfeat in_word)."""
    m = np.zeros(n_frames, dtype=bool)
    for s, e in words:
        m[int(s):int(e) + 1] = True
    return m


def _feature_series(x: np.ndarray, names: list, rules: dict) -> np.ndarray:
    """(T, 4) articulation features: each the mean of its blendshape signals."""
    ix = {nm: k for k, nm in enumerate(names)}
    return np.stack([x[:, [ix[c] for c in f]].mean(axis=1) for f in rules["articulation"]["features"]], axis=1)


def feature_keys(rules: dict = None) -> list:
    return ["+".join(f) for f in (rules or RULES)["articulation"]["features"]]


def frame_series(track: dict, blendshape_names: list, gaze_rest: np.ndarray, rules: dict = None) -> dict:
    """Per-frame series over the whole track: head speed (deg/s), face speed (1/s), gaze away (bool), articulation
    features (T, 4). Smoothing and gradients run over the whole track, so a segment boundary adds no edge effect."""
    rules = rules or RULES
    names = _vocabulary.signal_names(blendshape_names)
    ix = {nm: k for k, nm in enumerate(names)}
    y, _, _, _ = _encoder.track_signals(track, blendshape_names)
    hz = float(_vocabulary.VOCABULARY["hz"])
    sig = float(rules["smooth_sigma_frames"])
    rot = y[:, [ix["%s.%s" % (_vocabulary.ROTATION_TEXT, a)] for a in _vocabulary.ROTATION_AXES]]
    grad = lambda a: np.gradient(gaussian_filter1d(a, sig, axis=0, mode="nearest"), axis=0) * hz
    head = np.linalg.norm(grad(rot), axis=1) if len(y) > 1 else np.zeros(len(y))
    face_x = y[:, [ix[c] for c in rules["face_energy"]["channels"]]]
    face = np.sqrt(np.mean(grad(face_x) ** 2, axis=1)) if len(y) > 1 else np.zeros(len(y))
    gz = y[:, [ix[g] for g in _vocabulary.GAZE_TEXT]]
    away = np.linalg.norm(gz - gaze_rest, axis=1) > float(rules["gaze_wander"]["threshold"])
    return {"head": head, "face": face, "away": away, "artic": _feature_series(y, names, rules)}


def _unit_starts(events: list, hz: float, rules: dict) -> dict:
    """rate param -> sorted start frames of the counted event units."""
    out = {}
    for p, spec in rules["rates"].items():
        if p == "rule":
            continue
        starts = []
        for e in events:
            if e["family"] == spec["family"] and ("kind" not in spec or e.get("kind") == spec["kind"]):
                starts += [u[0] for u in _vocabulary.event_units(e, hz)]
        out[p] = np.sort(np.asarray(starts, dtype=int))
    return out


def _values(series: dict, starts: dict, in_word, a: int, b: int, hz: float, sigma: dict, rules: dict) -> dict:
    """The 7 values of frames [a, b), rounded to their steps; None = undefined."""
    minutes = (b - a) / hz / 60.0
    v = {p: float(np.sum((starts[p] >= a) & (starts[p] < b))) / minutes for p in starts}
    v["head_energy"] = float(series["head"][a:b].mean())
    v["face_energy"] = float(series["face"][a:b].mean())
    v["gaze_wander"] = float(series["away"][a:b].mean())
    v["articulation"] = None
    if in_word is not None:
        sel = np.flatnonzero(in_word[a:b]) + a
        if sel.size >= int(rules["articulation"]["min_frames"]):
            xf = series["artic"][sel]
            spread = np.percentile(xf, 90, axis=0) - np.percentile(xf, 10, axis=0)
            sd = np.array([float(sigma[k]) for k in feature_keys(rules)])
            v["articulation"] = float(np.mean(spread / sd))
    return {p: (None if v[p] is None else _vocabulary._round_step(v[p], rules["steps"][p])) for p in PARAMS}


def tendency_values(track: dict, blendshape_names: list, encoded: dict, segments: list, in_word=None,
                    sigma: dict = None, rules: dict = None) -> dict:
    """The seven tendency statistics (Supp. A "tendencies") over the whole clip and over each sentence
    segment.  track: the tracked clip; encoded: its notes (events, rest, hz); segments: merged sentence segments
    [a, b); in_word: per-frame word mask (mouth amplitude uses only frames inside words; None = undefined); sigma: the
    training-set articulation standard deviations."""
    rules = rules or RULES
    n, hz = int(encoded["n_frames"]), float(encoded["hz"])
    if sigma is None:
        sigma = load_quantiles()["articulation_sigma"]
    names = _vocabulary.signal_names(blendshape_names)
    rest = _vocabulary.rest_from_dict(encoded["rest"], names)
    gaze_rest = rest[[names.index(g) for g in _vocabulary.GAZE_TEXT]]
    series = frame_series(track, blendshape_names, gaze_rest, rules)
    if len(series["head"]) != n:
        raise ValueError("track has %d frames, the notes %d" % (len(series["head"]), n))
    if in_word is not None:
        in_word = np.asarray(in_word, dtype=bool)
        if in_word.shape != (n,):
            raise ValueError("in_word must have shape (%d,)" % n)
    starts = _unit_starts(encoded["event"], hz, rules)
    if not segments or segments[0][0] != 0 or segments[-1][1] != n or any(
            s[1] != t[0] or s[1] <= s[0] for s, t in zip(segments, segments[1:])) or segments[-1][1] <= segments[-1][0]:
        raise ValueError("segments must tile [0, %d): %s" % (n, segments))
    return {"default": _values(series, starts, in_word, 0, n, hz, sigma, rules),
            "segments": [{"frames": (int(a), int(b)), "values": _values(series, starts, in_word, a, b, hz, sigma, rules)}
                         for a, b in segments]}


def level_index(value: float, edges: list) -> int:
    """The number of edges strictly below value (ties at an edge go down)."""
    return int(sum(1 for e in edges if e < value))


def _entry(value, q: dict, p: str) -> dict:
    if value is None:
        return {"level": "mid"}
    return {"level": LEVELS[level_index(value, q["edges"][p])], "value": value}


def encode_tendency(values: dict, hz: float, q: dict) -> dict:
    """tendency_values -> the text block: default holds all 7 (undefined -> {"level": "mid"}); each segment entry
    {"t": [start, end) seconds, params whose level differs from the default's}; segments with none are not written."""
    default = {p: _entry(values["default"][p], q, p) for p in PARAMS}
    segs = []
    for s in values["segments"]:
        a, b = s["frames"]
        e = {"t": [_vocabulary.t_of_frame(a, hz), _vocabulary.t_of_frame(b, hz)]}
        for p in PARAMS:
            v = s["values"][p]
            if v is not None and _entry(v, q, p)["level"] != default[p]["level"]:
                e[p] = _entry(v, q, p)
        if len(e) > 1:
            segs.append(e)
    return {"default": default, "segments": segs}


def parse_param(x, p: str = None, q: dict = None) -> tuple:
    """A written tendency entry -> (level index, value or None): a level word, a number (its level by the
    training-set quintiles), or {"level", "value"}."""
    if isinstance(x, str):
        if x not in LEVELS:
            raise ValueError("tendency level %r is not one of %s" % (x, LEVELS))
        return LEVELS.index(x), None
    if _notes._num(x):
        x = {"value": x}
    if not isinstance(x, dict) or not x or set(x) - set(PARAM_KEYS):
        raise ValueError("tendency param %r: a level word, a number or {level, value}" % (x,))
    value = x.get("value")
    if value is not None and not (_notes._num(value) and value >= 0):
        raise ValueError("tendency value %r must be a finite number >= 0" % (value,))
    if "level" in x:
        if x["level"] not in LEVELS:
            raise ValueError("tendency level %r is not one of %s" % (x["level"], LEVELS))
        return LEVELS.index(x["level"]), (None if value is None else float(value))
    if value is None:
        raise ValueError("tendency param %r has neither level nor value" % (x,))
    if q is None or p is None:
        return None, float(value)
    return level_index(float(value), q["edges"][p]), float(value)


def validate_tendency(block, n_frames: int, hz: float) -> None:
    """Raise ValueError unless block is None or a well-formed tendency block: keys default / segments, params known
    and parseable, segment t = [start, end) on the hz grid inside [0, n_frames], ordered and not overlapping."""
    if block is None:
        return
    if not isinstance(block, dict) or set(block) - set(BLOCK_KEYS):
        raise ValueError("tendency must be null or an object with keys %s" % (BLOCK_KEYS,))
    default = block.get("default", {})
    if not isinstance(default, dict) or set(default) - set(PARAMS):
        raise ValueError("tendency default: unknown params %s" % sorted(set(default) - set(PARAMS)))
    for p, x in default.items():
        parse_param(x)
    segs = block.get("segments", [])
    if not isinstance(segs, list):
        raise ValueError("tendency segments must be a list")
    prev = 0
    for s in segs:
        if not isinstance(s, dict) or set(s) - set(PARAMS + ("t",)):
            raise ValueError("tendency segment: unknown keys %s" % sorted(set(s) - set(PARAMS + ("t",))))
        t = s.get("t")
        if not (isinstance(t, list) and len(t) == 2 and all(_notes._num(x) for x in t)):
            raise ValueError("tendency segment t must be [start, end] seconds")
        a, b = _vocabulary.tau_frames(t, hz)
        if not (prev <= a < b <= n_frames):
            raise ValueError("tendency segment t %s outside [0, n_frames], out of order or overlapping" % t)
        prev = b
        for p in PARAMS:
            if p in s:
                parse_param(s[p])


def _decode_value(x, p: str, q: dict) -> float:
    level, value = parse_param(x, p, q)
    return value if value is not None else float(q["representatives"][p][level])


def decode_tendency(block, n_frames: int, hz: float, q: dict = None, rest_stat: str = None, drop: bool = False) -> dict:
    """(T, 7) per-frame tendency values of a tendency block: the clip default, overridden by sentence
    segments.  A missing block (or drop) gives zeros and dropped = True: the empty-tendency condition."""
    validate_tendency(block, n_frames, hz)
    if block is None or drop:
        return {"values": np.zeros((n_frames, len(PARAMS))), "names": PARAMS, "dropped": True}
    q = q if q is not None else quantiles_for(rest_stat)
    default = block.get("default", {})
    out = np.empty((n_frames, len(PARAMS)))
    for k, p in enumerate(PARAMS):
        out[:, k] = _decode_value(default.get(p, "mid"), p, q)
    for s in block.get("segments", []):
        a, b = _vocabulary.tau_frames(s["t"], hz)
        for k, p in enumerate(PARAMS):
            if p in s:
                out[a:b, k] = _decode_value(s[p], p, q)
    return {"values": out, "names": PARAMS, "dropped": False}


def strip_values(block):
    """The tendency block with level words only (value dropout of the random masking, Sec. 3.4)."""
    if block is None:
        return None
    lv = lambda x: x["level"] if isinstance(x, dict) and "level" in x else x
    return {"default": {p: lv(x) for p, x in block.get("default", {}).items()},
            "segments": [{k: (v if k == "t" else lv(v)) for k, v in s.items()} for s in block.get("segments", [])]}


TENDENCY_VERSION = RULES["tendency_version"]
