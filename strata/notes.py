"""Director notes N = (h, T, S, E) (paper Eq. (1); Supp. A "full grammar"): validation of what a set of notes
may contain.  A state or event is {"family", "layer", "t": [tau0, tau1, tau2, tau3] (seconds), "level" (slight /
clear = moderate / strong), "value", "kind", "dir", "side", "asym", "repeat", "period"}.
"""
import json
import numpy as np
from strata import tendency as _tendency
from strata import vocabulary as _vocabulary


def strip_text(track: dict) -> dict:
    """Only the fields of the director notes (header, tendency, state, event) and the clip ids: everything a
    decoder may read."""
    return json.loads(json.dumps({k: v for k, v in track.items() if k in _vocabulary.META_KEYS + _vocabulary.TEXT_KEYS}))


def _num(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and np.isfinite(x)


def validate_track(track: dict, vocabulary: dict = None) -> None:
    """Raise ValueError unless `track` is well-formed director notes (Sec. 3.2; Supp. A "full grammar" and
    "rules"): keys, header, entries (family, layer, four time points on the frame grid and ordered, level, kind /
    direction / side / asymmetry / repeat of the family), no overlapping states on one timeline (except a shared
    transition) and no overlapping events within a family."""
    vocabulary = vocabulary or _vocabulary.VOCABULARY
    bad = sorted(set(track) - set(_vocabulary.META_KEYS + _vocabulary.TEXT_KEYS))
    if bad:
        raise ValueError("director notes: keys %s are not note fields" % bad)
    for k in _vocabulary.TEXT_KEYS:
        if k not in track:
            raise ValueError("director notes have no %r" % k)
    hz, n = track["hz"], track["n_frames"]
    if not (_num(hz) and hz > 0) or isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError("header hz / n_frames invalid: %r / %r" % (hz, n))
    rest, ref = track["rest"], track["reference"]
    if not isinstance(rest, dict) or not isinstance(ref, dict) or set(ref) != {_vocabulary.ROTATION_TEXT} \
            or set(ref[_vocabulary.ROTATION_TEXT]) != set(_vocabulary.ROTATION_AXES) or not all(_num(v) for v in ref[_vocabulary.ROTATION_TEXT].values()):
        raise ValueError("header rest / reference invalid")
    if track["tendency"] is not None:
        _tendency.validate_tendency(track["tendency"], n, hz)
    for layer, keys, levels in (("state", _vocabulary.STATE_KEYS, vocabulary["state"]["levels"]),
                                ("event", _vocabulary.EVENT_KEYS, vocabulary["event"]["levels"])):
        if not isinstance(track[layer], list):
            raise ValueError("%s must be a list" % layer)
        for e in track[layer]:
            where = "%s entry %s" % (layer, e.get("t"))
            if not isinstance(e, dict) or set(e) - set(keys):
                raise ValueError("%s: unknown keys %s" % (where, sorted(set(e) - set(keys))))
            fam = e.get("family")
            if fam not in vocabulary["families"] or e.get("layer") != layer or e.get("level") not in levels:
                raise ValueError("%s: family / layer / level invalid" % where)
            spec = vocabulary["families"][fam]
            if layer == "event" and "event" not in spec:
                raise ValueError("%s: family %s has no event layer" % (where, fam))
            t = e.get("t")
            if not (isinstance(t, list) and len(t) == 4 and all(_num(x) for x in t)):
                raise ValueError("%s: t must be 4 numbers" % where)
            fr = _vocabulary.tau_frames(t, hz)
            if not (0 <= fr[0] <= fr[1] <= fr[2] <= fr[3] <= n) or fr[0] == fr[3] or (layer == "event" and fr[1] == fr[2]):
                raise ValueError("%s: t not ordered inside [0, n_frames] (event apex needs >= 1 frame)" % where)
            _check_fields(e, spec, layer, where, vocabulary)
            if layer == "event" and "repeat" in e and _vocabulary.event_units(e, hz)[-1][3] > n:
                raise ValueError("%s: repeated units run past n_frames" % where)
    _check_overlaps(track, vocabulary)


def _check_fields(e: dict, spec: dict, layer: str, where: str, vocabulary: dict) -> None:
    t = spec["type"]
    v = e.get("value")
    vector_value = layer == "event" and t == "vector" and spec["event"].get("value_vector")
    if v is not None and not (_num(v) or (vector_value and isinstance(v, list) and len(v) == len(spec["axes"])
                                          and all(_num(x) for x in v))):
        raise ValueError("%s: value invalid" % where)
    if t == "vector":
        if e.get("dir") not in [d for ax in spec["axes"] for d in spec["dir"][ax].values()]:
            raise ValueError("%s: dir %r invalid for %s" % (where, e.get("dir"), e["family"]))
        if layer == "event" and "kind" in spec["event"]:
            axis = _vocabulary.entry_timeline(e, vocabulary).split(".", 1)[1]
            if e.get("kind") != spec["event"]["kind"][axis]:
                raise ValueError("%s: kind %r does not match dir %r" % (where, e.get("kind"), e.get("dir")))
        elif "kind" in e:
            raise ValueError("%s: %s takes no kind" % (where, e["family"]))
    elif t == "bs_diff":
        if layer == "state" and (e.get("dir") not in spec["dir"].values() or "kind" in e):
            raise ValueError("%s: an eyelid state needs a dir of %s and no kind" % (where, list(spec["dir"].values())))
        if layer == "event" and (e.get("kind") != spec["event"]["kind"] or "dir" in e):
            raise ValueError("%s: an eyelid event is kind %r with no dir" % (where, spec["event"]["kind"]))
    elif t == "bs_tension":
        if e.get("kind") not in spec["kinds"] or "dir" in e:
            raise ValueError("%s: mouth_tension needs a kind of %s and no dir" % (where, list(spec["kinds"])))
    elif "kind" in e or "dir" in e:
        raise ValueError("%s: %s takes no kind / dir" % (where, e["family"]))
    if "side" in e or "asym" in e:
        if not _vocabulary._pair_of(e["family"], e, vocabulary):
            raise ValueError("%s: side / asym on a family / kind without a channel pair" % where)
        if "side" in e and e["side"] not in _vocabulary.SIDES:
            raise ValueError("%s: side invalid" % where)
        if "asym" in e and not (_num(e["asym"]) and -1.0 <= e["asym"] <= 1.0):
            raise ValueError("%s: asym must be in [-1, 1]" % where)
    if ("repeat" in e) != ("period" in e):
        raise ValueError("%s: repeat and period go together" % where)
    if "repeat" in e:
        if not spec.get("event", {}).get("repeat"):
            raise ValueError("%s: %s events do not repeat" % (where, e["family"]))
        if isinstance(e["repeat"], bool) or not isinstance(e["repeat"], int) or e["repeat"] < 2 \
                or not (_num(e["period"]) and e["period"] > 0):
            raise ValueError("%s: repeat must be an integer >= 2 with period > 0" % where)


def _check_overlaps(track: dict, vocabulary: dict) -> None:
    hz = track["hz"]
    by_tl = {}
    for e in track["state"]:
        by_tl.setdefault(_vocabulary.entry_timeline(e, vocabulary), []).append(_vocabulary.tau_frames(e["t"], hz))
    for tl, spans in by_tl.items():
        spans.sort()
        for p, c in zip(spans, spans[1:]):
            if not (p[3] <= c[0] or (p[2] == c[0] and p[3] == c[1])):
                raise ValueError("state timeline %s: entries %s and %s overlap" % (tl, p, c))
    by_fam = {}
    for e in track["event"]:
        by_fam.setdefault(e["family"], []).extend(_vocabulary.event_units(e, hz))
    for fam, units in by_fam.items():
        units.sort()
        for p, c in zip(units, units[1:]):
            if p[3] > c[0]:
                raise ValueError("events of %s overlap: %s and %s" % (fam, p, c))
