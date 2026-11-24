"""The vocabulary of the director language (paper Sec. 3.2; Supp. A).

Nine motion families f with their channels C(f) (eyelid, squint, brow raise, brow lower = frown, smile, mouth
tension, head rotation, head translation, gaze), 14 state timelines, the 14 speech channels (driven by the script)
and the unused channels.  The hand-set constants live in assets/l2_v2_vocabulary.json, the constants calibrated
on the training set in assets/l2_v2_fitted.json (Supp. B "properties and constants").
"""
import hashlib
import json
import numpy as np
import os
from strata import representation as _representation


SCHEMAS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")


VOCABULARY_PATH = os.path.join(SCHEMAS, "l2_v2_vocabulary.json")


FITTED_PATH = os.path.join(SCHEMAS, "l2_v2_fitted.json")


ROTATION_TEXT = _representation.CHANNEL_RULES["channels"][_representation.ROTATION_CHANNEL]["text_name"]


ROTATION_AXES = tuple(_representation.CHANNEL_RULES["channels"][_representation.ROTATION_CHANNEL]["axes"])


TRANSLATION_TEXT = tuple(_representation.CHANNEL_RULES["channels"]["headpose_translation"]["text_names"])


GAZE_TEXT = tuple(_representation.CHANNEL_RULES["channels"]["gaze"]["text_names"])


EXCLUDED = tuple(_representation.CHANNEL_RULES["channels"]["blendshape"]["exclude"])


META_KEYS = ("track_id", "clip_id", "segment_id", "track_start", "time_sec")


TEXT_KEYS = ("hz", "n_frames", "rest", "reference", "tendency", "state", "event")


STATE_KEYS = ("family", "layer", "t", "level", "value", "kind", "dir", "side", "asym")


EVENT_KEYS = ("family", "layer", "t", "level", "value", "kind", "dir", "side", "asym", "repeat", "period")


SIDES = ("both", "left", "right")


def _sha256_file(path: str) -> str:
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def _load(path: str) -> dict:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


VOCABULARY = _load(VOCABULARY_PATH)


VOCABULARY_SHA256 = _sha256_file(VOCABULARY_PATH)


def load_calibration(path: str = FITTED_PATH) -> dict:
    """The calibrated constants (Supp. B "properties and constants"; fitted once on the 480 TalkVid training
    clips), refused unless they were calibrated under this vocabulary file."""
    fitted = _load(path)
    if fitted.get("vocabulary_sha256") != VOCABULARY_SHA256:
        raise ValueError("%s was fitted under vocabulary sha %s, the loaded vocabulary is %s: refit (scripts/l2_v2.py fit)"
                         % (path, fitted.get("vocabulary_sha256"), VOCABULARY_SHA256))
    return fitted


def calibration_for(rest_stat: str = None, fitted: dict = None) -> dict:
    """The calibrated constants of one neutral-value variant (default: the vocabulary's, 10th percentile for
    one-sided families)."""
    rest_stat = rest_stat or VOCABULARY["rest"]["one_sided_stat"]
    fitted = fitted if fitted is not None else load_calibration()
    if rest_stat not in fitted.get("rest_variants", {}):
        raise ValueError("no fitted constants for rest_stat %r" % rest_stat)
    return fitted["rest_variants"][rest_stat]


def signal_names(blendshape_names: list) -> list:
    return ([n for n in blendshape_names if n not in EXCLUDED] + ["%s.%s" % (ROTATION_TEXT, a) for a in ROTATION_AXES]
            + list(TRANSLATION_TEXT) + list(GAZE_TEXT))


def timelines(vocabulary: dict = None) -> list:
    """(timeline, family f, axis or None): one per scalar family, one per axis of a vector family (14 in total;
    Supp. B "states")."""
    out = []
    for fam, spec in (vocabulary or VOCABULARY)["families"].items():
        if spec["type"] == "vector":
            out += [("%s.%s" % (fam, a), fam, a) for a in spec["axes"]]
        else:
            out.append((fam, fam, None))
    return out


def _axis_signal(spec: dict, axis: str) -> str:
    if spec["group"] == "rotation":
        return "%s.%s" % (ROTATION_TEXT, axis)
    names = TRANSLATION_TEXT if spec["group"] == "translation" else GAZE_TEXT
    return names[spec["axes"].index(axis)]


def channels_of(fam: str, vocabulary: dict = None) -> list:
    """C(f): the channels of family f (Sec. 3.2; Supp. A, motion-family table).  Head rotation: its 3 rotation-
    vector axes; gaze: pitch, yaw and the eight eyeLook* blendshapes."""
    spec = (vocabulary or VOCABULARY)["families"][fam]
    if spec["type"] == "vector":
        return [_axis_signal(spec, a) for a in spec["axes"]] + list(spec.get("members", []))
    return list(spec["channels"])


def signal_columns(blendshape_names: list) -> dict:
    """channel name -> its column(s) in the 63-D frame x_t (head rotation axes -> the 6 rotation columns)."""
    bs, rot = _representation.channel("blendshape"), _representation.channel(_representation.ROTATION_CHANNEL)
    out = {n: [bs.start + k] for k, n in enumerate(blendshape_names) if n not in EXCLUDED}
    out.update({"%s.%s" % (ROTATION_TEXT, a): list(range(rot.start, rot.end)) for a in ROTATION_AXES})
    out.update({n: [_representation.channel("headpose_translation").start + k] for k, n in enumerate(TRANSLATION_TEXT)})
    out.update({n: [_representation.channel("gaze").start + k] for k, n in enumerate(GAZE_TEXT)})
    return out


def rest_dict(rest: np.ndarray, names: list) -> dict:
    """The neutral values (rest) of the clip header as stored in the notes."""
    out = {}
    for name, v in zip(names, rest.tolist()):
        if name.startswith(ROTATION_TEXT + "."):
            out.setdefault(ROTATION_TEXT, {})[name.split(".", 1)[1]] = v
        else:
            out[name] = v
    return out


def rest_from_dict(rest: dict, names: list) -> np.ndarray:
    return np.array([float(rest[ROTATION_TEXT][n.split(".", 1)[1]]) if n.startswith(ROTATION_TEXT + ".") else float(rest[n])
                     for n in names])


def _round_step(x: float, step: float) -> float:
    """x to the nearest multiple of step (4 decimals, never -0.0): how values are written into the notes."""
    q = round(float(np.sign(x) * np.floor(abs(x) / step + 0.5) * step), 4)
    return q + 0.0 if q else 0.0


def t_of_frame(frame: int, hz: float) -> float:
    return _representation.frame_to_time(int(frame), hz)


def tau_frames(t: list, hz: float) -> list:
    return [_representation.time_to_frame(x, hz) for x in t]


def _side(asym: float, vocabulary: dict = None) -> str:
    th = (vocabulary or VOCABULARY)["side"]["both_below"]
    return "both" if abs(asym) < th else ("left" if asym > 0 else "right")


def _asym(d_left: float, d_right: float, vocabulary: dict = None) -> float:
    s = d_left + d_right
    if s <= 1e-6:
        return 0.0
    return _round_step(float(np.clip((d_left - d_right) / s, -1.0, 1.0)), (vocabulary or VOCABULARY)["side"]["asym_step"])


def _entry_asym(entry: dict, vocabulary: dict = None) -> float:
    if "asym" in entry:
        return float(entry["asym"])
    a = (vocabulary or VOCABULARY)["side"]["side_only_asym"]
    return {"both": 0.0, "left": a, "right": -a}.get(entry.get("side", "both"), 0.0)


def _pair_of(fam: str, entry: dict, vocabulary: dict = None):
    spec = (vocabulary or VOCABULARY)["families"][fam]
    if spec["type"] == "bs_tension":
        return spec["kinds"][entry["kind"]].get("pair") if "kind" in entry else None
    return spec.get("pair")


def level_boundaries(tl: str, fitted: dict) -> list:
    return fitted["state_edges"][tl]


def entry_timeline(e: dict, vocabulary: dict = None) -> str:
    """The timeline of a state or event (a vector family's axis comes from its direction)."""
    vocabulary = vocabulary or VOCABULARY
    spec = vocabulary["families"][e["family"]]
    if spec["type"] != "vector":
        return e["family"]
    for axis in spec["axes"]:
        if e.get("dir") in spec["dir"][axis].values():
            return "%s.%s" % (e["family"], axis)
    raise ValueError("%s entry %s: dir %r names no axis" % (e["family"], e.get("t"), e.get("dir")))


def _entry_sign(e: dict, spec: dict, axis: str) -> float:
    if spec["type"] == "vector":
        return 1.0 if spec["dir"][axis]["+"] == e["dir"] else -1.0
    if spec["type"] == "bs_diff":
        return 1.0 if spec["dir"]["+"] == e["dir"] else -1.0
    return 1.0


def event_level_key(fam: str, core: dict) -> str:
    """The key of the event-amplitude tertiles: mouth tension per kind, head rotation per axis, else the
    family (Supp. B "events")."""
    if fam == "mouth_tension":
        return "mouth_tension.%s" % core["kind"]
    if fam == "head_rot":
        return "head_rot.%s" % core["axis"]
    return fam


def event_units(e: dict, hz: float) -> list:
    """[(tau0, tau1, tau2, tau3)] in frames for each unit of an event; a repeated event (repeat, period;
    Supp. A "rules") is the first unit shifted by k * period."""
    base = tau_frames(e["t"], hz)
    p = _representation.time_to_frame(e["period"], hz) if "repeat" in e else 0
    return [tuple(x + k * p for x in base) for k in range(int(e.get("repeat", 1)))]


def delta_c_row(blendshape_names: list, vocabulary: dict = None) -> np.ndarray:
    """(63,) projection tolerance delta_c per column (Eq. (6); Supp. B, encoder-constants table); NaN for
    columns without an event family.  Head-rotation columns hold the per-axis tolerance in degrees."""
    vocabulary = vocabulary or VOCABULARY
    cols = signal_columns(blendshape_names)
    row = np.full(_representation.TOTAL_DIM, np.nan)
    for fam, spec in vocabulary["families"].items():
        if "event" not in spec:
            continue
        ev = spec["event"]
        if spec["type"] == "vector":
            for a in spec["axes"]:
                row[cols[_axis_signal(spec, a)]] = ev["delta"]
            for m in spec.get("members", []):
                row[cols[m]] = ev["member_delta"]
        elif spec["type"] == "bs_tension":
            for k in spec["kinds"].values():
                for c in k["channels"]:
                    row[cols[c]] = k["delta"]
        else:
            for c in spec["channels"]:
                row[cols[c]] = ev["delta"] if c in ev["drivers"] else vocabulary["delta"]["non_driver_blendshape"]
    return row


def family_columns(fam: str, blendshape_names: list, vocabulary: dict = None) -> list:
    cols = signal_columns(blendshape_names)
    return sorted({c for s in channels_of(fam, vocabulary) for c in cols[s]})


VOCABULARY_VERSION = VOCABULARY["vocabulary_version"]


FAMILIES = tuple(VOCABULARY["families"])
