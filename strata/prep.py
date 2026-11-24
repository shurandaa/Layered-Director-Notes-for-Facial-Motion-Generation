"""Training and validation data (paper Sec. 3.4 "training"; Supp. C "training data and masking").

From the tracked curves and the script of every clip: full notes by the encoder, then 8 randomly masked variants
per training clip (R(N): a density among full / standard / lite / tendencies only, how many entries to keep,
values dropped with p = 0.5, each layer dropped with p = 0.15, blinks dropped with p = 0.9), and the three
densities of each validation clip.  Every condition array comes from a set of notes through notes_conditions.
"""
import copy
import hashlib
import json
import numpy as np
import os
import zlib
from strata import conditions as _conditions
from strata import decoder as _decoder
from strata import evaluate as _evaluate
from strata import notes as _notes
from strata import representation as _representation
from strata import script as _script
from strata import sparsify as _sparsify
from strata import tendency as _tendency
from strata import vocabulary as _vocabulary


FORMAT = "gen_prep_v2"


SPLITS = ("train", "val")


DRAW = {"seed": 0, "n_variants_train": 8, "g7_mirror_variants": 4,
        "tier_p": {"full": 0.15, "standard": 0.30, "lite": 0.30, "tendency_only": 0.125, "tendency_partial": 0.125},
        "value_dropout_p": 0.5, "value_dropout_tiers": ["full", "standard", "tendency_only", "tendency_partial"],
        "param_keep_p": 0.5, "layer_drop_p": {"state": 0.15, "event": 0.15, "tendency": 0.15}}


TIERS = tuple(DRAW["tier_p"])


LAYERS = ("state", "event", "tendency")


G7_KEY = 7


STD_FLOOR = 1e-3


def codec_info(codec: dict) -> dict:
    return {"vocabulary_version": _vocabulary.VOCABULARY_VERSION, "vocabulary_sha256": _vocabulary.VOCABULARY_SHA256,
            "fitted_sha256": _vocabulary._sha256_file(_vocabulary.FITTED_PATH), "rest_stat": codec["rest_stat"],
            "tendency_version": _tendency.TENDENCY_VERSION, "tendency_rules_sha256": _tendency.RULES_SHA256,
            "tendency_quantiles_sha256": _vocabulary._sha256_file(_tendency.QUANTILES_PATH), "sparsify_version": _sparsify.SPARSIFY_VERSION,
            "viseme_version": _script.VERSION}


def variant_rng(track_id: str, r: int, seed: int = DRAW["seed"], g7: bool = False):
    return np.random.default_rng([seed, zlib.crc32(track_id.encode()), r] + ([G7_KEY] if g7 else []))


def strip_entry_values(text: dict) -> dict:
    """Value dropout: `value` removed from every state / event entry, the tendency block levels only (a new dict)."""
    out = copy.deepcopy(text)
    for layer in ("state", "event"):
        out[layer] = [{k: v for k, v in e.items() if k != "value"} for e in out[layer]]
    out["tendency"] = _tendency.strip_values(out["tendency"])
    return out


def keep_params(block: dict, keep: list) -> dict:
    """The tendency block with only the params in keep: dropped params leave the default and every segment; a segment
    left with only `t` is removed."""
    default = {p: x for p, x in block.get("default", {}).items() if p in keep}
    segs = [{k: v for k, v in s.items() if k == "t" or k in keep} for s in block.get("segments", [])]
    return {"default": default, "segments": [s for s in segs if len(s) > 1]}


def draw_quota_u(rng, params: dict, rec: dict):
    """The per_track_stochastic uniform of a TRAIN variant, drawn right after K / C (only in that mode, so the other
    modes' draws are unchanged) and recorded."""
    if params["quota"] != "per_track_stochastic":
        return None
    rec["quota_u"] = float(rng.random())
    return rec["quota_u"]


def draw_tier(rng, draw: dict = DRAW) -> str:
    """Step 1: one uniform against the cumulative tier_p (TIERS order)."""
    p = np.array([draw["tier_p"][t] for t in TIERS])
    return TIERS[int(np.searchsorted(np.cumsum(p) / p.sum(), rng.random(), side="right"))]


def draw_variant(full: dict, rng, params: dict, codec: dict, draw: dict = DRAW) -> tuple:
    """(variant text, draw record) of one TRAIN variant r >= 1 (module doc, steps 1-6, in this draw order)."""
    tier = draw_tier(rng, draw)
    rec = {"tier": tier}
    text = _notes.strip_text(full)
    if tier == "standard":
        rec["K"] = int(rng.integers(params["k_lite"], params["k_std"] + 1))
        rec["C"] = int(rng.integers(params["c_lite"], params["c_std"] + 1))
        text = _conditions.sparsify(text, rec["K"], rec["C"], "standard", params, codec, draw_quota_u(rng, params, rec))
    elif tier == "lite":
        rec["K"], rec["C"] = int(rng.integers(1, params["k_lite"] + 1)), params["c_lite"]
        text = _conditions.sparsify(text, rec["K"], rec["C"], "lite", params, codec, draw_quota_u(rng, params, rec))
    if tier in draw["value_dropout_tiers"]:
        rec["value_dropout"] = bool(rng.random() < draw["value_dropout_p"])
        if rec["value_dropout"]:
            text = strip_entry_values(text)
    if tier in ("tendency_only", "tendency_partial"):
        text["state"], text["event"] = [], []
    if tier == "tendency_partial" and text["tendency"] is not None:
        keep = []
        while not keep:
            keep = [q for q, u in zip(_tendency.PARAMS, rng.random(len(_tendency.PARAMS))) if u < draw["param_keep_p"]]
        rec["kept_params"] = keep
        text["tendency"] = keep_params(text["tendency"], keep)
    u = rng.random(len(LAYERS))
    rec["layer_drop"] = []
    for layer, x in zip(LAYERS, u):
        present = text[layer] is not None if layer == "tendency" else bool(text[layer])
        if present and x < draw["layer_drop_p"][layer]:
            text[layer] = None if layer == "tendency" else []
            rec["layer_drop"].append(layer)
    if draw.get("blink_drop_p"):
        text["event"], rec["blink_dropped"] = drop_blinks(text["event"], rng, draw["blink_drop_p"])
    text = json.loads(json.dumps(text))
    _notes.validate_track(text)
    return text, rec


def is_blink(e: dict) -> bool:
    return e["family"] == "eyelid" and e.get("kind") == "blink"


def drop_blinks(events: list, rng, p: float) -> tuple:
    """(events without the dropped blinks, number dropped): one uniform per blink entry, dropped when < p."""
    out, n = [], 0
    for e in events:
        if is_blink(e) and rng.random() < p:
            n += 1
        else:
            out.append(e)
    return out, n


def make_draw(blink_drop_p: float = 0.0) -> dict:
    """DRAW, + blink_drop_p only when > 0 (a p = 0 prep keeps the signed manifest's draw block)."""
    if not 0.0 <= blink_drop_p <= 1.0:
        raise ValueError("blink_drop_p must be in [0, 1]")
    return dict(DRAW, blink_drop_p=float(blink_drop_p)) if blink_drop_p > 0 else DRAW


def train_variants(full: dict, track_id: str, params: dict, codec: dict, n_variants: int, g7: bool = False,
                   draw: dict = DRAW) -> list:
    """[(text, record)]: v0 = the full text, then v1..v<n_variants> drawn with variant_rng(track_id, r)."""
    out = [(_notes.strip_text(full), {"tier": "full", "variant0": True})]
    for r in range(1, n_variants + 1):
        out.append(draw_variant(full, variant_rng(track_id, r, g7=g7), params, codec, draw))
    return out


def prior_header(train_texts: list, priors: dict, names: list, codec: dict) -> dict:
    """The TRAIN-prior header (a): rest = per-channel (per-axis for the rotation) median of the TRAIN v2 rests,
    rounded to the vocabulary's rest decimals; reference = the v1 prior rotation reference; rest_null = the rest_row
    of that header (the CFG L2-null rest)."""
    if not train_texts:
        raise ValueError("prior_header needs TRAIN texts")
    dec = _vocabulary.VOCABULARY["rest"]["decimals"]
    med = lambda v: round(float(np.median(v)), dec)  # noqa: E731
    rest = {k: ({a: med([t["rest"][k][a] for t in train_texts]) for a in v} if isinstance(v, dict)
                else med([t["rest"][k] for t in train_texts])) for k, v in train_texts[0]["rest"].items()}
    head = {"rest": rest, "reference": copy.deepcopy(priors["rotation_reference"])}
    empty = dict(head, hz=float(train_texts[0]["hz"]), n_frames=1, tendency=None, state=[], event=[])
    null = _decoder.decode(empty, names, codec["rest_stat"], codec["fitted"], tendency_q=codec["tendency_q"])["rest_row"]
    return dict(head, rest_null=null.tolist(), n_tracks=len(train_texts), priors_sha256=_representation.priors_sha256(priors))


def track_text_source(manifest: dict, track_id: str) -> dict:
    """The content-text source of one track of a prep_v2 manifest (TalkVid's versions.text_source, or its dataset's)."""
    ds = manifest["tracks"][track_id].get("dataset", "talkvid")
    return manifest["versions"]["text_source"] if ds == "talkvid" else manifest["xdata"]["text_sources"][ds]


def arrays_sha256(z: dict) -> str:
    """sha256 over the sorted arrays (name, dtype, shape, bytes): the npz content (the zip itself carries times)."""
    h = hashlib.sha256()
    for k in sorted(z):
        a = np.ascontiguousarray(z[k])
        h.update(("%s|%s|%s|" % (k, a.dtype.str, a.shape)).encode())
        h.update(a.tobytes())
    return h.hexdigest()


HEADER_KEYS = ("X_obs", "rot_curve")  # per variant and policy


HEADER_ONCE = ("rest", "reference")  # per track and policy (the variants keep the header)


TEXT_ARRAYS = ("M", "state", "asym", "state_on", "tend", "tend_mask")  # per variant, header-free


def variant_arrays(texts: list, names: list, policies: tuple, prior: dict, codec: dict, prefix: str = "",
                   target: np.ndarray = None) -> dict:
    """Arrays of a variant list: <prefix>v{r}_<X_obs|rot_curve>_<policy>, <prefix>v{r}_<M|state|...>,
    <prefix><rest|reference>_<policy>, delta; target given: <prefix>target_proj_v{r} (static_ref decode)."""
    z = {}
    for r, text in enumerate(texts):
        for pol in policies:
            c = _conditions.notes_conditions(text, names, pol, prior, codec)
            for k in HEADER_KEYS:
                z["%sv%d_%s_%s" % (prefix, r, k, pol)] = c[k]
            for k in HEADER_ONCE:
                key = "%s%s_%s" % (prefix, k, pol)
                if key in z and not np.array_equal(z[key], c[k]):
                    raise AssertionError("%s differs between variants" % key)
                z[key] = c[k]
            for k in TEXT_ARRAYS:
                key = "%sv%d_%s" % (prefix, r, k)
                if key in z and not np.array_equal(z[key], c[k]):
                    raise AssertionError("%s differs between header policies" % key)
                z[key] = c[k]
            z["delta"] = c["delta"]
            if target is not None and pol == "static_ref":
                z["%starget_proj_v%d" % (prefix, r)] = _decoder.Pi_delta(target, _conditions.projection_input(c))
    return z


def prep_track_v2(full: dict, raw: dict, tt: dict, split: str, names: list, params: dict, codec: dict,
                  prior: dict = None, n_variants: int = DRAW["n_variants_train"], mirrored: tuple = None,
                  draw: dict = DRAW) -> tuple:
    """(arrays, texts record) of one track. full: its encoded L2 v2 text; raw: the stitched tensor track (target /
    valid only); tt: the content-text track (load_conditioning_v2); mirrored: (mirrored full text, mirrored raw) for
    G7 (TRAIN). Every condition array comes from a text through condition_v2 (or from tt)."""
    if split not in SPLITS:
        raise AssertionError("prep_v2 refuses split %r" % split)
    n = int(full["n_frames"])
    tid = full["track_id"]
    z = {}
    z["text"], z["viseme"] = _script.features(tt, n, lexical="viseme_v2")
    feats = np.asarray(raw["features"], dtype=np.float64)
    if feats.shape != (n, D):
        raise ValueError("%s: tensor %s, L2 header %d frames" % (tid, feats.shape, n))
    z["target_static_ref"] = feats.copy()
    z["valid"] = valid_cells(raw, names)
    rec = {"track_id": tid, "split": split, "full": full}
    if split == "train":
        vs = train_variants(full, tid, params, codec, n_variants, draw=draw)
        z.update(variant_arrays([t for t, _ in vs], names, ("static_ref",), prior, codec, target=z["target_static_ref"]))
    else:
        vs = _conditions.validation_densities(full, params, codec)
        z["target_prior"] = rereference(feats, full["reference"], prior["reference"])
        z.update(variant_arrays([t for t, _ in vs], names, POLICIES, prior, codec))
    rec["variants"] = [{"r": r, "draw": d, "text": t} for r, (t, d) in enumerate(vs)]
    if mirrored is not None:
        mfull, mraw = mirrored
        if split != "train":
            raise AssertionError("G7 is TRAIN only")
        mt = np.asarray(mraw["features"], dtype=np.float64)
        if not np.array_equal(valid_cells(mraw, names), z["valid"]):
            raise AssertionError("%s: mirrored masks differ" % tid)
        z["g7_target_static_ref"] = mt.copy()
        gs = train_variants(mfull, tid, params, codec, DRAW["g7_mirror_variants"], g7=True, draw=draw)
        z.update(variant_arrays([t for t, _ in gs], names, ("static_ref",), prior, codec, prefix="g7_",
                                target=z["g7_target_static_ref"]))
        rec["g7"] = {"full": mfull, "variants": [{"r": r, "draw": d, "text": t} for r, (t, d) in enumerate(gs)]}
    return z, rec


def viseme_units(tt: dict) -> dict:
    """The per-word viseme units of a content-text track (gen.viseme._word), so G3c re-allocation needs no g2p; + the
    silent flag (ctext.SILENT) when the track carries it (absent otherwise: existing preps byte-identical)."""
    lang = tt["language"]
    out = {"language": lang, "words": [{"w": w["w"], "start_frame": int(w["start_frame"]),
                                        "end_frame": int(w["end_frame"]),
                                        "units": [list(u) for u in _script._word(w["w"], lang)[0]]} for w in tt["words"]]}
    if tt.get(_script.SILENT):
        out[_script.SILENT] = True
    return out


def condition_keys(z_keys, r: int, policy: str, prefix: str = "") -> dict:
    keys = {k: "%sv%d_%s_%s" % (prefix, r, k, policy) for k in HEADER_KEYS}
    keys.update({k: "%s%s_%s" % (prefix, k, policy) for k in HEADER_ONCE})
    keys.update({k: "%sv%d_%s" % (prefix, r, k) for k in TEXT_ARRAYS})
    keys.update({"delta": "delta", "text": "text", "viseme": "viseme"})
    if any(v.startswith(TARGET_PREFIXES) or v == "valid" for v in keys.values()):
        raise AssertionError("a condition key names a target")
    missing = [v for v in keys.values() if v not in z_keys]
    if missing:
        raise KeyError("no arrays %s" % missing)
    return keys


def norm_stats_v2(z_train: list, names: list, slots: list) -> dict:
    """stats_norm_v2 from the TRAIN tracks' arrays (module doc)."""
    targets = [z["target_static_ref"] for z in z_train]
    valids = [z["valid"] for z in z_train]
    conds = [(z["v0_X_obs_static_ref"], z["v0_M"], z["rest_static_ref"]) for z in z_train]
    ns = norm_stats(targets, valids, conds, excluded_columns(names))
    s = np.concatenate([z["v0_state"].astype(np.float64) for z in z_train])
    rms = np.sqrt((s ** 2).mean(0))
    tv = np.concatenate([z["v0_tend"].astype(np.float64) for z in z_train])
    tm = np.concatenate([z["v0_tend_mask"] for z in z_train])
    tmean = np.array([tv[tm[:, k], k].mean() if tm[:, k].any() else 0.0 for k in range(tv.shape[1])])
    tstd = np.array([tv[tm[:, k], k].std() if tm[:, k].any() else 0.0 for k in range(tv.shape[1])])
    ns.update({"format": "gen_prep_v2_norm", "policy": "static_ref", "split": "train", "n_tracks": len(z_train),
               "state_slots": slots, "state_rms_raw": rms.tolist(), "state_rms": np.maximum(rms, STD_FLOOR).tolist(),
               "tend_params": list(_tendency.PARAMS), "tend_mean": tmean.tolist(), "tend_std_raw": tstd.tolist(),
               "tend_std": np.maximum(tstd, STD_FLOOR).tolist(), "tend_n_frames": tm.sum(0).tolist(),
               "std_floor": STD_FLOOR, "source": "TRAIN variant 0 (full text) static_ref"})
    return ns


class PrepV2:
    """A written prep_v2 dir (README "gen prep v2")."""

    def __init__(self, path: str, check_sha: bool = False):
        self.path = path
        self.manifest = json.load(open(os.path.join(path, "manifest.json")))
        if self.manifest.get("format") != FORMAT:
            raise ValueError("%s is not a %s prep" % (path, FORMAT))
        if check_sha:
            for e in self.manifest["tracks"].values():
                for f, sha in ((e["file"], e["sha256"]), (e["texts_file"], e["texts_sha256"])):
                    if sha256_file(os.path.join(path, f)) != sha:
                        raise ValueError("%s: sha256 mismatch" % f)

    def ids(self, split: str, status: str = None) -> list:
        return [t for t, e in self.manifest["tracks"].items() if e["split"] == split and (status is None or
                                                                                      e["status"] == status)]

    def track(self, track_id: str) -> dict:
        """Every stored array, targets / valid included: training-target access only (never a generation input)."""
        with np.load(os.path.join(self.path, self.manifest["tracks"][track_id]["file"])) as z:
            return {k: z[k] for k in z.files}

    def texts(self, track_id: str) -> dict:
        return json.load(open(os.path.join(self.path, self.manifest["tracks"][track_id]["texts_file"])))

    def stats(self) -> dict:
        return json.load(open(os.path.join(self.path, self.manifest["stats"]["norm_v2"]["file"])))

    def condition(self, track_id: str, r: int, policy: str = _conditions.PRIMARY_POLICY, prefix: str = "", raw: bool = False) -> dict:
        """The condition arrays of variant / row r (prefix "g7_": a mirrored variant) and nothing else; no target_* /
        valid is read. raw: the stored arrays (condition_v2 layout + text / viseme); else normalised_v2 with this
        dir's stats + text / viseme."""
        e = self.manifest["tracks"][track_id]
        with np.load(os.path.join(self.path, e["file"])) as z:
            keys = condition_keys(z.files, r, policy, prefix)
            a = {n: z[k] for n, k in keys.items()}
        if raw:
            return a
        out = _conditions.normalise_conditions(a, self.stats())
        out.update(text=a["text"], viseme=a["viseme"])
        return out


POLICIES_prep = ("static_ref", "prior")  # (b) primary, (a) reported


ROT, TRANS, GAZE = slice(55, 61), slice(52, 55), slice(61, 63)


D = _representation.TOTAL_DIM


X_STD_FLOOR = 1e-3


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def track_file(track_id: str) -> str:
    return track_id.replace("#", "__seg") + ".npz"


def ref_vec(ref: dict) -> np.ndarray:
    return np.array([ref[_representation.ROTATION_TEXT][a] for a in _representation.ROTATION_AXES], dtype=np.float64)


def rereference(x: np.ndarray, ref_clip: dict, ref_prior: dict) -> np.ndarray:
    """(T, 63) float64 with rot6d R_prior . R_ref,clip^-1 . R_true: the rotation vector about the clip reference
    (l1.rotvec_deg_from_r6), re-expressed about the prior (l1.r6_from_rotvec_deg). Other columns unchanged."""
    out = np.array(x, dtype=np.float64, copy=True)
    out[:, ROT] = _representation.r6_from_omega(_representation.omega_from_r6(out[:, ROT], ref_vec(ref_clip)), ref_vec(ref_prior))
    return out


def valid_cells(raw_track: dict, names: list) -> np.ndarray:
    """(T, 63) target validity from the tracker masks (decode_generation_batch.described_cols)."""
    return _evaluate.described_cols(raw_track, names)


TARGET_PREFIXES_prep = ("target_", "g7_target_")  # with "valid": the arrays that are never a condition


def r6_matrix(r6: np.ndarray) -> np.ndarray:
    """(..., 6) -> (..., 3, 3) Gram-Schmidt (FD headpose.rotation_6d_to_matrix column order)."""
    r6 = np.asarray(r6, dtype=np.float64)
    a1, a2 = r6[..., 0:3], r6[..., 3:6]
    b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)
    b2 = a2 - (b1 * a2).sum(-1, keepdims=True) * b1
    b2 = b2 / np.linalg.norm(b2, axis=-1, keepdims=True)
    return np.stack([b1, b2, np.cross(b1, b2)], axis=-1)


def runs(mask: np.ndarray) -> list:
    """[(s, e)] maximal True runs of a 1-D bool array."""
    m = np.concatenate([[False], np.asarray(mask, dtype=bool), [False]])
    d = np.flatnonzero(m[1:] != m[:-1])
    return list(zip(d[::2], d[1::2]))


def orthonormalise_r6(r6: np.ndarray) -> np.ndarray:
    m = r6_matrix(r6)
    return np.concatenate([m[..., :, 0], m[..., :, 1]], axis=-1)


def r1_interp(x_obs: np.ndarray, m: np.ndarray, rest: np.ndarray, excluded_cols=()) -> np.ndarray:
    """R1: per column, linear interpolation between the covered neighbours of each interior gap; leading / trailing
    gaps, uncovered columns and excluded columns = rest; interpolated rotation frames Gram-Schmidt re-orthonormalised."""
    y = np.where(m, x_obs, np.asarray(rest, dtype=np.float64)[None, :])
    n = len(y)
    for c in range(D):
        if c in excluded_cols:
            continue
        for s, e in runs(~m[:, c]):
            if s > 0 and e < n:
                a, b = y[s - 1, c], y[e, c]
                t = (np.arange(s, e) - (s - 1)) / float(e - (s - 1))
                y[s:e, c] = a + (b - a) * t
    gap = ~m[:, ROT.start]
    if gap.any():
        y[gap, ROT] = orthonormalise_r6(y[gap, ROT])
    return y


def excluded_columns(names: list) -> list:
    ex = _representation.CHANNEL_RULES["channels"]["blendshape"]["exclude"]
    bs = _representation.channel("blendshape")
    return [bs.start + k for k, n in enumerate(names) if n in ex]


def norm_stats(targets: list, valids: list, conds: list, excluded: list) -> dict:
    """x_mean / x_std over valid target cells and G1's s_col = std of (target - R1) over M = 0 valid cells, per column,
    std floored at X_STD_FLOOR. conds: [(X_obs, M, rest)] of the same tracks and policy."""
    x = np.concatenate(targets)
    ok = np.concatenate(valids)
    mean = np.array([x[ok[:, c], c].mean() if ok[:, c].any() else 0.0 for c in range(D)])
    std = np.array([x[ok[:, c], c].std() if ok[:, c].any() else 0.0 for c in range(D)])
    res, rm = [], []
    for t, vld, (xo, m, rest) in zip(targets, valids, conds):
        res.append(t - r1_interp(xo, m, rest, excluded))
        rm.append(~m & vld)
    r, rmask = np.concatenate(res), np.concatenate(rm)
    s_col = np.array([r[rmask[:, c], c].std() if rmask[:, c].any() else 0.0 for c in range(D)])
    return {"x_mean": mean.tolist(), "x_std_raw": std.tolist(), "x_std": np.maximum(std, X_STD_FLOOR).tolist(),
            "s_col_raw": s_col.tolist(), "s_col": np.maximum(s_col, X_STD_FLOOR).tolist(), "floor": X_STD_FLOOR,
            "n_cells_valid": ok.sum(0).tolist(), "n_cells_s_col": rmask.sum(0).tolist(),
            "s_col_definition": "std of (target - R1(X_obs, M, rest)) over M = 0 valid TRAIN cells (fold decodes)"}


class Prep:
    """A written prep dir: manifest, stats and per-track arrays (README "gen-mode prep")."""

    def __init__(self, path: str, check_sha: bool = False):
        self.path = path
        self.manifest = json.load(open(os.path.join(path, "manifest.json")))
        if check_sha:
            for e in self.manifest["tracks"].values():
                if sha256_file(os.path.join(path, e["file"])) != e["sha256"]:
                    raise ValueError("%s: sha256 mismatch" % e["file"])

    def ids(self, split: str, status: str = None) -> list:
        return [t for t, e in self.manifest["tracks"].items() if e["split"] == split and (status is None or
                                                                                      e["status"] == status)]

    def track(self, track_id: str) -> dict:
        """Every stored array, targets / valid included: training-target access only (never a generation input)."""
        e = self.manifest["tracks"][track_id]
        with np.load(os.path.join(self.path, e["file"])) as z:
            return {k: z[k] for k in z.files}

    def condition(self, track_id: str, policy: str, prefix: str = "", k: int = None, raw: bool = False) -> dict:
        """The ONLY loader generation / inference may use (D244): the condition arrays of a track and nothing else,
        {"X_obs" (normalised_condition with this dir's stats_norm_<policy>), "M", "rest", "phrase_span", "phrase_type",
        "text", "letters"}; prefix "g3_" (copy k) / "g7_" as normalised_condition. No target_* / valid is even read.
        raw=True (T0511): X_obs = the stored decode (T, 63) f64, NaN where M = 0, M bool, rest the raw row (no stats
        read): what the G-diff sampler / hard copy and G-detail take (gen.sample run_from_obs, gen.detail refine)."""
        if policy not in POLICIES_prep:
            raise ValueError("policy %r" % policy)
        g7 = "g7_" if prefix == "g7_" else ""
        keys = {"x": prefix + "X_obs_" + policy, "m": prefix + "M", "rest": g7 + "rest_" + policy,
                "span": prefix + "phrase_span", "type": g7 + "phrase_type", "text": "text", "letters": "letters"}
        if any(v.startswith(TARGET_PREFIXES_prep) or v == "valid" for v in keys.values()):
            raise AssertionError("a condition key names a target")
        e = self.manifest["tracks"][track_id]
        with np.load(os.path.join(self.path, e["file"])) as z:
            a = {n: z[v] for n, v in keys.items()}
        if raw:
            x, m = (a["x"][k], a["m"][k]) if k is not None else (a["x"], a["m"])
            span = a["span"][k] if prefix == "g3_" and k is not None else a["span"]
            return {"X_obs": x, "M": m.astype(bool), "rest": a["rest"], "phrase_span": span, "phrase_type": a["type"],
                    "text": a["text"], "letters": a["letters"]}
        xn, m, rest = normalised_condition({keys["x"]: a["x"], keys["m"]: a["m"], keys["rest"]: a["rest"]},
                                           self.stats("norm", policy), policy, prefix, k)
        span = a["span"][k] if prefix == "g3_" and k is not None else a["span"]
        return {"X_obs": xn, "M": m, "rest": rest, "phrase_span": span, "phrase_type": a["type"], "text": a["text"],
                "letters": a["letters"]}

    def stats(self, kind: str, policy: str = _conditions.PRIMARY_POLICY) -> dict:
        e = self.manifest["stats"]["%s_%s" % (kind, policy)]
        return json.load(open(os.path.join(self.path, e["file"])))


def normalised_condition(z: dict, norm: dict, policy: str = _conditions.PRIMARY_POLICY, prefix: str = "", k: int = None) -> tuple:
    """(X_obs_n (T, 63) float32 = M (X_obs - x_mean) / x_std with 0 (= the TRAIN mean) where M = 0, M (T, 63) float32,
    rest_n (63,) float32) of a track's arrays; prefix "g3_" with copy k, or "g7_"."""
    x, m = z[prefix + "X_obs_" + policy], z[prefix + "M"]
    if k is not None:
        x, m = x[k], m[k]
    rest = z[("g7_" if prefix == "g7_" else "") + "rest_" + policy]
    mean, std = np.asarray(norm["x_mean"]), np.asarray(norm["x_std"])
    xn = np.where(m, (np.where(m, x, 0.0) - mean) / std, 0.0)
    return xn.astype(np.float32), m.astype(np.float32), ((rest - mean) / std).astype(np.float32)


POLICIES = POLICIES_prep  # ("static_ref", "prior"): (b) primary, (a) reported


TARGET_PREFIXES = TARGET_PREFIXES_prep  # target_* (target_static_ref / target_prior / target_proj_v*), g7_target_*
