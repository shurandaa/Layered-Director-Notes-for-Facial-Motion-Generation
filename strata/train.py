"""Training of the generator (paper Sec. 3.4 "training", Eq. (5); Supp. C "diffusion training").

v-parameterisation with the cosine schedule; the target is the ground-truth motion of the whole clip, clipped to
x_hat +- delta on the constrained entries and normalised per channel (d_0); the loss is the mean squared error of
v over the entries with valid tracking.  AdamW, lr 2e-4, 500 warm-up steps and cosine decay, weight decay 0.01,
gradient clipping 1.0, EMA 0.9999, 50k iterations, batch 128 windows of 96 frames.
"""
import copy
import hashlib
import json
import math
import numpy as np
import os
import time
import torch
from strata import conditions as _conditions
from strata import evaluate as _evaluate
from strata import network as _network
from strata import prep as _prep
from strata import sampler as _sampler
from strata import script as _script


PATHS = ("G1", "G2", "G3", "G3c", "G4", "G6", "G7", "G8")


ARMS = ("l2_text", "text_only", "l2_only")  # §9.1 (design §5a; review FINAL layouts)


FRAME_BUDGET = 128 * 96  # 12 288 real frames per batch for every window (design §2 rule; D246)


BATCH = {"W96": 128, "W192": 64, "Wfull": FRAME_BUDGET}


CONFIG_VERSION = 1


OPTIONAL_KEYS = ("timeline", "l2v2", "blink_w")  # absent from DEFAULT_CONFIG: a config without them is byte-identical to before


L2V2_DEFAULT = {"obs_mode": "bounded", "lexical": "viseme_v2", "state": True, "tendency": True,
                "cfg_p": {"text": 0.2, "l2": 0.1, "both": 0.1}}


L2V2_OPTIONAL = ("dataset_weights", "dataset_share")


DATASET_SHARES = ("frames",)


OBS_MODES = ("bounded", "copy")


L2V2_REFUSED_PATHS = ("G2", "G3")


DEFAULT_CONFIG = {
    "config_version": CONFIG_VERSION,
    # T0507 prep (T0514): (b) arrays + stats bit-identical to W/T0486/prep, plus the TRAIN rest priors that rest_null and
    # the (a) conditions (gen.sample val --condition_policy prior) need
    "prep": "/work/neu/p2026_0014_neu/facial_dynamic/runs/ai_workflow/T0507/prep",
    "prep_manifest_sha256": None,  # filled by gen.fit from the prep manifest (data identity is part of the config)
    "policy": "static_ref",  # Q022 (b) PRIMARY; "prior" = path (a), a reported row
    "window": "W96",  # W96 / W192 / Wfull (design §2)
    "batch": None,  # None = BATCH[window] (Wfull: a real-frame budget, not a track count)
    "steps": 15000,
    "seed": 0,
    "lr": 2e-4, "weight_decay": 0.01, "warmup": 500, "grad_clip": 1.0,
    "ema": 0.9999,
    "ema_warmup": True,  # decay_k = min(ema, (1 + k) / (10 + k)); see README / T0488 result Issues
    "model": {"d_model": 256, "n_layers": 6, "n_heads": 4, "ff": 1024, "dropout": 0.1},
    "arm": "l2_text",
    "paths": {p: False for p in PATHS},
    "g2": {"p": 0.5, "p_historical": 0.5, "p_phrase_drop": 0.3},
    "g3": {"p_copy": 0.5},
    "g3c": {"p_jitter": 0.5, "p_retime": 0.25, "shift": 3, "p_word_drop": 0.1, "rate": [2.5, 4.5],
            "text_version": "v1", "text_dir": None},
    "g4": {"p_text": 0.2, "p_obs": 0.1, "p_both": 0.1},
    "g6": {"lambda": 1.0, "edge_weight": 4.0, "edge_radius": 3},
    "g7": {"p": 0.5},
    "sampler": {"steps": 25, "s_o": 1.0, "s_t": 1.5, "repaint_u": 1},  # s_o / s_t used only with G4 (m1: (1.0, 1.5))
    "log_every": 500,
}


def check_config(c: dict) -> dict:
    unknown = set(c) - set(DEFAULT_CONFIG) - set(OPTIONAL_KEYS)
    if unknown:
        raise ValueError("unknown config keys %s" % sorted(unknown))
    if c["window"] not in _sampler.WINDOWS:
        raise ValueError("window must be one of %s" % sorted(_sampler.WINDOWS))
    if c["arm"] not in ARMS:
        raise ValueError("arm must be one of %s" % (ARMS,))
    if set(c["paths"]) != set(PATHS):
        raise ValueError("paths must have exactly %s" % (PATHS,))
    if c["policy"] not in ("static_ref", "prior"):
        raise ValueError("policy %r" % c["policy"])
    if c["arm"] != "l2_text" and any(c["paths"].values()):
        raise ValueError("the §9.1 arms run on G0* only (no path flag)")
    if "timeline" in c:
        if not isinstance(c["timeline"], bool):
            raise ValueError("timeline must be true / false")
        if not c["timeline"]:
            del c["timeline"]  # off = absent: the primary config bytes (and sha) stay those of the run without it
        elif c["arm"] != "l2_text":
            raise ValueError("the timeline arm replaces the L2 condition of the l2_text arm only")
    if "l2v2" in c:
        c["l2v2"] = check_l2v2(c)
    if "blink_w" in c:
        c["blink_w"] = check_blink_w(c)
        if c["blink_w"] is None:
            del c["blink_w"]  # weight 1 = off = absent (config bytes of the run without it)
    return c


BLINK_W_DEFAULT = {"weight": 1.0, "thresh": 0.2, "radius": 2}


BLINK_COLS = ("eyeBlinkLeft", "eyeBlinkRight")


def check_blink_w(c: dict):
    """The full blink_w block, or None when weight == 1 (off); l2v2 configs only."""
    v = c["blink_w"]
    if not isinstance(v, dict) or set(v) - set(BLINK_W_DEFAULT):
        raise ValueError("blink_w must be a dict of %s" % sorted(BLINK_W_DEFAULT))
    v = dict(BLINK_W_DEFAULT, **v)
    if not (v["weight"] >= 1.0 and v["thresh"] > 0 and int(v["radius"]) == v["radius"] >= 0):
        raise ValueError("blink_w: weight >= 1, thresh > 0, radius an integer >= 0")
    if "l2v2" not in c:
        raise ValueError("blink_w needs an l2v2 config")
    v = {"weight": float(v["weight"]), "thresh": float(v["thresh"]), "radius": int(v["radius"])}
    return None if v["weight"] == 1.0 else v


def check_l2v2(c: dict) -> dict:
    """The full l2v2 block (defaults filled in, idempotent) of a config; refuses G2 / G3 / timeline / policy prior."""
    v = c["l2v2"]
    if not isinstance(v, dict):
        raise ValueError("l2v2 must be a dict of %s" % sorted(L2V2_DEFAULT))
    unknown = set(v) - set(L2V2_DEFAULT) - set(L2V2_OPTIONAL)
    if unknown:
        raise ValueError("unknown l2v2 fields %s" % sorted(unknown))
    full = copy.deepcopy(L2V2_DEFAULT)
    for k, x in v.items():
        if k == "cfg_p":
            if not isinstance(x, dict) or set(x) - set(L2V2_DEFAULT["cfg_p"]):
                raise ValueError("l2v2.cfg_p must hold %s" % sorted(L2V2_DEFAULT["cfg_p"]))
            full["cfg_p"].update(x)
        else:
            full[k] = x
    if full["obs_mode"] not in OBS_MODES:
        raise ValueError("l2v2.obs_mode must be one of %s" % (OBS_MODES,))
    if full["lexical"] not in _network.L2V2_LEXICAL:
        raise ValueError("l2v2.lexical must be one of %s" % sorted(_network.L2V2_LEXICAL))
    if not all(isinstance(full[k], bool) for k in ("state", "tendency")):
        raise ValueError("l2v2.state / l2v2.tendency must be true / false")
    p = full["cfg_p"]
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) or x < 0 for x in p.values()) or sum(p.values()) > 1:
        raise ValueError("l2v2.cfg_p: probabilities >= 0 with sum <= 1")
    if "dataset_weights" in full:
        w = full["dataset_weights"]
        if not isinstance(w, dict) or not w or any(not isinstance(k, str) or isinstance(x, bool) or
                                                   not isinstance(x, (int, float)) or not x > 0 for k, x in w.items()):
            raise ValueError("l2v2.dataset_weights: {dataset: weight > 0}")
    if "dataset_share" in full:
        if full["dataset_share"] not in DATASET_SHARES:
            raise ValueError("l2v2.dataset_share must be one of %s" % (DATASET_SHARES,))
        if "dataset_weights" not in full:
            raise ValueError("l2v2.dataset_share needs l2v2.dataset_weights")
    bad = [q for q in L2V2_REFUSED_PATHS if c["paths"][q]]
    if bad:
        raise ValueError("l2v2 refuses the v1-phrase paths %s" % bad)
    if "timeline" in c:
        raise ValueError("l2v2 refuses the timeline arm")
    if c["policy"] != "static_ref":
        raise ValueError("l2v2 trains under policy static_ref only ((a) = gen.sample condition_policy prior)")
    return full


def timeline_on(c: dict) -> bool:
    return bool(c.get("timeline", False))


def l2v2_on(c: dict) -> bool:
    return "l2v2" in c


def n_v2_inputs(c: dict, slots: dict) -> int:
    """Width of the v2 block from the prep manifest's slot lists {"state", "asym", "tendency"}."""
    n = {"state": len(slots["state"]), "asym": len(slots["asym"]), "state_on": 1, "tend": len(slots["tendency"]),
         "tend_mask": len(slots["tendency"])}
    return sum(n[k] for k in _conditions.notes_block_parts(c))


def load_config(path: str) -> dict:
    with open(path) as f:
        c = json.load(f)
    full = copy.deepcopy(DEFAULT_CONFIG)
    for k, v in c.items():
        if isinstance(v, dict) and isinstance(full.get(k), dict):
            full[k].update(v)
        else:
            full[k] = v
    return check_config(full)


def config_bytes(c: dict) -> bytes:
    return json.dumps(c, sort_keys=True, separators=(",", ":")).encode()


def config_sha256(c: dict) -> str:
    return hashlib.sha256(config_bytes(c)).hexdigest()


def batch_size(c: dict) -> int:
    return int(c["batch"] or BATCH[c["window"]])


def path_name(c: dict) -> str:
    on = [p for p in PATHS if c["paths"][p]]
    base = "G0" if not on else "+".join(on)
    if l2v2_on(c):
        return (base if c["arm"] == "l2_text" else c["arm"]) + "_V2"
    if timeline_on(c):
        return base + "_TL"
    return base if c["arm"] == "l2_text" else c["arm"]


def free_columns(excluded: list) -> np.ndarray:
    f = np.ones(_network.D, dtype=bool)
    f[list(excluded)] = False
    return f


def r1_fast(x_obs: np.ndarray, m: np.ndarray, rest: np.ndarray, excluded=()) -> np.ndarray:
    """gen.prep.r1_interp without the per-column python loop: linear interpolation inside interior gaps, rest on
    leading / trailing gaps, uncovered and excluded columns; interpolated rotation frames re-orthonormalised."""
    m = np.asarray(m, dtype=bool)
    n = len(m)
    rest = np.asarray(rest, dtype=np.float64)
    x = np.where(m, x_obs, 0.0).astype(np.float64)
    y = np.where(m, x, rest[None, :])
    t = np.arange(n)[:, None]
    prev = np.maximum.accumulate(np.where(m, t, -1), axis=0)
    nxt = np.minimum.accumulate(np.where(m, t, n)[::-1], axis=0)[::-1]
    inner = ~m & (prev >= 0) & (nxt < n)
    if excluded:
        inner[:, list(excluded)] = False
    if inner.any():
        cols = np.broadcast_to(np.arange(_network.D), (n, _network.D))
        p, q = np.clip(prev, 0, n - 1), np.clip(nxt, 0, n - 1)
        a, b = x[p, cols], x[q, cols]
        w = (t - prev) / np.maximum(q - prev, 1)
        y = np.where(inner, a + (b - a) * w, y)
    gap = ~m[:, _network.ROT.start]
    if gap.any():
        y[gap, _network.ROT] = _orthonormalise_np(y[gap, _network.ROT])
    return y


def _orthonormalise_np(r6: np.ndarray) -> np.ndarray:
    return _prep.orthonormalise_r6(r6)


def ema_decay(c: dict, k: int) -> float:
    if c["ema_warmup"]:
        return min(float(c["ema"]), (1.0 + k) / (10.0 + k))
    return float(c["ema"])


def lr_lambda(c: dict):
    warm, steps = int(c["warmup"]), int(c["steps"])
    return lambda i: min(1.0, (i + 1) / max(warm, 1)) * 0.5 * (1 + math.cos(math.pi * min(i, steps) / steps))


D = _network.D


class Letters:
    """Cached anyascii letter ids per word string (gen.textfeat.word_letters)."""

    def __init__(self, on: bool):
        self.on, self.cache = on, {}

    def ids(self, w: str) -> list:
        if w not in self.cache:
            self.cache[w] = _script.word_letters(w)
        return self.cache[w]


def text_features(words: list, n: int, lt: Letters, silent: bool = False) -> tuple:
    """gen.textfeat.features of a word list (marks re-derived), letters from the cache (same split rule); silent = the
    track's ctext.SILENT flag (no words only). A talking track whose G3c perturbation drops every word stays a track
    without the flag, i.e. the null text (speech unknown), never the silent condition."""
    tr = {"words": words, "marks": _script.derive_marks(words)}
    if silent:
        tr[_script.SILENT] = True
    x, ids = _script.features(tr, n, letters=False)
    if lt.on:
        for w in words:
            s, e = w["start_frame"], w["end_frame"]
            k = e - s + 1
            ids_w = np.asarray(lt.ids(w["w"]), dtype=np.int16)
            ids[s:e + 1] = ids_w[(np.arange(k) * len(ids_w)) // k]
    return x, ids


def jitter_words(words: list, n: int, rng, shift: int = 3, p_drop: float = 0.1) -> list:
    """G3c: each word's start and end move by U{-shift..shift}; a word starts after the previous (moved) word's end,
    ends before the next word's original start and inside [0, n), keeps >= 1 frame (else it keeps its original frames,
    always valid); then each word is dropped with p_drop (its frames become non-word)."""
    out = []
    prev_end = -1
    for i, w in enumerate(words):
        s0, e0 = w["start_frame"], w["end_frame"]
        hi = (words[i + 1]["start_frame"] - 1) if i + 1 < len(words) else n - 1
        ds, de = (int(v) for v in rng.integers(-shift, shift + 1, 2))
        s1, e1 = max(s0 + ds, prev_end + 1, 0), min(e0 + de, hi, n - 1)
        if e1 < s1:
            s1, e1 = s0, e0
        out.append({"w": w["w"], "start_frame": int(s1), "end_frame": int(e1)})
        prev_end = e1
    keep = rng.random(len(out)) >= p_drop
    return [w for w, k in zip(out, keep) if k]


def perturb_text(words: list, n: int, g: dict, rng) -> tuple:
    """(words', kind) of G3c (module doc)."""
    if not words:
        return words, "none"
    r = rng.random()
    if r < g["p_retime"]:
        rate = float(rng.uniform(*g["rate"]))
        return _script.retime_words(words, n, rate=rate)[0]["words"], "retime"
    if r < g["p_retime"] + g["p_jitter"]:
        return jitter_words(words, n, rng, g["shift"], g["p_word_drop"]), "jitter"
    return words, "none"


def _read(prep: _prep.Prep, track_id: str, keys) -> dict:
    with np.load(os.path.join(prep.path, prep.manifest["tracks"][track_id]["file"])) as z:
        return {k: z[k] for k in keys}


def _is_target(key: str) -> bool:
    return key.startswith(_prep.TARGET_PREFIXES_prep) or key == "valid"


def target_arrays(prep: _prep.Prep, track_id: str, keys) -> dict:
    """Training-target access (target_* / g7_target_* / valid only; never a model input)."""
    if not all(_is_target(q) for q in keys):
        raise AssertionError("target_arrays reads target / valid keys only")
    return _read(prep, track_id, keys)


CODEC_KEYS = ("vocabulary_version", "vocabulary_sha256", "fitted_sha256", "rest_stat", "tendency_version",
              "tendency_rules_sha256", "tendency_quantiles_sha256")


def live_codec_mismatch(versions: dict) -> list:
    """The FD L2 v2 constants (vocabulary / fitted / tendency) whose live sha / version differs from `versions` (a
    prep_v2 manifest's or a checkpoint's): the decode must be the one the data was built with."""
    live = _prep.codec_info(_conditions.load_constants(versions["rest_stat"]))
    return [k for k in CODEC_KEYS if live[k] != versions[k]]


def dataset_track_p(datasets: list, size, weights: dict = None, n_real=None):
    """Per-track draw probabilities: dataset d gets weights[d] (normalised over the datasets present), split inside d
    by size (start count, or Wfull frames). None without weights. A present dataset without a weight raises.
    W96 / W192: the share of crops drawn per dataset; Wfull: the expected share of real frames per dataset (TrainData.batch
    picks inside a length bucket by track_p / n_eff).
    n_real (W96 / W192 under l2v2.dataset_share "frames", R1-M4 / R3-M2 2026-10-08): the real frames of one crop of each
    track, min(n, L); dataset d's mass is then rescaled by 1 / its expected real frames per crop, so weights[d] is the
    expected share of REAL (loss) frames, as under Wfull."""
    if not weights:
        return None
    present = sorted(set(datasets))
    missing = [d for d in present if d not in weights]
    if missing:
        raise ValueError("l2v2.dataset_weights has no weight for %s" % missing)
    size = np.asarray(size, dtype=np.float64)
    if not np.all(size > 0) or any(not weights[d] > 0 for d in present):
        raise ValueError("dataset_track_p: sizes and weights must be > 0")
    ds = np.asarray(datasets)
    tot = sum(weights[d] for d in present)
    p = np.zeros(len(ds))
    for d in present:
        m = ds == d
        q = size[m] / size[m].sum()
        p[m] = weights[d] / tot * (q if n_real is None else q / float(q @ np.asarray(n_real, dtype=np.float64)[m]))
    return p / p.sum()


def viseme_ids(words: list, n: int, units_of: dict) -> np.ndarray:
    """(n,) int16 viseme ids of a word list from cached per-word units (gen.prep_v2 viseme_units.json; =
    gen.viseme.viseme_features with the g2p already done): gen.viseme.allocate per word, 0 off words."""
    ids = np.zeros(n, dtype=np.int16)
    for w in words:
        s, e = w["start_frame"], w["end_frame"]
        u = units_of[w["w"]]
        ids[s:e + 1] = [u[j][0] for j in _script.allocate(u, e - s + 1)]
    return ids


class TrainDataV2:
    """The training clips of a prep directory (strata.prep) for the final configuration."""

    def mean_std(self):
        return np.asarray(self.norm["x_mean"], dtype=np.float64), np.asarray(self.norm["x_std"], dtype=np.float64)

    def pad(self, s: dict) -> dict:
        L = self.L
        k = len(s["m"])
        out = {}
        for key, a in s.items():
            if key == "rest":
                out[key] = np.asarray(a, dtype=np.float32)
                continue
            out[key] = np.concatenate([a, np.zeros((L - k,) + a.shape[1:], dtype=a.dtype)]) if k < L else a
        out["pad"] = np.arange(L) < k
        return out

    def batch(self, B: int, rng) -> dict:
        """W96 / W192: B crops. Wfull: B = the real-frame budget (module doc), arrays cut to the longest track."""
        if self.full:
            q = self.buckets[int(rng.choice(len(self.buckets), p=self.bucket_p))]
            ks, real = [], 0
            while real < B:
                # weighted: inside the bucket by track_p / n_eff, so each dataset's expected FRAME share is its
                # track_p mass (a draw brings n_eff frames); unweighted this is the uniform draw it reduces to
                ks.append(int(q[rng.integers(len(q))]) if self.track_p is None else
                          int(rng.choice(q, p=(self.track_p[q] / self.n_eff[q]) / (self.track_p[q] / self.n_eff[q]).sum())))
                real += int(self.n_eff[ks[-1]])
            samples = [self.sample(k, rng) for k in ks]
            n = int(max(self.n_eff[k] for k in ks))
            return {key: np.stack([s[key] if key == "rest" else s[key][:n] for s in samples]) for key in samples[0]}
        ks = rng.choice(len(self.tracks), size=B,
                        p=self.n_starts / self.n_starts.sum() if self.track_p is None else self.track_p)
        samples = [self.sample(int(k), rng) for k in ks]
        return {key: np.stack([s[key] for s in samples]) for key in samples[0]}


    def __init__(self, c: dict, prep=None, track_ids: list = None, log=print):
        self.c = c
        self.prep = prep or _prep.PrepV2(c["prep"], check_sha=True)
        man = self.prep.manifest
        if man.get("format") != _prep.FORMAT:
            raise ValueError("an l2v2 config needs a %s prep" % _prep.FORMAT)
        bad = live_codec_mismatch(man["versions"])
        if bad:
            raise ValueError("the prep_v2 was built with other FD L2 v2 constants than the live ones: %s" % bad)
        self.v2c = c["l2v2"]
        self.bounded = self.v2c["obs_mode"] == "bounded"
        self.lexical = self.v2c["lexical"]
        self.norm = self.prep.stats()
        self.names = man["names"]
        self.excluded = list(_prep.excluded_columns(self.names))
        self.free = free_columns(self.excluded)
        self.slots = man["slots"]
        self.n_v2 = n_v2_inputs(c, self.slots)
        self.timeline = False
        self.prior_header = json.load(open(os.path.join(self.prep.path, man["prior_header"]["file"])))
        self.rest_null = np.asarray(self.prior_header["rest_null"], dtype=np.float64)
        self.rest_null_resource = "prep_v2 prior_header (%d TRAIN tracks)" % self.prior_header["n_tracks"]
        P_ = c["paths"]
        self.R = int(man["draw"]["n_variants_train"])
        self.R7 = int(man["draw"]["g7_mirror_variants"]) if man["g7"] else 0
        if P_["G7"] and not self.R7:
            raise ValueError("G7 needs a prep_v2 with mirrored variants")
        ids = sorted(track_ids if track_ids is not None else self.prep.ids("train"))
        if any(man["tracks"][t]["split"] != "train" for t in ids):
            raise ValueError("TrainDataV2: only TRAIN tracks")
        self.lt = Letters(self.lexical == "letters")
        need_words = P_["G3c"] or self.lexical == "letters"
        units = json.load(open(os.path.join(self.prep.path, man["viseme_units"]["file"]))) if need_words else None
        self.tracks = []
        for tid in ids:
            e = man["tracks"][tid]
            n = int(e["n_frames"])
            keys = ["target_static_ref", "valid"] + (["target_proj_v%d" % r for r in range(1, self.R + 1)]
                                                     if self.bounded else [])
            if P_["G7"]:
                keys += ["g7_target_static_ref"] + (["g7_target_proj_v%d" % r for r in range(1, self.R7 + 1)]
                                                    if self.bounded else [])
            tg = target_arrays(self.prep, tid, keys)
            c0 = self.prep.condition(tid, 1, raw=True)
            tr = {"id": tid, "n": n, "text": c0["text"], "lex": c0["viseme"], "valid": tg["valid"].astype(bool),
                  "target": tg["target_static_ref"],
                  "var": [self._variant(tid, r, "", tg.get("target_proj_v%d" % r)) for r in range(1, self.R + 1)]}
            if P_["G7"]:
                tr["g7_target"] = tg["g7_target_static_ref"]
                tr["g7"] = [self._variant(tid, r, "g7_", tg.get("g7_target_proj_v%d" % r))
                            for r in range(1, self.R7 + 1)]
            if c.get("blink_w"):
                tr["blink"] = blink_frames(tr["target"], self.names, c["blink_w"])
                if P_["G7"]:
                    tr["g7_blink"] = blink_frames(tr["g7_target"], self.names, c["blink_w"])
            if need_words:
                u = units[tid]
                tr["words"] = [{"w": w["w"], "start_frame": w["start_frame"], "end_frame": w["end_frame"]}
                               for w in u["words"]]
                tr["silent"] = bool(u.get(_script.SILENT, False))
                tr["units"] = {}
                for w in u["words"]:
                    if tr["units"].setdefault(w["w"], w["units"]) != w["units"]:
                        raise AssertionError("%s: word %r with two unit lists" % (tid, w["w"]))
                if self.lexical == "letters":
                    tt = _script.load_track_text(e["clip_id"], tid, n, _prep.track_text_source(man, tid))
                    if [(w["w"], w["start_frame"], w["end_frame"]) for w in tt["words"]] != \
                            [(w["w"], w["start_frame"], w["end_frame"]) for w in tr["words"]]:
                        raise AssertionError("%s: viseme_units words != the content text" % tid)
                x, lx = self.text_of(tr, tr["words"])
                if not (np.array_equal(x, tr["text"]) and (self.lexical == "letters" or np.array_equal(lx, tr["lex"]))):
                    raise AssertionError("%s: words rebuilt from viseme_units differ from the prep text" % tid)
                tr["lex"] = lx
            self.tracks.append(tr)
        self.L = _sampler.window_len(c)
        self.full = c["window"] == "Wfull"
        self.n_starts = np.array([max(1, t["n"] - self.L + 1) for t in self.tracks])
        if self.full:
            self.n_eff = np.minimum([t["n"] for t in self.tracks], self.L)
            j = np.floor(np.log2(self.n_eff)).astype(int)
            self.buckets = [np.flatnonzero(j == q) for q in np.unique(j)]
            fr = np.array([self.n_eff[b].sum() for b in self.buckets], dtype=np.float64)
            self.bucket_p = fr / fr.sum()
        self.datasets = [man["tracks"][t["id"]].get("dataset", "talkvid") for t in self.tracks]
        n_real = None
        if self.v2c.get("dataset_share") == "frames" and not self.full:
            # every real frame of a dataset equally likely (a draw of track k covers min(n, L) of its n frames),
            # datasets by real-frame share (dataset_track_p); Wfull is frame-proportional already
            n_real = np.minimum([t["n"] for t in self.tracks], self.L).astype(np.float64)
        self.track_p = dataset_track_p(self.datasets, self.n_eff if self.full else
                                       (self.n_starts if n_real is None else np.array([t["n"] for t in self.tracks]) / n_real),
                                       self.v2c.get("dataset_weights"), n_real)
        if self.track_p is not None and self.full:
            self.bucket_p = np.array([self.track_p[b].sum() for b in self.buckets])
        if self.track_p is not None:
            share = {d: float(self.track_p[[x == d for x in self.datasets]].sum()) for d in sorted(set(self.datasets))}
            log("data v2: dataset_weights %s -> draw share %s" % (self.v2c["dataset_weights"], share))
            if n_real is not None:
                fr = {d: float((self.track_p * n_real)[[x == d for x in self.datasets]].sum() / (self.track_p @ n_real))
                      for d in sorted(set(self.datasets))}
                log("data v2: dataset_share frames -> real-frame share %s" % fr)
        self.var_scale = var_scale(c, self.norm)
        log("data v2: %d TRAIN tracks x %d variants (+%d G7), %d frames, obs_mode %s, lexical %s, n_v2 %d, window %s"
            % (len(self.tracks), self.R, self.R7 if P_["G7"] else 0, sum(t["n"] for t in self.tracks),
               self.v2c["obs_mode"], self.lexical, self.n_v2, c["window"]))

    def _variant(self, tid: str, r: int, prefix: str, target_proj) -> dict:
        """One stored variant: its raw condition (PrepV2.condition: no target / valid key) + target_proj (bounded)."""
        cond = self.prep.condition(tid, r, prefix=prefix, raw=True)
        cond = {k: v for k, v in cond.items() if k not in ("text", "viseme")}
        return {"cond": cond, "target_proj": target_proj}

    def text_of(self, tr: dict, words: list) -> tuple:
        """(timing, lexical ids) of a word list of track tr (letters: the anyascii ids; viseme_v2: the cached units)."""
        x, ids = text_features(words, tr["n"], self.lt, tr.get("silent", False))
        if self.lexical == "viseme_v2":
            ids = viseme_ids(words, tr["n"], tr["units"])
        return x, ids

    def meta(self, prep_sha: str) -> dict:
        """The checkpoint's l2v2_meta (T0597 spec_step5.json checkpoint_fields_added)."""
        man = self.prep.manifest
        return {"fd_commit": _evaluate.commits()["facial-dynamics"], "versions": man["versions"],
                "slots": man["slots"], "n_v2": self.n_v2, "names": self.names,
                "norm_v2": {k: self.norm[k] for k in ("state_rms", "tend_mean", "tend_std")}, "delta": man["delta"],
                "prior_header": self.prior_header, "prep_manifest_sha256": prep_sha, "prep_smoke": bool(man["smoke"]),
                "draw": man["draw"], "params": man["params"],
                # TRAIN tracks carrying the silent flag (R1-M1 2026-10-08: 0 = the silent condition was never trained;
                # gen.sample.generate_v2 warns); absent in older checkpoints = 0
                "silent_train_tracks": self.silent_train_tracks()}

    def silent_train_tracks(self) -> int:
        units = json.load(open(os.path.join(self.prep.path, self.prep.manifest["viseme_units"]["file"])))
        return sum(bool(units[t["id"]].get(_script.SILENT, False)) for t in self.tracks)

    # one sample ------------------------------------------------------------
    def sample(self, k: int, rng, start: int = None) -> dict:
        """The training sample of track k: G7 / variant draw, G3c, arm + G4 drops (module doc), then inputs()."""
        c, tr = self.c, self.tracks[k]
        P_ = c["paths"]
        g7 = bool(P_["G7"] and rng.random() < c["g7"]["p"])
        r = 1 + int(rng.integers(self.R7 if g7 else self.R))
        text = None
        if P_["G3c"]:
            words, _ = perturb_text(tr["words"], tr["n"], c["g3c"], rng)
            if words is not tr["words"]:
                text = self.text_of(tr, words)
        no_text = no_l2 = False
        if P_["G4"]:
            g, u = self.v2c["cfg_p"], rng.random()
            if u < g["text"]:
                no_text = True
            elif u < g["text"] + g["l2"]:
                no_l2 = True
            elif u < g["text"] + g["l2"] + g["both"]:
                no_text = no_l2 = True
        if start is None:
            start = int(rng.integers(0, max(1, tr["n"] - self.L + 1)))
        return self.inputs(k, r, g7, no_text, no_l2, start, text)

    def inputs(self, k: int, r: int, g7: bool = False, no_text: bool = False, no_l2: bool = False, start: int = 0,
               text: tuple = None) -> dict:
        """The padded training sample of track k, variant r (1-based; g7: the mirrored variant r), the given drops and
        crop start; text = (timing, lexical ids) overriding the track's (G3c)."""
        c, tr = self.c, self.tracks[k]
        v = (tr["g7"] if g7 else tr["var"])[r - 1]
        cond = v["cond"]
        target = tr["g7_target"] if g7 else tr["target"]
        tx, lx = text if text is not None else (tr["text"], tr["lex"])
        inp = _conditions.model_inputs(c, self.norm, cond, tx, lx, self.rest_null, no_text, no_l2)
        n, L = tr["n"], self.L
        sl = slice(start, start + min(L, n - start))
        Mi = inp["m"].astype(bool)
        mean, std = self.mean_std()
        x_dec = np.asarray(cond["X_obs"], dtype=np.float64)
        null_l2 = no_l2 or c["arm"] == "text_only"
        rest_eff = self.rest_null if null_l2 else np.asarray(cond["rest"], dtype=np.float64)
        if c["paths"]["G1"]:
            base = r1_fast(x_dec, Mi, rest_eff, self.excluded)[sl]
        else:
            base = np.broadcast_to(mean, (sl.stop - sl.start, D))
        sp = _sampler.Space(base, self.var_scale)
        Mc, vd = Mi[sl], tr["valid"][sl]
        if self.bounded:
            tg = (target if null_l2 else v["target_proj"])[sl]  # T0611: under the L2 null M' = 0, so the raw target
            d0 = np.where(vd, sp.to_var(np.where(vd, tg, 0.0)), 0.0)
            w = vd & self.free[None, :]
        else:
            tg, xd = target[sl], x_dec[sl]
            d0 = np.where(Mc, sp.to_var(np.where(Mc, xd, 0.0)), np.where(vd, sp.to_var(np.where(vd, tg, 0.0)), 0.0))
            w = vd & ~Mc & self.free[None, :]
        if c.get("blink_w"):
            w = blink_weight(w, (tr["g7_blink"] if g7 else tr["blink"])[sl], self.names, c["blink_w"]["weight"])
        out = {"x_obs": inp["x_obs"][sl], "m": Mc, "rest": inp["rest"], "text": inp["text"][sl],
               "letters": inp["letters"][sl], "v2": inp["v2"][sl], "d0": d0, "w": w}
        if c["paths"]["G6"]:
            out["base_n"] = (base - mean) / std
            out["tgt_n"] = np.where(vd, (np.where(vd, tg, 0.0) - mean) / std, 0.0)
            out["valid"] = vd
            if self.bounded:
                out["m_copy"] = np.zeros_like(Mc)  # no hard copy: the composite is x^0
        return self.pad(out)


def blink_frames(target: np.ndarray, names: list, bw: dict) -> np.ndarray:
    """(T,) bool: the blink frames of a target (model.check_blink_w): mean eyeBlinkL/R > its track median + thresh,
    dilated by radius frames."""
    x = np.asarray(target, dtype=np.float64)[:, [names.index(n) for n in BLINK_COLS]].mean(axis=1)
    b = x > np.median(x) + bw["thresh"]
    r = bw["radius"]
    if r and b.any():
        b = np.convolve(b.astype(np.int32), np.ones(2 * r + 1, dtype=np.int32), mode="same") > 0
    return b


def blink_weight(w: np.ndarray, blink: np.ndarray, names: list, weight: float) -> np.ndarray:
    """The loss weight w (T, 63) as float32, the eyeBlink columns x weight on the blink frames."""
    out = w.astype(np.float32)
    cols = [names.index(n) for n in BLINK_COLS]
    out[:, cols] *= np.where(blink, np.float32(weight), np.float32(1.0))[:, None]
    return out


TORCH_FLOAT = ("x_obs", "m", "rest", "text", "d0", "w", "base_n", "tgt_n", "timeline", "v2", "m_copy")


def to_torch(b: dict, device) -> dict:
    out = {}
    for k, v in b.items():
        if k in TORCH_FLOAT:
            out[k] = torch.as_tensor(np.asarray(v, dtype=np.float32), device=device)
        elif k == "letters":
            out[k] = torch.as_tensor(np.asarray(v, dtype=np.int64), device=device)
        else:
            out[k] = torch.as_tensor(np.asarray(v), device=device)
    return out


def edge_weight(m: torch.Tensor, pad: torch.Tensor, radius: int, weight: float) -> torch.Tensor:
    """(B, L-1, 63) weights of the first differences: `weight` within +-radius frames of an M edge (M[t] != M[t+1])
    of the column, else 1."""
    edge = (m[:, 1:] != m[:, :-1]).float() * (pad[:, 1:] & pad[:, :-1]).float()[..., None]
    k = 2 * radius + 1
    near = torch.nn.functional.max_pool1d(edge.transpose(1, 2), k, stride=1, padding=radius).transpose(1, 2)
    return 1.0 + (weight - 1.0) * near


def seam_loss(x0: torch.Tensor, b: dict, scale_n: torch.Tensor, free: torch.Tensor, g6: dict) -> torch.Tensor:
    """G6 (module doc): composite in G0-normalised tensor units vs the target, 1st + 2nd differences."""
    m = b["m"]
    mc = b.get("m_copy", m)  # v2 bounded mode: no hard copy (m_copy 0), the edges stay those of M
    comp = mc * b["x_obs"] + (1 - mc) * (b["base_n"] + x0 * scale_n)
    tgt = b["tgt_n"]
    ok = b["valid"].bool() & b["pad"][..., None]
    ok = ok.float() * free
    ew = edge_weight(m, b["pad"], int(g6["edge_radius"]), float(g6["edge_weight"]))
    d1c, d1t = comp[:, 1:] - comp[:, :-1], tgt[:, 1:] - tgt[:, :-1]
    w1 = ok[:, 1:] * ok[:, :-1] * ew
    d2c, d2t = d1c[:, 1:] - d1c[:, :-1], d1t[:, 1:] - d1t[:, :-1]
    w2 = ok[:, 2:] * ok[:, 1:-1] * ok[:, :-2] * torch.maximum(ew[:, 1:], ew[:, :-1])
    l1 = ((d1c - d1t) ** 2 * w1).sum() / w1.sum().clamp_min(1.0)
    l2 = ((d2c - d2t) ** 2 * w2).sum() / w2.sum().clamp_min(1.0)
    return float(g6["lambda"]) * (l1 + l2)


def training_loss(model, b: dict, c: dict, scale_n: torch.Tensor, generator=None) -> tuple:
    """(loss, {"v": v-MSE, "seam": G6 term or 0})."""
    d0 = b["d0"]
    B = d0.shape[0]
    t = torch.randint(0, _network.N_T, (B,), device=d0.device, generator=generator)
    eps = torch.randn(d0.shape, device=d0.device, generator=generator)
    a, s = _network.alpha_sigma(t)
    a, s = a[:, None, None], s[:, None, None]
    z = a * d0 + s * eps
    v = a * eps - s * d0
    pred = model(z, t, b["x_obs"], b["m"], b["rest"], b["text"], b["letters"], b["pad"], b.get("timeline"),
                 **({"v2": b["v2"]} if "v2" in b else {}))
    w = b["w"].float() * b["pad"][..., None].float()
    lv = ((pred - v) ** 2 * w).sum() / w.sum().clamp_min(1.0)
    parts = {"v": lv, "seam": torch.zeros((), device=d0.device)}
    if c["paths"]["G6"]:
        x0 = a * z - s * pred
        parts["seam"] = seam_loss(x0, b, scale_n, model.free, c["g6"])
    return lv + parts["seam"], parts


_POOL_DATA = None


def _pool_init(c: dict, prep_path: str, track_ids: list):
    global _POOL_DATA
    _POOL_DATA = TrainDataV2(c, _prep.PrepV2(prep_path), track_ids, log=lambda s: None)


def _pool_batch(args):
    B, seed, step = args
    return _POOL_DATA.batch(B, np.random.default_rng([seed, 1, step]))


def batches(data, B: int, seed: int, steps: int, workers: int = 0) -> tuple:
    """(iterator, close): training batch k (1..steps) = data.batch(B, default_rng([seed, 1, k])), so the stream does
    not depend on the number of workers (spawned processes that rebuild the same TrainData from the prep dir; results
    in order)."""
    if workers <= 0:
        return (data.batch(B, np.random.default_rng([seed, 1, k])) for k in range(1, steps + 1)), (lambda: None)
    import multiprocessing as mp
    pool = mp.get_context("spawn").Pool(workers, _pool_init, (data.c, data.prep.path, [t["id"] for t in data.tracks]))
    it = pool.imap(_pool_batch, ((B, seed, k) for k in range(1, steps + 1)), chunksize=4)
    return it, pool.terminate


def var_scale(c: dict, norm: dict) -> np.ndarray:
    """The variable scale in tensor units (G1: s_col, else x_std)."""
    return np.asarray(norm["s_col" if c["paths"]["G1"] else "x_std"], dtype=np.float64)


def scale_n(c: dict, norm: dict) -> np.ndarray:
    """The variable scale in G0-normalised units (G1: s_col / x_std, else 1)."""
    return var_scale(c, norm) / np.asarray(norm["x_std"], dtype=np.float64)


def fit(c: dict, out: str, device: str = None, data=None, max_minutes: float = None, max_steps: int = None,
        log=print, workers: int = 0) -> dict:
    """Train one config (module doc); returns {"history", "checkpoint", "config_sha256"}."""
    c = check_config(dict(c))
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(c["seed"])
    v2 = l2v2_on(c)
    data = data or TrainDataV2(c, log=log)
    sha_prep = _prep.sha256_file(os.path.join(data.prep.path, "manifest.json"))
    if c["prep_manifest_sha256"] not in (None, sha_prep):
        raise ValueError("config prep_manifest_sha256 %s != the prep manifest %s" % (c["prep_manifest_sha256"], sha_prep))
    c = dict(c, prep_manifest_sha256=sha_prep)
    os.makedirs(out, exist_ok=True)
    sha = config_sha256(c)
    stream, close = batches(data, batch_size(c), c["seed"], int(c["steps"]) if max_steps is None else
                            min(int(c["steps"]), int(max_steps)), workers)
    n_v2 = data.n_v2 if v2 else None
    model = _network.build_model(c, data.free, n_v2).to(device)
    ema = _network.build_model(c, data.free, n_v2).to(device)
    ema.load_state_dict(model.state_dict())
    ema.eval()
    for p_ in ema.parameters():
        p_.requires_grad_(False)
    n_params = sum(p_.numel() for p_ in model.parameters())
    log("fit: %s %s seed %d, %d params, device %s, config %s" % (path_name(c), c["window"], c["seed"], n_params,
                                                                 device, sha[:12]))
    opt = torch.optim.AdamW(model.parameters(), lr=c["lr"], weight_decay=c["weight_decay"])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(c))
    gen = torch.Generator(device=device).manual_seed(c["seed"])
    sc = torch.as_tensor(scale_n(c, data.norm), dtype=torch.float32, device=device)
    B = batch_size(c)
    steps = int(c["steps"]) if max_steps is None else min(int(c["steps"]), int(max_steps))
    history, t0, t_data = [], time.time(), 0.0
    run = {"loss": 0.0, "v": 0.0, "seam": 0.0}
    step = 0
    for step in range(1, steps + 1):
        model.train()
        td = time.time()
        b = to_torch(next(stream), device)
        t_data += time.time() - td
        loss, parts = training_loss(model, b, c, sc, gen)
        if not torch.isfinite(loss):
            raise SystemExit("fit: non-finite loss at step %d" % step)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), c["grad_clip"])
        opt.step()
        sched.step()
        dk = ema_decay(c, step - 1)
        with torch.no_grad():
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.mul_(dk).add_(pm.detach(), alpha=1 - dk)
        run["loss"] += float(loss.detach())
        run["v"] += float(parts["v"].detach())
        run["seam"] += float(parts["seam"].detach())
        stop = max_minutes is not None and time.time() - t0 > 60 * max_minutes
        if step % c["log_every"] == 0 or step == steps or stop:
            k = step % c["log_every"] or c["log_every"]
            rec = {"step": step, "loss": run["loss"] / k, "loss_v": run["v"] / k, "loss_seam": run["seam"] / k,
                   "lr": sched.get_last_lr()[0], "ema_decay": dk, "minutes": (time.time() - t0) / 60,
                   "data_minutes": t_data / 60}
            history.append(rec)
            log("fit: " + json.dumps(rec))
            run = {"loss": 0.0, "v": 0.0, "seam": 0.0}
        if stop:
            log("fit: --max_minutes reached at step %d" % step)
            break
    close()
    info = {"config_sha256": sha, "n_params": n_params, "steps_done": step, "minutes": (time.time() - t0) / 60,
            "data_minutes": t_data / 60, "device": device,
            "gpu": torch.cuda.get_device_name(0) if device.startswith("cuda") else None,
            "torch": torch.__version__, "numpy": np.__version__, "n_train_tracks": len(data.tracks),
            "path": path_name(c), "rest_null_resource": data.rest_null_resource}
    ck = os.path.join(out, "checkpoint.pt")
    ckd = {"ema": ema.state_dict(), "raw": model.state_dict(), "config": c, "info": info,
           "norm": data.norm, "free": data.free, "excluded": data.excluded, "rest_null": data.rest_null}
    if v2:
        ckd["l2v2_meta"] = data.meta(sha_prep)
    torch.save(ckd, ck)
    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump(c, f, indent=1, sort_keys=True)
    with open(os.path.join(out, "config_sha256.txt"), "w") as f:
        f.write(sha + "\n")
    with open(os.path.join(out, "history.json"), "w") as f:
        json.dump({"info": info, "history": history}, f, indent=1)
    log("fit: done, %d steps, %.1f min" % (step, info["minutes"]))
    return {"history": history, "checkpoint": ck, "config_sha256": sha, "info": info}
