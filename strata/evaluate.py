"""Naturalness metrics (Supp. D "FMD", "seam"): velocity FMD on unconstrained entries (30-frame windows, PCA
fitted on the training set), seam at the boundaries of the constraint mask, W1 velocity distance; per dataset and
density, seeds pooled.
"""
import glob
import hashlib
import json
import numpy as np
import os
import subprocess
import time
from strata import network as _network
from strata import representation as _representation
from strata import vocabulary as _vocabulary


def iou(a: dict, b: dict) -> float:
    inter = max(0, min(a["e"], b["e"]) - max(a["s"], b["s"]))
    return inter / float((a["e"] - a["s"]) + (b["e"] - b["s"]) - inter)


GATE_POOLED, GATE_FAMILY, GATE_FAMILY_MIN_N, D0_MIN = 95.0, 90.0, 20, 99.0


OK = "ok"


def edit_gate(summary: dict) -> dict:
    """summary: {"pooled_pct": x, "families": {name: {"n": k, "success_pct": y}}} -> {"pass": bool, "reasons": [...]}."""
    reasons = []
    if summary["pooled_pct"] is None or not summary["pooled_pct"] >= GATE_POOLED:
        reasons.append("pooled %s < %.1f" % (summary["pooled_pct"], GATE_POOLED))
    for name, f in sorted(summary["families"].items()):
        if f["n"] >= GATE_FAMILY_MIN_N and not f["success_pct"] >= GATE_FAMILY:
            reasons.append("%s %.2f %% < %.1f (n %d)" % (name, f["success_pct"], GATE_FAMILY, f["n"]))
    return {"pass": not reasons, "reasons": reasons}


def d0_sanity(summary: dict) -> bool:
    return summary["pooled_pct"] is not None and summary["pooled_pct"] >= D0_MIN


PRIMARY = "fmd_unc|all"  # uncovered FMD vel "all" (M1 standardised), design §5 item 2


TIE_BREAK = ("w1_headline", "static_gap_headline", "seam_headline|all")  # T0481 M2 order: W1 -> static -> seam


def primary_val(meta_val: list) -> list:
    """Indices of the VAL tracks whose content-text status is ok."""
    return [i for i, v in enumerate(meta_val) if v["status"] == OK]


W = os.environ.get("STRATA_RESEARCH_RUNS", "runs")  # research runs (default stats only)


DEFAULT_POLICY = "static_ref"  # Q022 (b) primary; "prior" = (a), a reported row (explicit --policy prior)


STATS = {"static_ref": (W + "/T0486/prep/stats_vel_static_ref.json",
                        "aeb9bbe74b02a0f6cb1007511a07454296b4fc709e115c0f5e2c9e6a51801d29"),
         "prior": (W + "/T0486/prep/stats_vel_prior.json",
                   "78d8615e77d979483f2b81c593d0f654c494d413572fdeaf5417eb8edec63083")}


RT_IOU_MIN = 0.5


ROWS = ("full", "standard", "lite")


DATASETS = ("talkvid", "ravdess", "crema-d")


PRIMARY_v2score = PRIMARY


def sha256_file(path: str) -> str:
    return sha256_file_bundle(path)


def _ds(e: dict) -> str:
    return e.get("dataset", "talkvid")


def _excluded(names: list) -> list:
    """gen.prep.excluded_columns without importing gen: the `_neutral` blendshape column (rules-excluded)."""
    return [k for k, n in enumerate(names) if n == "_neutral"]


def _text_columns(names: list) -> dict:
    return {f: _vocabulary.family_columns(f, names) for f in _vocabulary.VOCABULARY["families"]}


def build(prep: str, out: str, rows=ROWS, datasets=None, log=print) -> dict:
    """Write out/<row>/<dataset>/ bundles (module doc); returns {row: {dataset: dir}}."""
    t0 = time.time()
    man = json.load(open(os.path.join(prep, "manifest.json")))
    names = man["names"]
    prior_h = json.load(open(os.path.join(prep, man["prior_header"]["file"])))
    prior_ref = prior_h["reference"]
    datasets = datasets or sorted({_ds(e) for e in man["tracks"].values()}, key=lambda d: DATASETS.index(d)
                                  if d in DATASETS else 99)
    val_rows = man["val_rows"]
    tcols = _text_columns(names)
    excl = _excluded(names)
    manifest_sha = sha256_file(os.path.join(prep, "manifest.json"))
    out_dirs = {r: {} for r in rows}
    for ds in datasets:
        tr_ids = sorted(t for t, e in man["tracks"].items() if e["split"] == "train" and _ds(e) == ds)
        va_ids = sorted(t for t, e in man["tracks"].items() if e["split"] == "val" and _ds(e) == ds)
        T = {k: [] for k in ("truth_fold", "truth_bank", "truth_raw", "valid", "m")}
        tmeta = []
        for tid in tr_ids:
            e = man["tracks"][tid]
            with np.load(os.path.join(prep, e["file"])) as z:
                x, v = z["target_static_ref"].astype(np.float64), z["valid"].astype(bool)
            ref = json.load(open(os.path.join(prep, e["texts_file"])))["full"]["reference"]
            T["truth_raw"].append(x)
            T["truth_fold"].append(x)  # no folds in v2; static_ref scoring reads truth_raw
            T["truth_bank"].append(rereference(x, ref, prior_ref))
            T["valid"].append(v)
            T["m"].append(np.zeros(x.shape, dtype=bool))
            tmeta.append({"track_id": tid, "clip_id": e["clip_id"], "speaker": e["speaker"], "n_frames": len(x)})
        stats = compute_stats(list(zip(T["truth_raw"], T["valid"])), names,
                                 source={"tool": "fdeval.genmode.v2score", "truth": "TRAIN target_static_ref (raw)",
                                         "dataset": ds, "n_tracks": len(tr_ids), "prep": prep,
                                         "prep_manifest_sha256": manifest_sha}, excluded=excl)
        tz = {k: pack(a)[0] for k, a in T.items()}
        toff = pack(T["m"])[1]
        vals = {}
        for tid in va_ids:
            e = man["tracks"][tid]
            with np.load(os.path.join(prep, e["file"])) as z:
                vals[tid] = {k: z[k] for k in z.files if k.startswith(("target_", "valid", "rest_")) or
                             any(k.startswith("v%d_" % r) for r in range(len(val_rows)))}
        vmeta = []
        for tid in va_ids:
            e = man["tracks"][tid]
            vmeta.append({"track_id": tid, "clip_id": e["clip_id"], "speaker": e["speaker"], "n_frames": e["n_frames"],
                          "status": e["status"], "dataset": ds,
                          "reference": json.load(open(os.path.join(prep, e["texts_file"])))["full"]["reference"]})
        for row in rows:
            r = val_rows.index(row)
            V = {k: [] for k in ("truth", "truth_raw", "valid", "m", "x_obs", "x_obs_b")}
            R = {"rest": [], "rest_b": []}
            for tid in va_ids:
                a = vals[tid]
                V["truth_raw"].append(a["target_static_ref"].astype(np.float64))
                V["truth"].append(a["target_prior"].astype(np.float64))
                V["valid"].append(a["valid"].astype(bool))
                V["m"].append(a["v%d_M" % r].astype(bool))
                V["x_obs_b"].append(a["v%d_X_obs_static_ref" % r].astype(np.float64))
                V["x_obs"].append(a["v%d_X_obs_prior" % r].astype(np.float64))
                R["rest_b"].append(a["rest_static_ref"].astype(np.float64)[None])
                R["rest"].append(a["rest_prior"].astype(np.float64)[None])
            d = os.path.join(out, row, ds)
            os.makedirs(d, exist_ok=True)
            vz = {k: pack(x)[0] for k, x in V.items()}
            vz.update({k: np.concatenate(x) for k, x in R.items()})
            np.savez(os.path.join(d, "val.npz"), offsets=pack(V["m"])[1], **vz)
            np.savez(os.path.join(d, "train.npz"), offsets=toff, **tz)
            json.dump(stats, open(os.path.join(d, "stats_v2.json"), "w"), indent=1, sort_keys=True)
            meta = {"tool": "fdeval.genmode.v2score bundle", "commits": commits(), "prep": prep,
                    "prep_manifest_sha256": manifest_sha, "row": row, "row_index": r, "dataset": ds,
                    "run_dir": man.get("run_dir"), "names": names, "gaze_unit": man["splits"]["val"]["gaze_unit"],
                    "hz": man["hz"], "priors": {"rotation_reference": prior_ref},
                    "excluded_columns": excl, "text_columns": tcols, "val": vmeta, "train": tmeta,
                    "stats_v2_sha256": sha256_file(os.path.join(d, "stats_v2.json")),
                    "train_m": "all zero (module doc)", "runtime_s": time.time() - t0}
            json.dump(meta, open(os.path.join(d, "meta.json"), "w"), indent=1)
            out_dirs[row][ds] = d
            log("bundle %s / %s: %d VAL, %d TRAIN (%.0f s)" % (row, ds, len(va_ids), len(tr_ids), time.time() - t0))
    json.dump({"prep": prep, "prep_manifest_sha256": manifest_sha, "rows": list(rows), "datasets": list(datasets),
               "dirs": out_dirs}, open(os.path.join(out, "bundles.json"), "w"), indent=1)
    return out_dirs


def load_bundle(d: str):
    b = Bundle(d)
    return b, os.path.join(d, "stats_v2.json"), b.meta["stats_v2_sha256"]


def load_rows(path: str, bundle, any_mask: bool = False, frame: str = "gen") -> dict:
    """eval_genmode.load_row restricted to the bundle's tracks (a row dir holds every dataset's VAL tracks)."""
    out = {}
    for i, tid in enumerate(bundle.val_ids):
        f = os.path.join(path, tid.replace("#", "__seg") + ".npz")
        if not os.path.exists(f):
            hits = [g for g in glob.glob(os.path.join(path, "*.npz")) if str(np.load(g)["clip"]) == tid]
            if not hits:
                raise ValueError("%s: no output for %s" % (path, tid))
            f = hits[0]
        z = np.load(f)
        if str(z["clip"]) != tid:
            raise ValueError("%s: clip %s != %s" % (f, z["clip"], tid))
        x = np.asarray(z["X"], dtype=np.float64)
        x = x[None] if x.ndim == 2 else x
        m = bundle.val("m", i)
        if x.shape[1:] != m.shape:
            raise ValueError("%s: X %s, track has %s" % (f, x.shape, m.shape))
        if not any_mask and not np.array_equal(z["M"].astype(bool), m):
            raise ValueError("%s: M differs from the decode M" % f)
        if not np.isfinite(x).all():
            raise ValueError("%s: X not finite" % f)
        if frame in ("clip", "prior"):
            x = np.stack([to_frame(s, bundle, i, frame) for s in x])
        out[i] = x
    return out


def pooled(rowsets: list) -> dict:
    """Seeds pooled on the sample axis (T0495 pool_rows): {i: concat over the seeds}."""
    return {i: np.concatenate([r[i] for r in rowsets]) for i in rowsets[0]}


def eval_one(task: dict) -> dict:
    """One (window, dataset, row) eval of the round-1 rows -> eval.json + boot.npz in task['out']."""
    b, st, sha = load_bundle(task["bundle"])
    rows = {}
    for name, dirs in task["rows"].items():
        rs = [load_rows(os.path.join(d, task.get("text", "rows_asr"), task["row"]), b) for d in dirs]
        rows[name] = rs[0] if len(rs) == 1 else pooled(rs)
    evaluate(b, rows, task["out"], ref=task.get("ref", "G0"), policy="static_ref", stats_path=st, stats_sha=sha,
               n_boot=task["n_boot"], memorisation=task["memorisation"], save_boot=True)
    return {"out": task["out"], "eval_json_sha256": sha256_file(os.path.join(task["out"], "eval.json"))}


def _boot(d: str) -> dict:
    return dict(np.load(os.path.join(d, "boot.npz")))


KEYS = (PRIMARY_v2score,) + TIE_BREAK


def row_points(evals: dict, row: str, set_name: str = "ok99") -> dict:
    """{dataset: {key: point}} of one row (the reference of a relative macro)."""
    return {ds: {k: json.load(open(os.path.join(d, "eval.json")))["sets"][set_name]["rows"][row]["point"].get(k)
                 for k in KEYS} for ds, d in evals.items()}


def macro_candidates(evals: dict, cand_rows: dict, set_name: str = "ok99", ref: dict = None) -> tuple:
    """evals: {dataset: eval dir}; cand_rows: {candidate: row name in those evals}. -> (cands for rules.select,
    per-dataset point table). Point = mean over datasets; boot = per-draw mean over datasets (stratified).
    ref {dataset: {key: point}} (relative macro, user 2026-10-08, R3 A1.2): each dataset's point and draws of key k are
    divided by ref[dataset][k] first, so every dataset enters by its relative change (the reference -> 1)."""
    per_ds, cands = {}, {}
    ev = {ds: (json.load(open(os.path.join(d, "eval.json"))), _boot(d)) for ds, d in evals.items()}

    def rel(ds, k, x):
        if ref is None or x is None:
            return x
        r = ref[ds][k]
        return None if r is None or not np.isfinite(r) or r == 0 else x / r
    for c, row in cand_rows.items():
        pts, bts = {}, []
        for ds, (js, bt) in ev.items():
            p = js["sets"][set_name]["rows"][row]["point"]
            pts[ds] = {k: rel(ds, k, p.get(k)) for k in KEYS}
            b = bt["%s::%s::%s" % (set_name, row, PRIMARY_v2score)]
            bts.append(b if ref is None else b / ref[ds][PRIMARY_v2score])
        n = {len(b) for b in bts}
        if len(n) != 1:
            raise ValueError("unequal bootstrap sizes %s" % n)

        bad = [ds for ds in pts if pts[ds][PRIMARY_v2score] is None or not np.isfinite(pts[ds][PRIMARY_v2score])]
        if bad:  # R3 A1.4: never drop a candidate silently (rules.select drops a missing primary)
            raise ValueError("candidate %s (row %s): no finite %s on %s" % (c, row, PRIMARY_v2score, bad))

        def mean(k):
            v = [pts[ds][k] for ds in pts]
            return None if any(x is None or not np.isfinite(x) for x in v) else float(np.mean(v))
        cands[c] = {"point": {k: mean(k) for k in (PRIMARY_v2score,) + TIE_BREAK}, "edits": None,
                    "boot": np.mean(np.stack(bts), axis=0)}
        per_ds[c] = pts
    return cands, per_ds


def ref_vec(ref: dict) -> np.ndarray:
    return np.array([ref[_representation.ROTATION_TEXT][a] for a in _representation.ROTATION_AXES], dtype=np.float64)


def rereference(x: np.ndarray, ref_clip: dict, ref_prior: dict) -> np.ndarray:
    """(T, 63) with rot6d R_prior . R_ref,clip^-1 . R_true, composed as l1 decodes a rotation about its reference
    (rotation vector about the clip reference, re-expressed about the prior). Other columns unchanged."""
    out = np.asarray(x, dtype=np.float64).copy()
    rv = _representation.omega_from_r6(out[:, _network.ROT], ref_vec(ref_clip))
    out[:, _network.ROT] = _representation.r6_from_omega(rv, ref_vec(ref_prior))
    return out


def sha256_file_bundle(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def pack(arrs: list) -> tuple:
    off = np.concatenate([[0], np.cumsum([len(a) for a in arrs])]).astype(np.int64)
    return np.concatenate(arrs), off


class Bundle:
    """The saved bundle: per-track VAL / TRAIN arrays + meta."""

    def __init__(self, path: str):
        self.path = path
        self.meta = json.load(open(os.path.join(path, "meta.json")))
        self._v = dict(np.load(os.path.join(path, "val.npz")))
        self._t = dict(np.load(os.path.join(path, "train.npz")))
        self.names = self.meta["names"]
        self.val_ids = [v["track_id"] for v in self.meta["val"]]
        self.train_ids = [t["track_id"] for t in self.meta["train"]]
        self.excluded = tuple(self.meta["excluded_columns"])

    def val(self, key: str, i: int) -> np.ndarray:
        o = self._v["offsets"]
        return self._v[key][o[i]:o[i + 1]]

    def train(self, key: str, i: int) -> np.ndarray:
        o = self._t["offsets"]
        return self._t[key][o[i]:o[i + 1]]

    def rest(self, i: int, policy: str = "prior") -> np.ndarray:
        return self._v["rest" if policy == "prior" else "rest_b"][i]

    def x_obs(self, i: int, policy: str = "prior") -> np.ndarray:
        return self.val("x_obs" if policy == "prior" else "x_obs_b", i)

    def index(self, clip_or_track: str) -> int:
        for i, v in enumerate(self.meta["val"]):
            if clip_or_track in (v["track_id"], v["clip_id"]):
                return i
        raise KeyError(clip_or_track)


HEADLINE = ("fmd_unc|all", "w1_headline", "static_gap_headline", "seam_headline|all")


def sha256_file_eval_genmode(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def load_stats(bundle, path: str = None, sha: str = None, policy: str = DEFAULT_POLICY, drop: tuple = ()) -> tuple:
    """(stats, source): default (no path) the sha-pinned T0486 stats json of the header policy (rules.STATS; D242 /
    T0503), refused on a wrong sha256; an explicit path is checked against sha when given (and against the pin when it
    is a pinned file). drop (info runs only): stats recomputed with metrics.compute_stats (the 1e-3 kept-column rule) on
    the policy's TRAIN truth with these columns dropped as well. The old sigma < 1e-8 rule is never recomputed."""
    if drop:
        if path:
            raise ValueError("drop_columns: recomputed stats only (no --stats)")
        key = "truth_fold" if policy == "prior" else "truth_raw"
        n = len(bundle.train_ids)
        st = compute_stats([(bundle.train(key, i), bundle.train("valid", i)) for i in range(n)], bundle.names,
                              source={"truth": "%s TRAIN truth" % key, "extra_dropped": list(drop)}, drop=drop,
                              excluded=list(bundle.excluded))
        return st, {"path": None, "note": "INFO: stats recomputed (1e-3 rule, %s) with columns %s dropped" % (key, list(drop))}
    if path is None:
        path, sha = STATS[policy]
    pins = {p: h for p, h in STATS.values()}
    if os.path.abspath(path) in pins and sha is None:
        sha = pins[os.path.abspath(path)]
    got = sha256_file_eval_genmode(path)
    if sha and got != sha:
        raise ValueError("stats %s sha256 %s != %s" % (path, got, sha))
    st = json.load(open(path))
    if st.get("format") != STATS_FORMAT:
        raise ValueError("stats json is not %s (metrics.compute_stats schema)" % STATS_FORMAT)
    return st, {"path": path, "sha256": got}


def train_tracks(bundle, policy: str) -> list:
    key = "truth_fold" if policy == "prior" else "truth_raw"
    return [Track(bundle.train(key, i), bundle.train("valid", i), bundle.train("m", i),
                     np.full(bundle.train("m", i).shape, np.nan), 0) for i in range(len(bundle.train_ids))]


def val_tracks(bundle, idx: list, truth_ref: str, policy: str) -> list:
    spk_names = sorted({bundle.meta["val"][i]["speaker"] for i in idx})
    out = []
    for i in idx:
        v = bundle.meta["val"][i]
        truth = bundle.val("truth" if truth_ref == "train_median" and policy == "prior" else "truth_raw", i)
        out.append(Track(truth, bundle.val("valid", i), bundle.val("m", i), bundle.x_obs(i, policy),
                            spk_names.index(v["speaker"]), truth_abs=bundle.val("truth_raw", i), track_id=v["track_id"]))
    return out, spk_names


def to_frame(x: np.ndarray, bundle, i: int, frame: str) -> np.ndarray:
    """frame 'clip': an output in the clip's absolute frame -> the TRAIN-prior frame (R_prior . R_ref,clip^-1, m4);
    'prior': an output in the TRAIN-prior frame -> the clip frame (R_ref,clip . R_prior^-1, the inverse; T0502 M1)."""
    ref, pr = bundle.meta["val"][i]["reference"], bundle.meta["priors"]["rotation_reference"]
    return rereference(x, ref, pr) if frame == "clip" else rereference(x, pr, ref)


def make_ctx(bundle, stats: dict, fits: dict) -> dict:
    w, inv = column_weights(stats)
    groups = stats["groups"]
    gcol = group_of_columns(groups)
    tcols = bundle.meta["text_columns"]
    gch = {g: [ch for ch, cols in tcols.items() if gcol[cols[0]] == g] for g in GROUPS}
    return {"groups": groups, "w": w, "inv": inv, "edges": fits["edges"], "eps": fits["eps"], "text_cols": tcols,
            "group_channels": gch, "stats": stats}


class SetEval:
    """One eval set: tracks, speaker draws, FMD windows (shared by every row: paired)."""

    def __init__(self, tracks: list, n_spk: int, ctx: dict, fits: dict, n_boot: int):
        self.tracks, self.ctx, self.fits = tracks, ctx, fits
        self.n_spk = n_spk
        self.dr = draws(n_spk, n_boot)
        self.spk = np.array([t.spk for t in tracks])
        for t in tracks:
            t.seam = seam_edges(t, ctx["text_cols"])
        self.win = {}
        truth = [t.truth for t in tracks]
        for (kind, g), pca in fits["pca"].items():
            cols = fmd_columns(ctx["groups"], ctx["w"], g)
            k2 = "unc" if kind == "raw" else kind
            idx = window_index(tracks, cols, k2)
            scale = fmd_scale(kind, ctx["w"], fits["pos_inv"])
            fo = window_features(truth, tracks, idx, cols, k2, scale, fits["pos_mean"])
            self.win[(kind, g)] = {"idx": idx, "cols": cols, "k2": k2, "scale": scale, "orig": fo,
                                   "spk": np.array([tracks[k].spk for k, _ in idx], dtype=int)}
        self.segs = [memorisation_segments(t, ctx["groups"], ctx["w"]) for t in tracks]
        self.to_bank = None  # static_ref: per track (ref_clip, ref_prior), outputs moved into the bank's prior frame

    def fmd(self, ys: list, boot_kinds=("unc",), yw: list = None) -> tuple:
        """{kind|group: point}, {kind|group: bootstrap array}; ys[k] (S, T, 63) per set track; yw[k] (windowed row)
        [S x {start: (30, 63)}] replaces ys for the window features."""
        point, boot = {}, {}
        S = len(ys[0])
        src = ys if yw is None else yw
        if yw is not None:
            for k, (t, w) in enumerate(zip(self.tracks, yw)):
                need = set(window_starts(t.n))
                if len(w) != S or any(not need <= set(s) for s in w):
                    raise ValueError("windowed row: track %s lacks eval windows" % t.track_id)
        for (kind, g), w in self.win.items():
            if not w["idx"]:
                point["fmd_%s|%s" % (kind, g)] = None
                continue
            gens = [window_features([y[s] for y in src], self.tracks, w["idx"], w["cols"], w["k2"], w["scale"],
                                       self.fits["pos_mean"]) for s in range(S)]
            p, b = fmd_samples(gens, w["orig"], self.fits["pca"][(kind, g)], w["spk"],
                                  self.dr if kind in boot_kinds else None)
            point["fmd_%s|%s" % (kind, g)] = p
            if b is not None:
                boot["fmd_%s|%s" % (kind, g)] = b
        return point, boot

    def additive(self, ys: list) -> tuple:
        per = [track_stats(y, t, self.ctx) for y, t in zip(ys, self.tracks)]
        by = [add_stats([p for p, t in zip(per, self.tracks) if t.spk == s]) for s in range(self.n_spk)]
        point = {k: float(v) for k, v in summarise(add_stats(by), self.ctx).items()}
        boot = summarise(add_stats(by, self.dr), self.ctx)
        point["decmax"] = {g: max(p["decmax|%s" % g] for p in per) for g in GROUPS}
        point["n_edges"] = {g: int(sum(p["edges|%s" % g][0] for p in per)) for g in REPORT_GROUPS}
        point["n_edges_skipped"] = int(sum(p["edges|all"][1] for p in per))
        return point, boot

    def memorisation(self, ys: list, bank) -> dict:
        """Per group + pooled: median / p10 of the segment distances of the output (mean over samples per segment) and
        of the original, with speaker-bootstrap CIs (weighted percentiles)."""
        rows = {"out": [], "orig": []}
        for k, (t, segs) in enumerate(zip(self.tracks, self.segs)):
            if not segs:
                continue
            fr = (lambda x: x) if self.to_bank is None else (lambda x: rereference(x, *self.to_bank[k]))
            d_out = np.mean([[d for _, d in memorisation(fr(y), segs, bank, self.ctx["w"])] for y in ys[k]], axis=0)
            d_org = [d for _, d in memorisation(fr(t.truth), segs, bank, self.ctx["w"])]
            for (g, cols, s, e), a, b in zip(segs, d_out, d_org):
                rows["out"].append((g, t.spk, float(a), e - s))
                rows["orig"].append((g, t.spk, float(b), e - s))
        out = {"n_segments": len(rows["out"]), "min_frames": MEM_MIN, "stride": 1,
               "space": "standardised velocity (M1 w-scaled), RMS distance", "groups": {}}
        for g in GROUPS + ("all",):
            res = {}
            for key in ("out", "orig"):
                r = [x for x in rows[key] if g == "all" or x[0] == g]
                if not r:
                    continue
                d = np.array([x[2] for x in r])
                spk = np.array([x[1] for x in r])
                fin = np.isfinite(d)  # NaN: no fully valid TRAIN window of the segment's length (counted)
                n_none = int((~fin).sum())
                d, spk = d[fin], spk[fin]
                if not len(d):
                    res[key] = {"n": 0, "n_no_train_window": n_none}
                    continue
                res[key] = {"n": len(d), "n_no_train_window": n_none, "median": float(np.median(d)),
                            "p10": float(np.percentile(d, 10)),
                            "median_ci": ci([wpct(d, dd[spk], 50) for dd in self.dr]),
                            "p10_ci": ci([wpct(d, dd[spk], 10) for dd in self.dr])}
            out["groups"][g] = res
        return out


def wpct(x: np.ndarray, w: np.ndarray, q: float) -> float:
    """Weighted percentile (lower), integer weights (a bootstrap draw)."""
    if w.sum() == 0:
        return float("nan")
    o = np.argsort(x, kind="stable")
    c = np.cumsum(w[o])
    return float(x[o][np.searchsorted(c, q / 100.0 * c[-1], side="left").clip(0, len(x) - 1)])


def gate_input(summary: dict) -> dict:
    """The edit_gate input of an edit_summary: pooled + the gated families (jawOpen is reported, never gated)."""
    return {"pooled_pct": summary["pooled_pct"],
            "families": {k: v for k, v in summary["families"].items() if k != "jawOpen"}}


def edit_summary(per_track: list, idx: list, spk: np.ndarray, n_spk: int, dr: np.ndarray) -> dict:
    """Pooled + per-family edit execution over the set's tracks (per_track[i] = run_edits record of bundle track i)."""
    fams = ("eyeBlinkLeft", "mouthSmileLeft", "browInnerUp", "head_rotation", "jawOpen")
    sums = np.zeros((n_spk, len(fams), 6))
    keys = ("success_sum", "success_n", "off_channel_sum", "off_channel_n", "off_time_sum", "off_time_n")
    noop = {"bs": 0, "head": 0}
    for k, i in enumerate(idx):
        r = per_track[i]
        for f, fam in enumerate(fams):
            sums[spk[k], f] += [r[fam][q] for q in keys]
        for q in noop:
            noop[q] += r["noop"][q]

    def pct(a, b):
        return float(100 * a / b) if b > 0 else None
    tot = sums.sum(0)
    out = {"families": {}, "noop": noop}
    bt = np.tensordot(dr.astype(np.float64), sums, axes=([1], [0]))  # (B, fam, 6)
    for f, fam in enumerate(fams):
        out["families"][fam] = {"n": int(round(tot[f, 1])), "success_pct": pct(tot[f, 0], tot[f, 1]),
                                "success_ci": ci(100 * bt[:, f, 0] / np.maximum(bt[:, f, 1], 1e-12))
                                if tot[f, 1] else [None, None],
                                "off_channel_pct": pct(tot[f, 2], tot[f, 3]), "off_time_pct": pct(tot[f, 4], tot[f, 5])}
    s, n = tot[:, 0].sum(), tot[:, 1].sum()
    out["pooled_pct"] = pct(s, n)
    out["n"] = int(round(n))
    out["pooled_ci"] = ci(100 * bt[:, :, 0].sum(1) / np.maximum(bt[:, :, 1].sum(1), 1e-12))
    gate_in = gate_input(out)
    out["gate"] = edit_gate(gate_in)
    out["d0_sanity_99"] = d0_sanity(gate_in)
    return out


def _f(x, nd=4):
    return "-" if x is None or (isinstance(x, float) and not np.isfinite(x)) else ("%.*g" % (nd, x))


def _ci(c, nd=4):
    return "" if not c or c[0] is None else " [%s, %s]" % (_f(c[0], nd), _f(c[1], nd))


def to_md(res: dict) -> str:
    c = res["config"]
    L = ["# eval_genmode (truth_ref %s, header policy %s)" % (c["truth_ref"], c["policy"]), "",
         "Stats: %s. Speaker bootstrap N %d, seed %d. Definitions: fdeval/genmode/metrics.py." % (
             c["stats"].get("note") or c["stats"].get("path"), c["n_boot"], c["seed"]), ""]
    for set_name, s in res["sets"].items():
        L += ["## Set %s (%d tracks, %d speakers)" % (set_name, s["n_tracks"], s["n_speakers"]), "",
              "| row | S | FMD vel unc all (primary) | W1 headline | static gap pp | seam abs log ratio a (all) | "
              "cov MAE rot deg | cov MAE blink | cov MAE gaze | cov MAE brows | cov MAE mouth | edits pooled % |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for name, r in s["rows"].items():
            p, ci = r["point"], r["ci"]
            e = r.get("edits")
            L.append("| %s%s | %d | %s%s | %s%s | %s%s | %s%s | %s | %s | %s | %s | %s | %s |" % (
                name, " (FMD windowed)" if r.get("fmd_windowed") else "", r["n_samples"], _f(p.get("fmd_unc|all")), _ci(ci.get("fmd_unc|all")),
                _f(p["w1_headline"]), _ci(ci.get("w1_headline")), _f(p["static_gap_headline"], 3),
                _ci(ci.get("static_gap_headline"), 3), _f(p["seam_headline|all"], 3), _ci(ci.get("seam_headline|all"), 3),
                _f(p["mae_cov|rotation"]), _f(p["mae_cov|blink"]), _f(p["mae_cov|gaze"]), _f(p["mae_cov|brows"]),
                _f(p["mae_cov|mouth"]), "-" if not e else "%s (n %d)%s" % (_f(e["pooled_pct"], 4), e["n"], _ci(e["pooled_ci"]))))
        L += ["", "FMD windows (30 / stride 15, >= 50 %% uncovered entries) per group: %s; TRAIN PCA windows: %s" % (
            ", ".join("%s %d" % (k.split("|")[1], v) for k, v in s["n_windows"].items() if k.startswith("unc|")),
            ", ".join("%s %d" % (k.split("|")[1], v) for k, v in c["n_train_windows"].items() if k.startswith("unc|"))),
              "", "Per group (uncovered FMD vel standardised [CI]; W1 / TRAIN mean |v|; static gen / orig %; seam ratio "
              "a / j; covered MAE; uncovered MAE (info); all-frame MAE (info)):", ""]
        for name, r in s["rows"].items():
            p, ci = r["point"], r["ci"]
            L += ["### %s" % name, "", "| group | FMD unc | FMD all-frame (info) | W1 norm | W1 raw | static % gen / orig "
                  "| seam a / j (edges) | cov MAE | MAE vs decode mean / max | unc MAE | all MAE |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
            for g in REPORT_GROUPS:
                if g == "all":
                    L.append("| all | %s%s | %s | %s | - | - | %s / %s (%d) | - | - | - | - |" % (
                        _f(p.get("fmd_unc|all")), _ci(ci.get("fmd_unc|all")), _f(p.get("fmd_all|all")),
                        _f(p["w1_headline"]), _f(p["seam_ratio_a|all"], 3), _f(p["seam_ratio_j|all"], 3),
                        p["n_edges"]["all"]))
                    continue
                L.append("| %s | %s%s | %s | %s | %s | %s / %s | %s / %s (%d) | %s | %s / %s | %s | %s |" % (
                    g, _f(p.get("fmd_unc|%s" % g)), _ci(ci.get("fmd_unc|%s" % g)), _f(p.get("fmd_all|%s" % g)),
                    _f(p["w1_norm|%s" % g]), _f(p["w1_raw|%s" % g]), _f(p["static|%s" % g], 3),
                    _f(p["static_orig|%s" % g], 3), _f(p["seam_ratio_a|%s" % g], 3), _f(p["seam_ratio_j|%s" % g], 3),
                    p["n_edges"][g], _f(p["mae_cov|%s" % g]), _f(p["mae_vs_decode|%s" % g]), _f(p["decmax"][g]),
                    _f(p["mae_unc|%s" % g]), _f(p["mae_allf|%s" % g])))
            L += ["", "info: raw-unit FMD vel unc all %s; position FMD (M = 0 cells) %s; absolute rot MAE vs raw truth all / "
                  "covered %s / %s deg; edges skipped (track ends / invalid truth) %d" % (
                      _f(p.get("fmd_raw|all")), _f(p.get("fmd_pos|all")), _f(p.get("mae_abs|rotation")),
                      _f(p.get("mae_abs_cov|rotation")), p["n_edges_skipped"]), ""]
            for label in [k for k in r if k.startswith("minus_") or k == "retimed_minus_asr"]:
                d = r[label]
                L.append("%s: " % label + "; ".join("%s %+.4g%s" % (k, d[k]["diff"], _ci(d[k]["ci"]))
                                                    for k in HEADLINE if k in d))
                L.append("")
            if "memorisation" in r:
                m = r["memorisation"]
                L.append("memorisation (%d segments >= %d frames, stride 1, %s; median / p10, n, segments without a TRAIN "
                         "window of their length): " % (m["n_segments"], m["min_frames"], m["space"]) +
                         "; ".join("%s out %s / %s, orig %s / %s (n %d, none %d)" % (
                             g, _f(v["out"].get("median"), 3), _f(v["out"].get("p10"), 3), _f(v["orig"].get("median"), 3),
                             _f(v["orig"].get("p10"), 3), v["out"]["n"], v["out"]["n_no_train_window"])
                             for g, v in m["groups"].items() if "out" in v))
                L.append("")
            if "edits" in r:
                e = r["edits"]
                L.append("edits: pooled %s %% (n %d)%s; gate(95/90) %s %s; " % (
                    _f(e["pooled_pct"]), e["n"], _ci(e["pooled_ci"]), "PASS" if e["gate"]["pass"] else "FAIL",
                    e["gate"]["reasons"]) + "; ".join("%s %s %% (n %d)%s off-ch %s off-t %s" % (
                        k, _f(v["success_pct"]), v["n"], _ci(v["success_ci"]), _f(v["off_channel_pct"], 3),
                        _f(v["off_time_pct"], 3)) for k, v in e["families"].items()) + "; L1 no-ops %s" % e["noop"])
                L.append("")
    return "\n".join(L) + "\n"


BS, _network.TRANS, _network.ROT, GAZE = slice(0, 52), slice(52, 55), slice(55, 61), slice(61, 63)


D = 63


GROUPS = ("rotation", "blink", "gaze", "brows", "mouth", "translation", "other")


REPORT_GROUPS = GROUPS + ("all",)


SIGMA_MIN = 1e-8


SIGMA_KEEP = 1e-3  # = rules.SIGMA_KEEP (compute_stats kept-column rule)


WIN, STRIDE, KEEP = 30, 15, 0.5


N_COMP = 64


EPS_PCT = 5.0


EDGE_BINS = 1000


SEAM_BEFORE, SEAM_AFTER = 3, 2  # frames t-3 .. t+2 around an edge at t


MEM_MIN = 30


def weighting_groups(bs_names: list) -> dict:
    """M1's 7 weighting groups -> sorted 63-dim columns."""
    g = {k: [] for k in GROUPS}
    for k, n in enumerate(bs_names):
        if n.startswith(("jaw", "mouth", "cheek")):
            g["mouth"].append(k)
        elif n.startswith("eyeBlink"):
            g["blink"].append(k)
        elif n.startswith("brow"):
            g["brows"].append(k)
        elif n.startswith("eyeLook"):
            g["gaze"].append(k)
        else:
            g["other"].append(k)
    g["gaze"] += list(range(GAZE.start, GAZE.stop))
    g["rotation"] = list(range(_network.ROT.start, _network.ROT.stop))
    g["translation"] = list(range(_network.TRANS.start, _network.TRANS.stop))
    assert sorted(c for v in g.values() for c in v) == list(range(D))
    return {k: sorted(v) for k, v in g.items()}


def group_of_columns(groups: dict) -> np.ndarray:
    out = np.empty(D, dtype=object)
    for g, cols in groups.items():
        out[cols] = g
    return out


def r6_matrix(r6: np.ndarray) -> np.ndarray:
    """(..., 6) -> (..., 3, 3), Gram-Schmidt with the column order of FD headpose.rotation_6d_to_matrix (first column =
    r6[0:3], second = r6[3:6])."""
    r6 = np.asarray(r6, dtype=np.float64)
    a, b = r6[..., 0:3], r6[..., 3:6]
    c1 = a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-12)
    b = b - np.sum(c1 * b, axis=-1, keepdims=True) * c1
    c2 = b / np.maximum(np.linalg.norm(b, axis=-1, keepdims=True), 1e-12)
    return np.stack([c1, c2, np.cross(c1, c2)], axis=-1)


def geodesic_deg(r6a: np.ndarray, r6b: np.ndarray) -> np.ndarray:
    """Per-row geodesic angle (deg) between two (..., 6) rotation arrays (= FD geodesic_angle_error, vectorised)."""
    a, b = r6_matrix(r6a), r6_matrix(r6b)
    tr = np.einsum("...ij,...ij->...", a, b)
    return np.degrees(np.arccos(np.clip((tr - 1.0) / 2.0, -1.0, 1.0)))


def rot_speed(x: np.ndarray) -> np.ndarray:
    """(T, 63) -> (T-1,) geodesic deg/frame between consecutive frames."""
    return geodesic_deg(x[:-1, _network.ROT], x[1:, _network.ROT])


STATS_FORMAT = "genmode_velocity_stats"


def compute_stats(seqs: list, bs_names: list, source: dict = None, drop: tuple = (), excluded: list = None) -> dict:
    """seqs: [(x (T, 63) re-referenced TRAIN truth, valid (T, 63))]. sigma_c = std (ddof 0) of v[:, c] over the
    entries valid at t and t+1 (any M); per-group TRAIN mean |v| (rotation: geodesic deg/frame). Kept columns (amendment
    M20261005T123523, = T0486 gen.prep.velocity_stats; T0503): not rules-excluded AND sigma_natural_c >= SIGMA_KEEP
    (1e-3; sigma_natural = sigma, rot6d x 180 / pi = deg/frame). drop: columns dropped in addition (info runs only).
    The harness's default stats are the sha-pinned T0486 files (rules.STATS); this recompute is a fallback / info path."""
    groups = weighting_groups(bs_names)
    vs, ms, rs = [], [], []
    for x, valid in seqs:
        x = np.asarray(x, dtype=np.float64)
        vs.append(np.diff(x, axis=0))
        ms.append(valid[:-1] & valid[1:])
        rs.append(rot_speed(x)[ms[-1][:, _network.ROT.start]])
    v, m = np.concatenate(vs), np.concatenate(ms)
    sigma = [float(v[m[:, c], c].std()) if m[:, c].any() else 0.0 for c in range(D)]
    if excluded is None:  # the rules-excluded blendshapes (codec import only here)
        excluded = [k for k, n in enumerate(bs_names) if n in _representation.CHANNEL_RULES["channels"]["blendshape"]["exclude"]]
    natural = [s * 180.0 / np.pi if _network.ROT.start <= c < _network.ROT.stop else s for c, s in enumerate(sigma)]
    keep = [c not in excluded and natural[c] >= SIGMA_KEEP and c not in drop for c in range(D)]
    mean_abs = {}
    for g, cols in groups.items():
        if g == "rotation":
            mean_abs[g] = float(np.concatenate(rs).mean())
        else:
            kc = [c for c in cols if keep[c]]
            mean_abs[g] = float(np.mean(np.concatenate([np.abs(v[m[:, c], c]) for c in kc])))
    return {"format": STATS_FORMAT, "version": 1, "groups": groups,
            "sigma": [s if k else None for s, k in zip(sigma, keep)],
            "dropped_columns": [c for c in range(D) if not keep[c]],
            "n_g": {g: sum(keep[c] for c in cols) for g, cols in groups.items()},
            "group_mean_abs_v": mean_abs, "rotation_mean_abs_v_unit": "geodesic deg/frame",
            "n_entries": [int(m[:, c].sum()) for c in range(D)], "n_tracks": len(seqs), "source": source or {}}


def column_weights(stats: dict) -> tuple:
    """(w (63,) = 1 / (sigma_c sqrt(n_g)) or 0 for a dropped column, inv_sigma (63,) or 0)."""
    w, inv = np.zeros(D), np.zeros(D)
    for g, cols in stats["groups"].items():
        for c in cols:
            s = stats["sigma"][c]
            if s is not None and s >= SIGMA_MIN:
                inv[c] = 1.0 / s
                w[c] = inv[c] / np.sqrt(stats["n_g"][g])
    return w, inv


def unc_entries(m: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """(T-1, 63) M2: a velocity entry touches a generated cell, truth valid at both ends."""
    return (~m[:-1] | ~m[1:]) & valid[:-1] & valid[1:]


def all_entries(m: np.ndarray, valid: np.ndarray) -> np.ndarray:
    return valid[:-1] & valid[1:]


def window_starts(n: int) -> range:
    return range(0, n - WIN + 1, STRIDE)


class Track:
    """One track's eval-side arrays (no output). y-independent masks are built once."""

    def __init__(self, truth, valid, m, x_obs, spk: int, truth_abs=None, track_id: str = None):
        self.truth = np.asarray(truth, dtype=np.float64)
        self.valid = np.asarray(valid, dtype=bool)
        self.m = np.asarray(m, dtype=bool)
        self.x_obs = np.asarray(x_obs, dtype=np.float64)
        self.spk, self.track_id = int(spk), track_id
        self.truth_abs = None if truth_abs is None else np.asarray(truth_abs, dtype=np.float64)
        self.n = len(self.truth)
        assert self.truth.shape == self.valid.shape == self.m.shape == self.x_obs.shape == (self.n, D)
        self.unc = unc_entries(self.m, self.valid)
        self.allv = all_entries(self.m, self.valid)


def window_index(tracks: list, cols: list, kind: str) -> list:
    """[(track k, start)] of the kept windows of a column set: kind 'unc' (>= 50 % uncovered entries), 'all' (every
    window with >= 1 valid entry), 'pos' (>= 50 % of the window's cells are M = 0 and valid)."""
    out = []
    cols = np.asarray(cols)
    for k, t in enumerate(tracks):
        if kind == "unc":
            e, need = t.unc[:, cols], KEEP * (WIN - 1) * len(cols)
        elif kind == "all":
            e, need = t.allv[:, cols], 1
        elif kind == "pos":
            e, need = (~t.m & t.valid)[:, cols], KEEP * WIN * len(cols)
        else:
            raise ValueError(kind)
        span = WIN if kind == "pos" else WIN - 1
        c = np.concatenate([[0], np.cumsum(e.sum(axis=1))])
        out += [(k, s) for s in window_starts(t.n) if c[s + span] - c[s] >= need]
    return out


def window_features(xs: list, tracks: list, index: list, cols: list, kind: str, scale: np.ndarray,
                    pos_mean: np.ndarray = None) -> np.ndarray:
    """(N, F) features of the windows `index` of the series xs[k] (T_k, 63) (truth or an output sample).
    kind 'unc' / 'all': velocity * scale masked by the track's entries; 'pos': (x - pos_mean) * scale on M = 0 valid.
    xs[k] may be a dict {start: (WIN, 63)} (a windowed row: each eval window read from its own source window)."""
    cols = np.asarray(cols)
    out = []
    cache = {}
    for k, s in index:
        if isinstance(xs[k], dict):
            t, x = tracks[k], np.asarray(xs[k][s], dtype=np.float64)
            if kind == "pos":
                c = (~t.m & t.valid)[s:s + WIN]
                f = np.where(c, (x - pos_mean) * scale, 0.0)
            else:
                e = (t.unc if kind == "unc" else t.allv)[s:s + WIN - 1]
                f = np.where(e, np.diff(x, axis=0) * scale, 0.0)
            out.append(f[:, cols].ravel())
            continue
        if k not in cache:
            t, x = tracks[k], xs[k]
            if kind == "pos":
                cache[k] = np.where(~t.m & t.valid, (x - pos_mean) * scale, 0.0)[:, cols]
            else:
                e = t.unc if kind == "unc" else t.allv
                cache[k] = np.where(e, np.diff(x, axis=0) * scale, 0.0)[:, cols]
        span = WIN if kind == "pos" else WIN - 1
        out.append(cache[k][s:s + span].ravel())
    return np.stack(out) if out else np.zeros((0, len(cols) * (WIN if kind == "pos" else WIN - 1)))


FMD_KINDS = ("unc", "all", "raw", "pos")  # raw / pos: group "all" only (info)


def fmd_columns(groups: dict, w: np.ndarray, g: str) -> list:
    cols = list(range(D)) if g == "all" else groups[g]
    return [c for c in cols if w[c] > 0]


def fmd_scale(kind: str, w: np.ndarray, pos_inv: np.ndarray = None) -> np.ndarray:
    if kind in ("unc", "all"):
        return w
    if kind == "raw":
        return (w > 0).astype(np.float64)
    return pos_inv


def fit_train(train_tracks: list, stats: dict) -> dict:
    """PCA per (kind, group), eps per group and W1 edges per group from TRAIN Track objects (truth = re-referenced
    TRAIN truth, m = the TRAIN track's own L2 coverage mask)."""
    groups = stats["groups"]
    w, inv = column_weights(stats)
    xs = [t.truth for t in train_tracks]
    allx = np.concatenate([t.truth[t.valid.all(1)] for t in train_tracks])
    pos_mean, pos_std = allx.mean(0), allx.std(0)
    pos_inv = np.where((pos_std > SIGMA_MIN) & (w > 0), 1.0 / np.maximum(pos_std, SIGMA_MIN), 0.0)
    out = {"pca": {}, "n_train_windows": {}, "pos_mean": pos_mean, "pos_inv": pos_inv}
    for kind in FMD_KINDS:
        for g in (REPORT_GROUPS if kind in ("unc", "all") else ("all",)):
            cols = fmd_columns(groups, w, g)
            idx = window_index(train_tracks, cols, "unc" if kind == "raw" else kind)
            feats = window_features(xs, train_tracks, idx, cols, "unc" if kind == "raw" else kind,
                                    fmd_scale(kind, w, pos_inv), pos_mean)
            out["pca"][(kind, g)] = PCA(feats, min(N_COMP, feats.shape[1]))
            out["n_train_windows"]["%s|%s" % (kind, g)] = len(idx)
    out["eps"], out["edges"] = {}, {}
    for g in GROUPS:
        sp = np.concatenate([group_speed(t.truth, t, groups[g], inv, g, "unc")[1] for t in train_tracks])
        out["eps"][g] = float(np.percentile(sp, EPS_PCT))
        vals = np.concatenate([abs_vel(t.truth, t, groups[g], w, g, "all") for t in train_tracks])
        lo = max(1e-2 * float(np.median(vals)), 1e-9)
        hi = max(1e2 * float(np.percentile(vals, 99.9)), 10 * lo)
        out["edges"][g] = np.concatenate([[0.0], np.geomspace(lo, hi, EDGE_BINS)])
    return out


def abs_vel(x: np.ndarray, t: Track, cols: list, w: np.ndarray, g: str, which: str = "unc") -> np.ndarray:
    """|v| samples of a group on the track's uncovered ('unc') or valid ('all') entries (rotation: geodesic deg)."""
    e = t.unc if which == "unc" else t.allv
    if g == "rotation":
        return rot_speed(x)[e[:, _network.ROT.start]]
    kc = [c for c in cols if w[c] > 0]
    return np.abs(np.diff(x[:, kc], axis=0))[e[:, kc]]


def group_speed(x: np.ndarray, t: Track, cols: list, inv: np.ndarray, g: str, which: str = "unc") -> tuple:
    """(frames, speed) of the velocity frames with >= 1 entry of the group: rotation geodesic deg/frame, else the RMS
    of v / sigma over those entries."""
    e = t.unc if which == "unc" else t.allv
    if g == "rotation":
        f = np.flatnonzero(e[:, _network.ROT.start])
        return f, rot_speed(x)[f]
    kc = [c for c in cols if inv[c] > 0]
    ee = e[:, kc]
    f = np.flatnonzero(ee.any(1))
    z = np.where(ee, np.diff(x[:, kc], axis=0) * inv[kc], 0.0)[f]
    return f, np.sqrt((z ** 2).sum(1) / ee[f].sum(1))


def seam_edges(t: Track, text_cols: dict) -> dict:
    """{text channel: [edge frames t kept]} and the number skipped, from M and truth validity only."""
    kept, skipped = {}, 0
    for ch, cols in text_cols.items():
        mc = t.m[:, cols[0]]
        edges = np.flatnonzero(mc[1:] != mc[:-1]) + 1
        ok = []
        for e in edges:
            if e - SEAM_BEFORE - 1 >= 0 and e + SEAM_AFTER + 2 <= t.n - 1 and \
                    t.valid[e - SEAM_BEFORE - 1:e + SEAM_AFTER + 3][:, cols].all():
                ok.append(int(e))
            else:
                skipped += 1
        kept[ch] = ok
    return {"edges": kept, "skipped": skipped}


def seam_values(x: np.ndarray, edges: list, cols: list, inv: np.ndarray) -> tuple:
    """(sum |a|, sum |j|, count) over the edges' frames k = t-3 .. t+2 (RMS over the channel's columns of the second /
    third difference / sigma_c)."""
    if not edges or not (inv[cols] > 0).any():
        return 0.0, 0.0, 0
    kc = [c for c in cols if inv[c] > 0]
    z = x[:, kc] * inv[kc]
    sa = sj = 0.0
    n = 0
    for e in edges:
        k = np.arange(e - SEAM_BEFORE, e + SEAM_AFTER + 1)
        a = z[k + 1] - 2 * z[k] + z[k - 1]
        j = z[k + 2] - 3 * z[k + 1] + 3 * z[k] - z[k - 1]
        sa += float(np.sqrt((a ** 2).mean(1)).sum())
        sj += float(np.sqrt((j ** 2).mean(1)).sum())
        n += len(k)
    return sa, sj, n


def cell_err(y: np.ndarray, ref: np.ndarray, cells: np.ndarray, cols: list, g: str) -> tuple:
    """(sum, count) of |y - ref| over the cells (rotation: geodesic deg per frame on frames whose rot cells are set)."""
    if g == "rotation":
        f = cells[:, _network.ROT.start]
        return float(geodesic_deg(y[f, _network.ROT], ref[f, _network.ROT]).sum()), int(f.sum())
    cc = cells[:, cols]
    return float(np.abs(y[:, cols] - ref[:, cols])[cc].sum()), int(cc.sum())


def track_stats(ys: np.ndarray, t: Track, ctx: dict) -> dict:
    """Additive statistics of one track, averaged over the S samples of ys (S, T, 63). ctx: groups, w, inv, edges,
    eps, text_cols, seam (seam_edges of the track)."""
    groups, w, inv = ctx["groups"], ctx["w"], ctx["inv"]
    S = len(ys)
    out = {}

    def acc(key, val):
        out[key] = out.get(key, 0.0) + np.asarray(val, dtype=np.float64) / S
    for y in ys:
        for g in GROUPS:  # MAE family per group (units differ between groups: no "all")
            cols = groups[g]
            for name, cells in (("cov", t.m & t.valid), ("unc", ~t.m & t.valid), ("allf", t.valid)):
                acc("mae_%s|%s" % (name, g), cell_err(y, t.truth, cells, cols, g))
            cov = t.m[:, cols]
            d = np.abs(np.where(cov, y[:, cols] - np.where(cov, t.x_obs[:, cols], 0.0), 0.0))
            acc("dec|%s" % g, [d.sum(), cov.sum()])
            out["decmax|%s" % g] = max(out.get("decmax|%s" % g, 0.0), float(d.max()) if d.size else 0.0)
        if t.truth_abs is not None:
            s, n = cell_err(y, t.truth_abs, t.valid, groups["rotation"], "rotation")
            acc("mae_abs|rotation", [s, n])
            s, n = cell_err(y, t.truth_abs, t.m & t.valid, groups["rotation"], "rotation")
            acc("mae_abs_cov|rotation", [s, n])
        for g in GROUPS:
            cols = groups[g]
            hy = histogram(abs_vel(y, t, cols, w, g), ctx["edges"][g])
            acc("hw1|%s" % g, hy)
            f, sp = group_speed(y, t, cols, inv, g)
            acc("static|%s" % g, [(sp < ctx["eps"][g]).sum(), len(f)])
        sa_all = [0.0, 0.0, 0]
        for g in GROUPS:
            sa, sj, n = 0.0, 0.0, 0
            for ch in ctx["group_channels"][g]:
                a, j, k = seam_values(y, t.seam["edges"][ch], ctx["text_cols"][ch], inv)
                sa, sj, n = sa + a, sj + j, n + k
            acc("seam|%s" % g, [sa, sj, n])
            sa_all = [sa_all[0] + sa, sa_all[1] + sj, sa_all[2] + n]
        acc("seam|all", sa_all)
    # the original at the same positions (sample-independent; added once)
    for g in GROUPS:
        cols = groups[g]
        out["hw1_o|%s" % g] = histogram(abs_vel(t.truth, t, cols, w, g), ctx["edges"][g])
        f, sp = group_speed(t.truth, t, cols, inv, g)
        out["static_o|%s" % g] = np.array([(sp < ctx["eps"][g]).sum(), len(f)], dtype=np.float64)
    tot = np.zeros(3)
    for g in GROUPS:
        v = np.zeros(3)
        for ch in ctx["group_channels"][g]:
            v += seam_values(t.truth, t.seam["edges"][ch], ctx["text_cols"][ch], inv)
        out["seam_o|%s" % g] = v
        tot += v
    out["seam_o|all"] = tot
    out["edges|all"] = np.array([sum(len(v) for v in t.seam["edges"].values()), t.seam["skipped"]], dtype=np.float64)
    for g in GROUPS:
        out["edges|%s" % g] = np.array([sum(len(t.seam["edges"][ch]) for ch in ctx["group_channels"][g]), 0.0])
    return out


def _r(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(b > 0, a / np.where(b > 0, b, 1.0), np.nan)


def summarise(s: dict, ctx: dict) -> dict:
    """Metric values from summed track_stats (leading batch dims allowed: bootstrap)."""
    out = {}
    for g in GROUPS:
        for name in ("cov", "unc", "allf"):
            v = s["mae_%s|%s" % (name, g)]
            out["mae_%s|%s" % (name, g)] = _r(v[..., 0], v[..., 1])
        v = s["dec|%s" % g]
        out["mae_vs_decode|%s" % g] = _r(v[..., 0], v[..., 1])
    for k in ("mae_abs|rotation", "mae_abs_cov|rotation"):
        if k in s:
            out[k] = _r(s[k][..., 0], s[k][..., 1])
    w1n, sg = [], []
    for g in GROUPS:
        w1 = w1_hist(s["hw1|%s" % g], s["hw1_o|%s" % g], ctx["edges"][g])
        out["w1_raw|%s" % g] = w1
        out["w1_norm|%s" % g] = w1 / ctx["stats"]["group_mean_abs_v"][g]
        w1n.append(out["w1_norm|%s" % g])
        st, so = s["static|%s" % g], s["static_o|%s" % g]
        out["static|%s" % g] = 100 * _r(st[..., 0], st[..., 1])
        out["static_orig|%s" % g] = 100 * _r(so[..., 0], so[..., 1])
        out["static_gap|%s" % g] = np.abs(out["static|%s" % g] - out["static_orig|%s" % g])
        sg.append(out["static_gap|%s" % g])
    out["w1_headline"] = np.mean(w1n, axis=0)
    out["static_gap_headline"] = np.nanmean(sg, axis=0)
    for g in REPORT_GROUPS:
        a, o = s["seam|%s" % g], s["seam_o|%s" % g]
        ra = _r(_r(a[..., 0], a[..., 2]), _r(o[..., 0], o[..., 2]))
        rj = _r(_r(a[..., 1], a[..., 2]), _r(o[..., 1], o[..., 2]))
        out["seam_ratio_a|%s" % g], out["seam_ratio_j|%s" % g] = ra, rj
        out["seam_headline|%s" % g] = np.abs(np.log(ra))
    return out


def add_stats(per_track: list, weights: np.ndarray = None) -> dict:
    """Sum of track_stats dicts, weights (n_tracks,) or (B, n_tracks) -> a leading batch dim; decmax is a max."""
    keys = [k for k in per_track[0] if not k.startswith("decmax")]
    if weights is None:
        weights = np.ones(len(per_track))
    return {k: np.tensordot(np.asarray(weights, dtype=np.float64), np.stack([p[k] for p in per_track]), axes=([-1], [0]))
            for k in keys}


def frechet(mu1, s1, mu2, s2) -> float:
    """The Frechet distance of diffusion/fmd.frechet, |mu1 - mu2|^2 + tr(S1 + S2 - 2 (S1 S2)^(1/2)), with the trace term
    computed stably as sum sqrt(eig(S1^(1/2) S2 S1^(1/2))) (symmetric PSD; eigenvalues clipped at 0). Equal to
    fmd.frechet on regular covariances (unit test); does not break down on a singular one (e.g. a constant output,
    whose covariance is 0, where scipy.linalg.sqrtm returns non-finite or wrong values)."""
    w1, v1 = np.linalg.eigh((s1 + s1.T) / 2)
    r1 = (v1 * np.sqrt(np.clip(w1, 0, None))) @ v1.T
    a = r1 @ s2 @ r1
    tr = float(np.sqrt(np.clip(np.linalg.eigvalsh((a + a.T) / 2), 0, None)).sum())
    d = mu1 - mu2
    return float(d @ d + np.trace(s1) + np.trace(s2) - 2.0 * tr)


def fmd_samples(gen_feats: list, orig_feats: np.ndarray, pca, spk_w: np.ndarray = None, dr: np.ndarray = None) -> tuple:
    """(point FMD = mean over samples, bootstrap (n_boot,) or None). gen_feats: [S x (N, F)] on the same windows."""
    xo = pca(orig_feats)
    xs = [pca(x) for x in gen_feats]
    mo = gaussian(xo)
    point = float(np.mean([frechet(*gaussian(x), *mo) for x in xs]))
    if dr is None:
        return point, None
    boot = np.empty(len(dr))
    for i, d in enumerate(dr):
        wgt = d[spk_w]
        go = gaussian(xo, wgt)
        boot[i] = np.mean([frechet(*gaussian(x, wgt), *go) for x in xs])
    return point, boot


def ci(v) -> list:
    v = np.asarray(v, dtype=np.float64)
    if not np.isfinite(v).any():
        return [None, None]
    return [float(x) for x in np.nanpercentile(v, [2.5, 97.5])]


def mask_blocks(m: np.ndarray, cols: list) -> list:
    """Columns of `cols` grouped by an identical M column over the track (sorted, deterministic)."""
    by = {}
    for c in cols:
        by.setdefault(m[:, c].tobytes(), []).append(c)
    return sorted(by.values())


def runs(flag: np.ndarray) -> list:
    """[(s, e)] maximal runs of True."""
    f = np.concatenate([[False], np.asarray(flag, dtype=bool), [False]])
    d = np.flatnonzero(f[1:] != f[:-1])
    return list(zip(d[0::2], d[1::2]))


class VelBank:
    """TRAIN standardised velocities (w-scaled) of the bank tracks, concatenated with one invalid separator row per
    track; windows with an invalid entry or crossing a track never match."""

    def __init__(self, bank_x: list, bank_valid: list, w: np.ndarray):
        vs, ok = [], []
        for x, v in zip(bank_x, bank_valid):
            vs.append(np.diff(np.asarray(x, dtype=np.float64), axis=0) * w)
            ok.append(v[:-1] & v[1:])
            vs.append(np.zeros((1, D)))
            ok.append(np.zeros((1, D), dtype=bool))
        self.v, self.ok = np.concatenate(vs), np.concatenate(ok)
        self.n_frames = int(sum(len(x) for x in bank_x))
        self.n_tracks = len(bank_x)

    def nearest(self, q: np.ndarray, cols: list) -> float:
        """RMS distance sqrt(||q - b||^2 / q.size) of q (n, |cols|) to its nearest valid bank window, stride 1."""
        from scipy.signal import fftconvolve
        n = len(q)
        b = self.v[:, cols]
        bad = np.concatenate([[0], np.cumsum(~self.ok[:, cols].all(1))])
        nwin = len(b) - n + 1
        if nwin <= 0:
            return float("nan")
        good = (bad[n:n + nwin] - bad[:nwin]) == 0
        if not good.any():
            return float("nan")
        bb = np.concatenate([[0], np.cumsum((b ** 2).sum(1))])
        sb = bb[n:n + nwin] - bb[:nwin]
        cross = fftconvolve(b, q[::-1], mode="valid", axes=0).sum(1)
        d2 = (q ** 2).sum() + sb - 2 * cross
        d2 = np.where(good, d2, np.inf)
        return float(np.sqrt(max(d2.min(), 0.0) / q.size))


def memorisation_segments(t: Track, groups: dict, w: np.ndarray) -> list:
    """[(group, cols, s, e)]: per group mask block, the maximal runs of M = 0 AND truth valid of >= MEM_MIN frames
    (fixed by M and truth validity: the same segments for every row)."""
    out = []
    for g in GROUPS:
        cols = [c for c in groups[g] if w[c] > 0]
        for block in mask_blocks(t.m, cols):
            flag = ~t.m[:, block[0]] & t.valid[:, block].all(1)
            out += [(g, block, int(s), int(e)) for s, e in runs(flag) if e - s >= MEM_MIN]
    return out


def memorisation(x: np.ndarray, segs: list, bank: VelBank, w: np.ndarray) -> list:
    """[(group, distance)] of the segments' standardised velocities to their nearest TRAIN window."""
    out = []
    for g, cols, s, e in segs:
        q = np.diff(x[s:e][:, cols], axis=0) * w[cols]
        out.append((g, bank.nearest(q, cols)))
    return out


N_BOOT_uncovered, SEED = 1000, 0


def histogram(vals: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Counts in the len(edges) - 1 bins [e_k, e_k+1); values >= the last edge go to the last bin."""
    idx = np.clip(np.searchsorted(edges, vals, side="right") - 1, 0, len(edges) - 2)
    return np.bincount(idx, minlength=len(edges) - 1).astype(np.float64)


def w1_hist(hx: np.ndarray, ho: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """W1 from histograms on the same edges (module docstring); leading batch dims allowed."""
    def cdf_mid(h):
        c = np.cumsum(h, axis=-1) / np.maximum(h.sum(axis=-1, keepdims=True), 1e-300)
        return 0.5 * (c + np.concatenate([np.zeros(c.shape[:-1] + (1,)), c[..., :-1]], axis=-1))
    return np.sum(np.abs(cdf_mid(hx) - cdf_mid(ho)) * np.diff(edges), axis=-1)


def git_state(path: str) -> dict:
    """{path, sha, dirty, untracked}; dirty = tracked changes, untracked = number of untracked, not ignored files
    (reported separately, T0044); sha None when path is not a git checkout."""
    def git(*args):
        return subprocess.run(["git", "-C", path, *args], capture_output=True, text=True, check=True).stdout.strip()
    try:
        return {"path": path, "sha": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
                "untracked": len(git("ls-files", "--others", "--exclude-standard").splitlines())}
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"path": path, "sha": None, "dirty": None, "untracked": None}


def commits() -> dict:
    return {"facial-dynamics": git_state(FD_REPO), "detail-net": git_state(DN_REPO),
            "l3": git_state(L3_REPO)}


HERE = os.path.dirname(os.path.abspath(__file__))


def _sandbox_root() -> str:
    """The first ancestor holding both sibling repos (works from R/l3 and from R/wt/<ID>/l3)."""
    path = os.path.dirname(HERE)
    while path != os.path.dirname(path):
        if os.path.isdir(os.path.join(path, "facial-dynamics")) and os.path.isdir(os.path.join(path, "detail-net")):
            return path
        path = os.path.dirname(path)
    return os.path.dirname(os.path.dirname(HERE))


ROOT = os.path.abspath(os.environ.get("FDEVAL_ROOT", _sandbox_root()))


FD_REPO = os.path.abspath(os.environ.get("FD_REPO", os.path.join(ROOT, "facial-dynamics")))


DN_REPO = os.path.abspath(os.environ.get("DN_REPO", os.path.join(ROOT, "detail-net")))


L3_REPO = os.path.dirname(HERE)


N_COMP, EPS = 64, 1e-6


class PCA:
    """Top-n principal axes of the TRAIN features (centred; no whitening)."""

    def __init__(self, x: np.ndarray, n: int = N_COMP):
        self.mean = x.mean(axis=0)
        _, s, vt = np.linalg.svd(x - self.mean, full_matrices=False)
        self.components = vt[:n]
        var = s ** 2
        self.explained = float(var[:n].sum() / var.sum())

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) @ self.components.T


def gaussian(x: np.ndarray, w: np.ndarray = None) -> tuple:
    """Mean and covariance (ddof 1); w: integer frequency weights per row (a speaker-bootstrap draw)."""
    if w is None:
        return x.mean(axis=0), np.cov(x, rowvar=False)
    w = np.asarray(w, dtype=np.float64)
    sw = w.sum()
    mu = w @ x / sw
    xc = x - mu
    return mu, (xc * w[:, None]).T @ xc / (sw - 1.0)


def described_cols(track: dict, names: list) -> np.ndarray:
    """(n, 63) bool: l1.describable of the column's channel."""
    n = track["features"].shape[0]
    d = _representation.tracked_valid(track["observed"], track["quality"], track["is_gap_filled"], np.zeros(n, dtype=bool))
    out = np.zeros((n, track["features"].shape[1]), dtype=bool)
    for k, spec in enumerate(_representation.CHANNELS):
        out[:, spec.slice] = d[:, k:k + 1]
    return out


def match(pred: list, true: list, iou_min: float = RT_IOU_MIN) -> list:
    """[(i pred, j true)]: greedy one-to-one, pairs with IoU >= iou_min by IoU descending (ties: true, then pred index)."""
    cand = sorted(((iou(p, t), j, i) for i, p in enumerate(pred) for j, t in enumerate(true) if iou(p, t) >= iou_min),
                  key=lambda c: (-c[0], c[1], c[2]))
    used_p, used_t, out = set(), set(), []
    for _, j, i in cand:
        if i not in used_p and j not in used_t:
            used_p.add(i), used_t.add(j)
            out.append((i, j))
    return sorted(out, key=lambda m: m[1])


N_BOOT, SEED = N_BOOT_uncovered, SEED


def draws(n_speakers: int, n: int = N_BOOT, seed: int = SEED) -> np.ndarray:
    return np.random.default_rng(seed).multinomial(n_speakers, np.full(n_speakers, 1.0 / n_speakers), size=n)


def evaluate(bundle, rows: dict, out_dir: str, ref: str = None, edits: dict = None, truth_ref: str = None,
             policy: str = DEFAULT_POLICY, stats_path: str = None, stats_sha: str = None, n_boot: int = N_BOOT,
             memorisation: bool = True, retimed: dict = None, windows: dict = None, drop: tuple = (),
             save_boot: bool = False) -> dict:
    """rows: {name: {bundle index: (S, T, 63)}}; edits: {name: per-track run_edits list (bundle order)}; windows:
    {name: load_windows(...)} of the windowed rows. truth_ref None: the policy's truth (static_ref: 'clip' = raw,
    prior: 'train_median'). save_boot: also DIR/boot.npz {"SET::ROW::KEY": per-draw values, "SET::draws": the speaker
    draws} (the paired arrays `select` needs; off by default, eval.json / eval.md unchanged)."""
    t0 = time.time()
    if truth_ref is None:
        truth_ref = "clip" if policy == "static_ref" else "train_median"
    stats, stats_src = load_stats(bundle, stats_path, stats_sha, policy, drop)
    fits = fit_train(train_tracks(bundle, policy), stats)
    ctx = make_ctx(bundle, stats, fits)
    bank = None
    if memorisation:  # the R2 bank's tracks (TRAIN truth in the all-TRAIN prior frame), standardised velocities
        n = len(bundle.train_ids)
        bank = VelBank([bundle.train("truth_bank", i) for i in range(n)], [bundle.train("valid", i) for i in range(n)],
                          ctx["w"])
    retimed, windows = retimed or {}, windows or {}
    boot_out = {}
    allrows = dict(rows)
    for name, r in retimed.items():
        allrows[name + "@retimed"] = r
    res = {"config": {"truth_ref": truth_ref, "policy": policy, "stats": stats_src, "n_boot": n_boot,
                      "seed": SEED, "bundle": bundle.path, "bundle_commits": bundle.meta["commits"],
                      "n_train_windows": fits["n_train_windows"], "eps": fits["eps"],
                      "pca_explained": {"%s|%s" % k: p.explained for k, p in fits["pca"].items()},
                      "group_mean_abs_v": stats["group_mean_abs_v"], "dropped_columns": stats["dropped_columns"],
                      "ref": ref}, "sets": {}}
    ok = primary_val(bundle.meta["val"])
    for set_name, idx in (("ok99", ok), ("all103", list(range(len(bundle.val_ids))))):
        tracks, spk_names = val_tracks(bundle, idx, truth_ref, policy)
        se = SetEval(tracks, len(spk_names), ctx, fits, n_boot)
        if policy == "static_ref":
            se.to_bank = [(bundle.meta["val"][i]["reference"], bundle.meta["priors"]["rotation_reference"]) for i in idx]
        sres = {"n_tracks": len(idx), "n_speakers": len(spk_names), "track_ids": [bundle.val_ids[i] for i in idx],
                "n_windows": {"%s|%s" % k: len(w["idx"]) for k, w in se.win.items()},
                "n_memorisation_segments": int(sum(len(s) for s in se.segs)), "rows": {}}
        boots = {}
        for name, r in allrows.items():
            ys = [r[i] for i in idx]
            yw = [windows[name][i] for i in idx] if windows.get(name) else None
            p1, b1 = se.fmd(ys, yw=yw)
            p2, b2 = se.additive(ys)
            point = {**p1, **p2}
            boot = {**b1, **{k: v for k, v in b2.items()}}
            row = {"n_samples": int(len(ys[0])), "point": point, "fmd_windowed": yw is not None,
                   "ci": {k: ci(v) for k, v in boot.items()}}
            if memorisation:
                row["memorisation"] = se.memorisation(ys, bank)
            base = name.split("@")[0]
            if edits and base in edits and "@" not in name:
                row["edits"] = edit_summary(edits[base], idx, np.array([t.spk for t in tracks]), len(spk_names), se.dr)
            boots[name] = boot
            if save_boot:
                boot_out.update({"%s::%s::%s" % (set_name, name, k): np.asarray(v, dtype=np.float64)
                                 for k, v in boot.items()})
            sres["rows"][name] = row
            print("%s %s done (%.0f s)" % (set_name, name, time.time() - t0), flush=True)
        for name in allrows:
            pairs = []
            if ref in boots and name != ref:
                pairs.append(("minus_" + ref, ref))
            if name.endswith("@retimed"):
                pairs.append(("retimed_minus_asr", name.split("@")[0]))
            for label, other in pairs:
                d = {}
                for k in boots[name]:
                    if k in boots[other]:
                        pv, po = sres["rows"][name]["point"].get(k), sres["rows"][other]["point"].get(k)
                        if isinstance(pv, float) and isinstance(po, float):
                            d[k] = {"diff": pv - po, "ci": ci(boots[name][k] - boots[other][k])}
                sres["rows"][name][label] = d
        res["sets"][set_name] = sres
        if save_boot:
            boot_out["%s::draws" % set_name] = se.dr
    res["runtime_s"] = time.time() - t0
    os.makedirs(out_dir, exist_ok=True)
    if save_boot:
        np.savez(os.path.join(out_dir, "boot.npz"), **boot_out)
    json.dump(res, open(os.path.join(out_dir, "eval.json"), "w"), indent=1, default=float)
    open(os.path.join(out_dir, "eval.md"), "w").write(to_md(res))
    return res
