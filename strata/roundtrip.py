"""Cycle-consistent evaluation (paper Sec. 4.1; Supp. D): generated motion is re-encoded by the same encoder
and compared with the notes it was generated from: event recall (IoU >= 0.5, one-to-one matching), event
intensity agreement, state execution, tendency execution; pooled per dataset, macro over datasets.
"""
import argparse
import glob
import json
import numpy as np
import os
from strata import conditions as _conditions
from strata import evaluate as _evaluate
from strata import representation as _representation
from strata import tendency as _tendency
from strata import vocabulary as _vocabulary


METRICS = {  # name: (key, numerator index, denominator index or "f1")
    "event exec (written recall)": ("ev_written", 2, 0),
    "event F1 vs full": ("ev_full", None, "f1"),
    "state exec (written, exact)": ("st_written", 0, 1),
    "state present (written)": ("st_written_on", 0, 1),
    "state agreement vs full": ("st_full", 0, 1),
    "tendency exec (written)": ("tend_written", 0, 1),
    "tendency agreement vs full": ("tend_full", 0, 1),
}


B, SEED = 1000, 0


def track_terms(samples: list, key: str, i, j) -> tuple:
    """(numerator, denominator) of one track, its samples summed."""
    num = den = 0.0
    for c in samples:
        x = c[key]
        if j == "f1":
            t, p, m = (sum(v[k] for v in x.values()) for k in range(3))
            num += 2 * m
            den += t + p
        elif isinstance(x, dict):
            num += sum(v[i] for v in x.values())
            den += sum(v[j] for v in x.values())
        else:
            num += x[i]
            den += x[j]
    return num, den


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--focus", default="G3c_p09")
    ap.add_argument("--vs", default="G0,R1")
    a = ap.parse_args(argv)
    recs = {}
    for f in sorted(glob.glob(os.path.join(a.dir, "rt_*.jsonl"))):
        for line in open(f):
            r = json.loads(line)
            recs.setdefault(r["row"], {}).setdefault(r["dataset"], []).append(r)
    out, L = {}, ["# §10 round trip (VAL, report only); ratio of pooled counts, macro over datasets [95 % CI]", ""]
    for row in ("full", "standard", "lite"):
        if row not in recs:
            continue
        dsets = sorted(recs[row])
        cands = list(recs[row][dsets[0]][0]["cands"])
        rng = np.random.default_rng(SEED)
        draws = {}
        for ds in dsets:
            spk = sorted({r["speaker"] for r in recs[row][ds]})
            idx = {s: k for k, s in enumerate(spk)}
            cnt = np.zeros((B, len(spk)))
            for b in range(B):
                cnt[b] = np.bincount(rng.integers(0, len(spk), len(spk)), minlength=len(spk))
            draws[ds] = (idx, cnt)
        out[row] = {}
        L += ["## row %s (%s)" % (row, ", ".join("%s %d tracks" % (d, len(recs[row][d])) for d in dsets)), "",
              "| metric | " + " | ".join(cands) + " | " + " | ".join("%s - %s" % (a.focus, v) for v in a.vs.split(",")) + " |",
              "|---" * (1 + len(cands) + len(a.vs.split(","))) + "|"]
        for mname, (key, i, j) in METRICS.items():
            pt, bt = {}, {}
            for c in cands:
                pds, bds = [], []
                for ds in dsets:
                    idx, cnt = draws[ds]
                    num = np.zeros(len(idx))
                    den = np.zeros(len(idx))
                    for r in recs[row][ds]:
                        n_, d_ = track_terms(r["cands"][c], key, i, j)
                        num[idx[r["speaker"]]] += n_
                        den[idx[r["speaker"]]] += d_
                    pds.append(num.sum() / den.sum() if den.sum() else np.nan)
                    with np.errstate(invalid="ignore", divide="ignore"):
                        bds.append((cnt @ num) / (cnt @ den))
                pt[c], bt[c] = float(np.nanmean(pds)), np.nanmean(np.stack(bds), axis=0)
            ci = {c: np.nanpercentile(bt[c], [2.5, 97.5]).tolist() for c in cands}
            diffs = {}
            for v in a.vs.split(","):
                if v in bt and a.focus in bt:
                    dd = bt[a.focus] - bt[v]
                    diffs[v] = [pt[a.focus] - pt[v]] + np.nanpercentile(dd, [2.5, 97.5]).tolist()
            out[row][mname] = {"point": pt, "ci": ci, "diff": diffs}
            L.append("| %s | " % mname + " | ".join("%.3f [%.3f, %.3f]" % (pt[c], *ci[c]) for c in cands) + " | " +
                     " | ".join("%+.3f [%+.3f, %+.3f]" % tuple(diffs[v]) if v in diffs else "-" for v in a.vs.split(","))
                     + " |")
        L.append("")
    json.dump(out, open(os.path.join(a.dir, "summary.json"), "w"), indent=1)
    open(os.path.join(a.dir, "summary.md"), "w").write("\n".join(L))
    print("\n".join(L))
    return 0


def encode(x: np.ndarray, meta: dict, tt: dict, names: list, hz: float, codec: dict) -> dict:
    n = x.shape[0]
    nch = len(_representation.CHANNEL_NAMES)
    track = {k: meta[k] for k in _vocabulary.META_KEYS if k in meta}
    track.update({"features": np.asarray(x, dtype=np.float32), "observed": np.ones((n, nch), dtype=bool),
                  "quality": np.ones((n, nch)), "is_gap_filled": np.zeros((n, nch), dtype=bool)})
    return _conditions.encode_full_notes(track, tt, names, hz, codec)


def written_state(written: dict, enc: dict, n: int, hz: float) -> tuple:
    lw, le = timeline_labels(written, n, hz), timeline_labels(enc, n, hz)
    agree = tot = on = 0
    for tl in lw:
        w = lw[tl] != "rest"
        tot += int(w.sum())
        agree += int((lw[tl][w] == le[tl][w]).sum())
        on += int((le[tl][w] != "rest").sum())
    return [agree, tot], [on, tot]


def counts(enc: dict, written: dict, full: dict, segs: list, n: int, hz: float, q: dict) -> dict:
    ue = v2_units(enc, hz)
    out = {"ev_written": event_counts(v2_units(written, hz), ue, _vocabulary.FAMILIES)["family"],
           "ev_full": event_counts(v2_units(full, hz), ue, _vocabulary.FAMILIES)["family"],
           "st_full": state_counts(full, enc, n, hz)}
    out["st_written"], out["st_written_on"] = written_state(written, enc, n, hz)
    out["tend_full"] = tendency_counts(full.get("tendency"), enc.get("tendency"), segs, hz, q) if segs else [0, 0]
    out["tend_written"] = (tendency_counts(written.get("tendency"), enc.get("tendency"), segs, hz, q)
                           if segs and written.get("tendency") else [0, 0])
    return out


IOU_MIN = 0.5


MID = 2  # l2_v2_tendency.LEVELS.index("mid")


def _match(pred: list, true: list) -> list:
    return _evaluate.match([{"s": s, "e": e} for s, e in pred], [{"s": s, "e": e} for s, e in true], IOU_MIN)


def v2_units(body: dict, hz: float) -> dict:
    """{family: [{"span": (s, e), "kind", "dir"}]} of the body's events (repeat expanded)."""
    out = {}
    for e in body.get("event") or []:
        for (s, _, _, en) in _vocabulary.event_units(e, hz):
            out.setdefault(e["family"], []).append({"span": (s, en), "kind": e.get("kind"), "dir": e.get("dir")})
    return out


def event_counts(true: dict, pred: dict, families) -> dict:
    """{"family": {f: [n_true, n_pred, n_match]}, "kind": [agree, n], "dir": [agree, n]}."""
    out = {"family": {}, "kind": [0, 0], "dir": [0, 0]}
    for f in families:
        t, p = true.get(f, []), pred.get(f, [])
        m = _match([u["span"] for u in p], [u["span"] for u in t])
        out["family"][f] = [len(t), len(p), len(m)]
        for i, j in m:
            for k in ("kind", "dir"):
                a, b = t[j].get(k), p[i].get(k)
                if a is not None or b is not None:
                    out[k][0] += int(a == b)
                    out[k][1] += 1
    return out


def timeline_labels(body: dict, n: int, hz: float) -> dict:
    """{timeline: (n,) labels} (T0569 timeline_labels with offset 0)."""
    lab = {tl: np.full(n, "rest", dtype=object) for tl, _, _ in _vocabulary.timelines()}
    by = {}
    for e in body.get("state") or []:
        by.setdefault(_vocabulary.entry_timeline(e), []).append(e)
    for tl, es in by.items():
        es = sorted(es, key=lambda e: _vocabulary.tau_frames(e["t"], hz)[0])
        for k, e in enumerate(es):
            fr = _vocabulary.tau_frames(e["t"], hz)
            s, en = fr[0], fr[3]
            if k + 1 < len(es):
                en = min(en, _vocabulary.tau_frames(es[k + 1]["t"], hz)[0])
            lab[tl][max(s, 0):max(en, 0)] = "%s|%s" % (e["level"], e.get("dir", ""))
    return lab


def state_counts(true: dict, pred: dict, n: int, hz: float) -> list:
    """[agree frames, total frames] over the 14 timelines."""
    lt, lp = timeline_labels(true, n, hz), timeline_labels(pred, n, hz)
    return [int(sum((lt[tl] == lp[tl]).sum() for tl in lt)), int(len(lt) * n)]


def tendency_levels(block, segments: list, hz: float, q: dict) -> list:
    """[[level index per param] per segment] (module docstring)."""
    out = []
    for a, b in segments:
        m = (int(a) + int(b)) // 2
        default = (block or {}).get("default") or {}
        seg = {}
        for s in (block or {}).get("segments") or []:
            sa, sb = _vocabulary.tau_frames(s["t"], hz)
            if sa <= m < sb:
                seg = s
                break
        row = []
        for p in _tendency.PARAMS:
            x = seg.get(p, default.get(p))
            row.append(MID if x is None else _tendency.parse_param(x, p, q)[0])
        out.append(row)
    return out


def tendency_counts(true_block, pred_block, segments: list, hz: float, q: dict) -> list:
    lt, lp = np.array(tendency_levels(true_block, segments, hz, q)), np.array(tendency_levels(pred_block, segments, hz, q))
    return [int((lt == lp).sum()), int(lt.size)]
