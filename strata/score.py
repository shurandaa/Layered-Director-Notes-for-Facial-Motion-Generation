"""Scoring of generated motion: Tab. 1 metrics of the paper (validation set, three datasets).

    python -m strata.score bundles   --prep PREP --out BUNDLES
    python -m strata.score fmd       --bundles BUNDLES --rows RUN1,RUN2,RUN3 --out OUT [--densities standard]
    python -m strata.score roundtrip --prep PREP --data DATA --rows RUN1,RUN2,RUN3 --out OUT [--densities standard]

A run directory holds rows_asr/<density>/<clip>.npz with X (S, T, 63) (S samples), M (the constraint mask) and clip,
as written by `python -m strata.generate`.  Seeds (run directories) are pooled.
* fmd: velocity FMD on unconstrained entries, seam, W1 per dataset (Supp. D); `fmd_unc|all` is the FMD of Tab. 1
  (absolute values; Tab. 1 divides by ours), `seam_headline|all` the seam.
* roundtrip: the generated motion is re-encoded (Sec. 4.1) and compared with the notes it was generated from;
  "event exec (written recall)" = event recall, "state exec (written, exact)" = state execution, "tendency exec
  (written)" = tendency execution of Tab. 1 (first sample of each run, as in the paper).  TRUTH = the real motion
  through the same pipeline.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

from strata import conditions as C, evaluate as E, roundtrip as RT, tendency as T

DATASETS = ("talkvid", "ravdess", "crema-d")
DENSITIES = ("full", "standard", "lite")


def cmd_bundles(a):
    E.build(a.prep, a.out, rows=DENSITIES)


def cmd_fmd(a):
    binfo = json.load(open(os.path.join(a.bundles, "bundles.json")))
    rows = a.rows.split(",")
    out = {}
    for den in a.densities.split(","):
        evals = {}
        for ds in binfo["datasets"]:
            o = os.path.join(a.out, "evals", den, ds)
            E.eval_one({"bundle": binfo["dirs"][den][ds], "rows": {"ours": rows}, "row": den, "n_boot": a.n_boot,
                        "memorisation": False, "ref": "ours", "out": o})
            evals[ds] = o
        _, per = E.macro_candidates(evals, {"ours": "ours"})
        out[den] = per["ours"]
        print("density %s:" % den)
        for ds, p in per["ours"].items():
            print("  %-8s FMD %.4f  seam %.4f  W1 %.4f" % (ds, p["fmd_unc|all"], p["seam_headline|all"],
                                                         p["w1_headline"]))
    json.dump(out, open(os.path.join(a.out, "fmd_summary.json"), "w"), indent=1, default=float)


def cmd_roundtrip(a):
    t0 = time.time()
    man = json.load(open(os.path.join(a.prep, "manifest.json")))
    names, hz = man["names"], float(man["hz"])
    scripts = json.load(open(os.path.join(a.data, "scripts.json")))
    codec = C.load_constants()
    q = codec["tendency_q"]
    dirs = a.rows.split(",")
    os.makedirs(a.out, exist_ok=True)
    for den in a.densities.split(","):
        r = man["val_rows"].index(den)
        for ds in DATASETS:
            ids = sorted(t for t, e in man["tracks"].items() if e["split"] == "val" and e.get("dataset") == ds)
            if a.limit:
                ids = ids[:a.limit]
            with open(os.path.join(a.out, "rt_%s_%s.jsonl" % (den, ds)), "w") as fo:
                for tid in ids:
                    e = man["tracks"][tid]
                    texts = json.load(open(os.path.join(a.prep, e["texts_file"])))
                    full, written = texts["full"], texts["variants"][r]["text"]
                    n = int(full["n_frames"])
                    tt = scripts[tid]
                    sentences, _ = C.content_words(tt)
                    segs = T.segments_from_sentences(sentences, n) if sentences else []
                    with np.load(os.path.join(a.prep, e["file"])) as z:
                        xs = {"TRUTH": [z["target_static_ref"]], "ours": []}
                    for d in dirs:
                        with np.load(os.path.join(d, "rows_asr", den, tid.replace("#", "__seg") + ".npz")) as z:
                            X = z["X"][None] if z["X"].ndim == 2 else z["X"]
                        xs["ours"].append(X[0])
                    rec = {"track_id": tid, "speaker": e["speaker"], "dataset": ds, "row": den, "n_frames": n,
                           "cands": {c: [RT.counts(RT.encode(x, full, tt, names, hz, codec), written, full, segs, n,
                                                   hz, q) for x in lst] for c, lst in xs.items()}}
                    fo.write(json.dumps(rec) + "\n")
            print("roundtrip %s %s: %d clips (%.0f s)" % (den, ds, len(ids), time.time() - t0), flush=True)
    RT.main(["--dir", a.out, "--focus", "ours", "--vs", "TRUTH"])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("bundles")
    b.add_argument("--prep", required=True)
    b.add_argument("--out", required=True)
    f = sub.add_parser("fmd")
    f.add_argument("--bundles", required=True)
    f.add_argument("--rows", required=True)
    f.add_argument("--out", required=True)
    f.add_argument("--densities", default="standard")
    f.add_argument("--n_boot", type=int, default=1000)
    r = sub.add_parser("roundtrip")
    r.add_argument("--prep", required=True)
    r.add_argument("--data", required=True)
    r.add_argument("--rows", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--densities", default="standard")
    r.add_argument("--limit", type=int, default=0)
    a = ap.parse_args(argv)
    {"bundles": cmd_bundles, "fmd": cmd_fmd, "roundtrip": cmd_roundtrip}[a.cmd](a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
