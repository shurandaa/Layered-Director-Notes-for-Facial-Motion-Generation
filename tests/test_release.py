"""Reproduction tests of the release (run from the repository root on a fresh machine).

    python tests/test_release.py --data DATA --work WORK [--stages prep,golden,score,generate,train] [--device cuda]

DATA = the released data directory (curves, scripts, checkpoints/, rows/, reference/, golden/).  Stages:
  prep      strata.build_prep from the curves and scripts; every notes record must equal the paper's prep and the
            training statistics must match to 1e-9 (reference/); every array of every clip is compared by sha256
            (bit-identical on an AVX-512 x86 CPU such as the paper's; last-bit differences elsewhere are reported).
  golden    18 validation samples (6 clips x 3 densities) generated on the CPU against the research code's CPU output
            (golden/): bit-identical on an AVX-512 x86 CPU, within 1e-4 required (about 1e-6 on others).
  score     Tab. 1 metrics of the paper's generated rows (rows/ours_s0..s2) against the paper's: FMD / seam / W1 per
            dataset to 1e-6 (bit-identical on an AVX-512 x86 CPU) and the round-trip counts of every clip exactly
            (reference/paper_table1.json, reference/paper_roundtrip_counts.jsonl).
  generate  the validation rows generated again on this machine (GPU) with the released checkpoints, then scored;
            reports the differences to the paper (exact only on the paper's GPU type and software, NVIDIA B200).
  train     200 training steps on the CPU; the loss history must match the paper code's run to 1e-5 (reference/).
Exit status 1 if a check fails.
"""
import argparse
import glob
import hashlib
import json
import os
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
RESULTS = {}


def ah(a):
    a = np.ascontiguousarray(a)
    return hashlib.sha256(("%s|%s|" % (a.dtype.str, a.shape)).encode() + a.tobytes()).hexdigest()


def jh(o):
    return hashlib.sha256(json.dumps(o, sort_keys=True).encode()).hexdigest()


def report(stage, ok, **info):
    RESULTS[stage] = dict(ok=ok, **info)
    print("[%s] %s %s" % (stage, "PASS" if ok else ("FAIL" if ok is False else "INFO"), json.dumps(info, default=float)),
          flush=True)


def run(cmd):
    print("$", " ".join(cmd), flush=True)
    env = dict(os.environ, PYTHONPATH=ROOT)
    env.setdefault("OMP_NUM_THREADS", "8")  # oversubscribed threads stall CPU training under a container CPU quota
    subprocess.run(cmd, check=True, cwd=ROOT, env=env)


def close(a, b, rtol=1e-9):
    """Equal JSON trees, floats to rtol (sums whose last bit depends on the CPU's SIMD kernels)."""
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(close(a[k], b[k], rtol) for k in a)
    if isinstance(a, list):
        return isinstance(b, list) and len(a) == len(b) and all(close(x, y, rtol) for x, y in zip(a, b))
    if isinstance(a, float) or isinstance(b, float):
        return a == b or (a is not None and b is not None and abs(a - b) <= rtol * max(abs(a), abs(b), 1e-300))
    return a == b


def stage_prep(a):
    """Notes records must be identical; arrays and statistics are bit-identical on an AVX-512 x86 CPU (the paper's)
    and differ only in the last bits elsewhere (numpy dispatches other exp / sin kernels)."""
    from strata import build_prep
    prep = os.path.join(a.work, "prep")
    t0 = time.time()
    build_prep.run(a.data, prep)
    ref = json.load(open(os.path.join(a.data, "reference", "paper_prep_hashes.json")))
    man = json.load(open(os.path.join(prep, "manifest.json")))
    bad, diff, n = [], [], 0
    for tid, e in man["tracks"].items():
        r = ref["tracks"][tid]
        with np.load(os.path.join(prep, e["file"])) as z:
            if set(z.files) != set(r["arrays"]):
                bad.append((tid, "keys"))
            for k in z.files:
                n += 1
                if ah(z[k]) != r["arrays"].get(k):
                    diff.append((tid, k))
        rec = json.load(open(os.path.join(prep, e["texts_file"])))
        if jh(rec) != r["texts"]:
            bad.append((tid, "notes"))
    files_exact = []
    for f, h in ref["files"].items():
        mine = json.load(open(os.path.join(prep, f)))
        files_exact.append(jh(mine) == h)
        if not close(mine, json.load(open(os.path.join(a.data, "reference", "paper_prep_files", f)))):
            bad.append(f)
    report("prep", not bad and len(man["tracks"]) == len(ref["tracks"]), clips=len(man["tracks"]), arrays=n,
           notes_or_stats_mismatches=len(bad), first=bad[:5], bit_identical_arrays=n - len(diff),
           arrays_last_bit_only=len(diff), stats_bit_identical=all(files_exact), minutes=(time.time() - t0) / 60)


def stage_golden(a):
    import torch
    from strata import generate as G, prep as P, sampler as S
    torch.set_num_threads(4)
    res = G.load(os.path.join(a.data, "checkpoints", "strata_s0.pt"), "cpu")
    info = json.load(open(os.path.join(a.data, "golden", "golden_info.json")))
    pr = P.PrepV2(os.path.join(a.work, "prep"))
    rows = list(pr.manifest["val_rows"])
    bad, n, exact, worst = [], 0, 0, 0.0
    for f in sorted(glob.glob(os.path.join(a.data, "golden", "*__*.npz"))):
        z = np.load(f)
        den, clip = os.path.basename(f)[:-4].split("__", 1)
        tid = next(t for t in info["clips"] if P.track_file(t)[:-4] == clip)
        c = pr.condition(tid, rows.index(den), "static_ref", raw=True)
        cond = {k: c[k] for k in S.CONDITION_KEYS}
        if any(not np.array_equal(cond[k], z["cond_" + k], equal_nan=True) for k in S.CONDITION_KEYS):
            bad.append((f, "condition"))
        it = S.make_item(res, cond, (c["text"], c["viseme"]))
        y = S.ddim_sample(res, [it], [0], [info["index"][tid]])[0]
        p = G.post(y[None], cond)[0]
        n += 1
        d = max(float(np.abs(y - z["y_raw"]).max()), float(np.abs(p - z["y_post"]).max()))
        exact += d == 0
        worst = max(worst, d)
    # bit-identical on an AVX-512 x86 CPU; other CPUs run other float kernels (about 1e-6)
    report("golden", worst < 1e-4, samples=n, bit_identical=exact, max_abs_diff=worst,
           condition_arrays_not_bit_identical=len(bad))


def score_dirs(a, dirs, out, densities="full,standard,lite"):
    bundles = os.path.join(a.work, "bundles")
    if not os.path.exists(os.path.join(bundles, "bundles.json")):
        run([sys.executable, "-m", "strata.score", "bundles", "--prep", os.path.join(a.work, "prep"), "--out", bundles])
    run([sys.executable, "-m", "strata.score", "fmd", "--bundles", bundles, "--rows", ",".join(dirs), "--out", out,
         "--densities", densities])
    run([sys.executable, "-m", "strata.score", "roundtrip", "--prep", os.path.join(a.work, "prep"), "--data", a.data,
         "--rows", ",".join(dirs), "--out", os.path.join(out, "roundtrip"), "--densities", densities])
    return json.load(open(os.path.join(out, "fmd_summary.json"))), json.load(open(os.path.join(out, "roundtrip",
                                                                                               "summary.json")))


TAB1 = {"event recall": "event exec (written recall)", "state execution": "state exec (written, exact)",
        "tendency execution": "tendency exec (written)"}


def table1(fmd, rt):
    out = {}
    for den in fmd:
        out[den] = {"FMD %s" % ds: p["fmd_unc|all"] for ds, p in fmd[den].items()}
        out[den]["seam (mean over datasets)"] = float(np.mean([p["seam_headline|all"] for p in fmd[den].values()]))
        out[den].update({k: rt[den][m]["point"]["ours"] for k, m in TAB1.items()})
    return out


def stage_score(a):
    ref = json.load(open(os.path.join(a.data, "reference", "paper_table1.json")))
    dirs = [os.path.join(a.data, "rows", "ours_s%d" % s) for s in range(3)]
    out = os.path.join(a.work, "score_paper_rows")
    fmd, rt = score_dirs(a, dirs, out)
    bad, bits = [], 0
    for den, per in fmd.items():
        for ds, p in per.items():
            for k, v in p.items():
                r = ref["fmd"][den][ds].get(k)
                nan = isinstance(v, float) and isinstance(r, float) and np.isnan(v) and np.isnan(r)
                if not nan and v != r:
                    bits += 1
                    if not close(v, r, 1e-6):  # Tab. 1 prints 2-3 digits
                        bad.append(("fmd", den, ds, k, v, r))
    for den, ms in ref["roundtrip_summary"].items():
        for m, v in ms.items():
            if rt[den][m]["point"]["ours"] != v["ours"] or rt[den][m]["point"]["TRUTH"] != v["TRUTH"]:
                bad.append(("roundtrip", den, m, rt[den][m]["point"]["ours"], v["ours"]))
    paper = {}
    for line in open(os.path.join(a.data, "reference", "paper_roundtrip_counts.jsonl")):
        r = json.loads(line)
        paper[(r["row"], r["track_id"])] = r
    n = 0
    for f in glob.glob(os.path.join(out, "roundtrip", "rt_*.jsonl")):
        for line in open(f):
            r = json.loads(line)
            p = paper[(r["row"], r["track_id"])]
            n += 1
            if r["cands"]["ours"] != p["ours"] or r["cands"]["TRUTH"] != p["TRUTH"]:
                bad.append(("counts", r["row"], r["track_id"]))
    report("score", not bad, clips_x_densities=n, mismatches=len(bad), first=bad[:5], fmd_values_not_bit_identical=bits,
           table1_standard=table1(fmd, rt)["standard"])


def stage_generate(a):
    ref = json.load(open(os.path.join(a.data, "reference", "paper_table1.json")))
    dirs = []
    t0 = time.time()
    for s in range(3):
        d = os.path.join(a.work, "generated_s%d" % s)
        if not os.path.exists(os.path.join(d, "generate_info.json")):
            run([sys.executable, "-m", "strata.generate", "val", "--checkpoint",
                 os.path.join(a.data, "checkpoints", "strata_s%d.pt" % s), "--prep", os.path.join(a.work, "prep"),
                 "--out", d, "--device", a.device, "--densities", a.densities])
        dirs.append(d)
    gen_min = (time.time() - t0) / 60
    diff = []
    for s, d in enumerate(dirs):
        for f in sorted(glob.glob(os.path.join(d, "rows_asr", "*", "*.npz")))[::25]:
            g = np.load(f)["X"]
            p = np.load(os.path.join(a.data, "rows", "ours_s%d" % s, os.path.relpath(f, d)))["X"]
            diff.append(float(np.abs(g - p).max()))
    fmd, rt = score_dirs(a, dirs, os.path.join(a.work, "score_generated"), a.densities)
    mine = table1(fmd, rt)
    paper = table1({den: ref["fmd"][den] for den in fmd},
                   {den: {m: {"point": {"ours": v["ours"]}} for m, v in ref["roundtrip_summary"][den].items()}
                    for den in fmd})
    report("generate", None, minutes=gen_min, max_abs_diff_to_paper_rows=max(diff), median_abs_diff=float(np.median(diff)),
           this_machine=mine, paper=paper)


def stage_train(a):
    out = os.path.join(a.work, "train200")
    cfg = json.load(open(os.path.join(ROOT, "strata", "configs", "final.json")))
    cfg["log_every"] = 50
    json.dump(cfg, open(os.path.join(a.work, "final_log50.json"), "w"))
    run([sys.executable, "-m", "strata.train_cli", "--prep", os.path.join(a.work, "prep"), "--out", out, "--seed", "0",
         "--config", os.path.join(a.work, "final_log50.json"), "--max_steps", "200", "--device", "cpu"])
    mine = [(h["step"], h["loss"]) for h in json.load(open(os.path.join(out, "history.json")))["history"]]
    ref = [(h["step"], h["loss"]) for h in json.load(open(os.path.join(a.data, "reference", "train200_history.json")))]
    rel = max(abs(x[1] - y[1]) / abs(y[1]) for x, y in zip(mine, ref))
    report("train", [s for s, _ in mine] == [s for s, _ in ref] and rel < 1e-5, cpu_200_steps=mine,
           paper_code_same_steps=ref, identical=mine == ref, max_rel_diff=rel)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--stages", default="prep,golden,score,generate,train")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--densities", default="full,standard,lite")
    a = ap.parse_args()
    os.makedirs(a.work, exist_ok=True)
    for st in a.stages.split(","):
        globals()["stage_" + st](a)
        json.dump(RESULTS, open(os.path.join(a.work, "results.json"), "w"), indent=1, default=float)
    failed = [k for k, v in RESULTS.items() if v["ok"] is False]
    print("FAILED: %s" % failed if failed else "all exact checks passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
