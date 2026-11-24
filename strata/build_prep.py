"""Build the prep directory (training variants and validation rows) from the released data.

    python -m strata.build_prep --data DATA --out PREP [--limit N]

Same procedure and settings as the paper's prep (Supp. C "training data and masking"): full notes by the encoder;
per training clip variant 0 = the full notes and 8 randomly masked variants R(N) (seeded by the clip id, so the
result does not depend on the order or the machine); per validation clip the full / standard / lite rows.  The
mirrored training variants of the research prep are not written: the final configuration does not use them.
tests/test_release.py checks every written array against the arrays the paper's model was trained and evaluated on.
"""
import argparse
import json
import os
import time

import numpy as np

from strata import conditions as C, data as D, prep as P, tendency as T, vocabulary as V

PARAMS = {"k_std": 30, "k_lite": 6, "c_std": 6, "c_lite": 2, "amplitude": "dev_high", "quota": "per_track_stochastic",
          "state_cap": "per_minute_ceil_min1"}  # standard / lite quotas (Supp. A "densities")
BLINK_DROP_P = 0.9  # blink masking (Sec. 3.4)
PRIORS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "priors.json")


def run(data_dir: str, out: str, limit: int = None, log=print) -> dict:
    t0 = time.time()
    clips, tracks, scripts, names, hz = D.load(data_dir)
    if limit:
        keep = {c["track_id"] for sp in ("train", "val") for c in [x for x in clips if x["split"] == sp][:limit]}
        clips = [c for c in clips if c["track_id"] in keep]
    # the order of the research prep (training statistics are sums over clips): split, then TalkVid, RAVDESS,
    # CREMA-D, then clip id
    order = {"talkvid": 0, "ravdess": 1, "crema-d": 2}
    clips = sorted(clips, key=lambda c: (c["split"] != "train", order[c["dataset"]], c["track_id"]))
    codec = C.load_constants()
    draw = P.make_draw(BLINK_DROP_P)
    fulls = {c["track_id"]: C.encode_full_notes(tracks[c["track_id"]], scripts[c["track_id"]], names, hz, codec)
             for c in clips}
    log("encoded %d clips (%.0f s)" % (len(fulls), time.time() - t0))
    priors = json.load(open(PRIORS))
    prior = P.prior_header([fulls[c["track_id"]] for c in clips if c["split"] == "train"], priors, names, codec)
    entries, z_train, units = {}, [], {}
    for c in clips:
        tid, sp = c["track_id"], c["split"]
        z, rec = P.prep_track_v2(fulls[tid], tracks[tid], scripts[tid], sp, names, PARAMS, codec, prior, draw=draw)
        if sp == "train":
            z_train.append(z)
        units[tid] = P.viseme_units(scripts[tid])
        fname = os.path.join(sp, P.track_file(tid))
        tname = os.path.join("texts", sp, P.track_file(tid)[:-4] + ".json")
        for f in (fname, tname):
            os.makedirs(os.path.dirname(os.path.join(out, f)), exist_ok=True)
        np.savez_compressed(os.path.join(out, fname), **z)
        with open(os.path.join(out, tname), "w") as f:
            json.dump(rec, f, sort_keys=True)
        entries[tid] = {"split": sp, "clip_id": c["clip_id"], "speaker": c["speaker"], "dataset": c["dataset"],
                        "n_frames": int(fulls[tid]["n_frames"]), "n_words": len(scripts[tid]["words"]),
                        "language": scripts[tid]["language"], "status": c["status"], "file": fname,
                        "sha256": P.sha256_file(os.path.join(out, fname)), "keys": sorted(z),
                        "arrays_sha256": P.arrays_sha256(z), "texts_file": tname,
                        "texts_sha256": P.sha256_file(os.path.join(out, tname))}
    slots, aslots = C.state_slots(), C.asym_slots()
    files = {}
    for key, name, obj in (("norm_v2", "stats_norm_v2.json", P.norm_stats_v2(z_train, names, [s for s, _, _, _ in slots])),
                           ("prior_header", "prior_header.json", prior), ("viseme_units", "viseme_units.json", units)):
        with open(os.path.join(out, name), "w") as f:
            json.dump(obj, f, indent=1 if key != "viseme_units" else None, sort_keys=True, ensure_ascii=False)
        files[key] = {"file": name, "sha256": P.sha256_file(os.path.join(out, name))}
    manifest = {"format": P.FORMAT, "names": names, "hz": hz, "params": PARAMS, "draw": draw, "g7": False,
                "val_rows": ["full", "standard", "lite"],
                "splits": {"val": {"gaze_unit": "dimensionless_iris_offset_proxy"}}, "versions": P.codec_info(codec), "smoke": bool(limit),
                "slots": {"state": [s for s, _, _, _ in slots], "asym": aslots, "tendency": list(T.PARAMS),
                          "n_v2_inputs": len(slots) + len(aslots) + 1 + 2 * len(T.PARAMS)},
                "prior_header": files["prior_header"], "viseme_units": files["viseme_units"],
                "stats": {"norm_v2": files["norm_v2"]}, "delta": V.delta_c_row(names).tolist(),
                "tracks": entries, "runtime_s": time.time() - t0}
    with open(os.path.join(out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1, sort_keys=True, default=str)
    log("prep: %d clips -> %s (%.0f s)" % (len(entries), out, time.time() - t0))
    return manifest


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Build the prep directory from the released data.")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None, help="first N clips per split (a quick test)")
    a = ap.parse_args(argv)
    run(a.data, a.out, a.limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
