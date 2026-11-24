"""Generate facial motion from director notes (paper Sec. 3.4).

    python -m strata.generate notes --checkpoint CK.pt --notes NOTES.json --script SCRIPT.json --out OUT.npz
        [--seed 0] [--device cuda]
    python -m strata.generate val --checkpoint CK.pt --prep PREP --out RUN [--samples 4] [--densities full,standard,lite]
        [--device cuda] [--limit N]

notes: NOTES.json is a set of director notes (strata.notes; header rest / reference, tendency, state, event),
SCRIPT.json the script {"words": [{"w", "start_frame", "end_frame"}], "marks": [...], "language": "en"} at 30 Hz
(strata.script).  OUT.npz holds x (T, 63).
val: the paper's validation rows: sample s of clip k uses seed s and noise index k (the position of the clip in the
sorted validation list), then head smoothing (sigma_h = 0.75) and re-projection, as for Tab. 1.  Writes
RUN/rows_asr/<density>/<clip>.npz {X (S, T, 63), M, clip}.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

from strata import conditions as C, network as N, notes as NO, postprocess as PP, prep as P, sampler as S
from strata import script as SC


def load(checkpoint: str, device: str = "cpu") -> S.Resources:
    """The released generator: EMA weights, configuration and training statistics."""
    ck = torch.load(checkpoint, map_location=device, weights_only=False)
    meta = ck.get("l2v2_meta", ck)  # a checkpoint written by strata.train carries the training metadata block
    model = N.build_model(ck["config"], ck["free"], meta["n_v2"]).to(device)
    model.load_state_dict(ck["ema"])
    model.eval()
    return S.Resources(model, ck["config"], ck["norm"], list(ck["excluded"]), np.asarray(ck["rest_null"]), None,
                       meta["names"], None, device)


def post(X: np.ndarray, cond: dict) -> np.ndarray:
    """Head smoothing sigma_h and re-projection with Pi_delta (Supp. C "sampling")."""
    X = PP.smooth_head(X)
    if np.asarray(cond["M"], bool).any():
        X = PP.reproject(X, C.projection_input(cond))
    return X


def from_notes(res: S.Resources, notes: dict, script: dict, seed: int = 0) -> np.ndarray:
    """x (T, 63) for one set of notes and its script."""
    NO.validate_track(notes)
    cond = C.notes_conditions(notes, res.names, "static_ref", None, C.load_constants())
    timing, visemes = SC.features(script, int(notes["n_frames"]), lexical="viseme_v2")
    it = S.make_item(res, {k: cond[k] for k in S.CONDITION_KEYS}, (timing, visemes))
    x = S.ddim_sample(res, [it], [seed], [0])[0]
    return post(x[None], cond)[0]


def cmd_notes(a):
    res = load(a.checkpoint, a.device)
    x = from_notes(res, json.load(open(a.notes)), json.load(open(a.script)), a.seed)
    np.savez(a.out, x=x)
    print("wrote", a.out, x.shape)


def cmd_val(a):
    t0 = time.time()
    res = load(a.checkpoint, a.device)
    pr = P.PrepV2(a.prep)
    rows = list(pr.manifest["val_rows"])
    ids = sorted(pr.ids("val"))[:a.limit]
    for den in a.densities.split(","):
        r = rows.index(den)
        conds = [pr.condition(tid, r, "static_ref", raw=True) for tid in ids]
        items = [S.make_item(res, {k: c[k] for k in S.CONDITION_KEYS}, (c["text"], c["viseme"])) for c in conds]
        ys = [S.ddim_sample(res, items, [s] * len(items), list(range(len(items)))) for s in range(a.samples)]
        d = os.path.join(a.out, "rows_asr", den)
        os.makedirs(d, exist_ok=True)
        for k, tid in enumerate(ids):
            X = post(np.stack([ys[s][k] for s in range(a.samples)]), conds[k])
            np.savez(os.path.join(d, P.track_file(tid)), X=X, M=items[k]["m_real"], clip=tid)
        print("density %s: %d clips x %d samples (%.1f min)" % (den, len(ids), a.samples, (time.time() - t0) / 60),
              flush=True)
    info = {"checkpoint": os.path.abspath(a.checkpoint), "samples": a.samples, "device": a.device,
            "gpu": torch.cuda.get_device_name(0) if a.device.startswith("cuda") else None,
            "torch": torch.__version__, "numpy": np.__version__, "minutes": (time.time() - t0) / 60}
    json.dump(info, open(os.path.join(a.out, "generate_info.json"), "w"), indent=1)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    n = sub.add_parser("notes")
    n.add_argument("--checkpoint", required=True)
    n.add_argument("--notes", required=True)
    n.add_argument("--script", required=True)
    n.add_argument("--out", required=True)
    n.add_argument("--seed", type=int, default=0)
    n.add_argument("--device", default="cpu")
    v = sub.add_parser("val")
    v.add_argument("--checkpoint", required=True)
    v.add_argument("--prep", required=True)
    v.add_argument("--out", required=True)
    v.add_argument("--samples", type=int, default=4)
    v.add_argument("--densities", default="full,standard,lite")
    v.add_argument("--device", default="cpu")
    v.add_argument("--limit", type=int, default=None)
    a = ap.parse_args(argv)
    torch.set_grad_enabled(False)
    {"notes": cmd_notes, "val": cmd_val}[a.cmd](a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
