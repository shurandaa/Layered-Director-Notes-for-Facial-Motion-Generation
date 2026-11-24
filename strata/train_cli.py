"""Train the generator (paper Sec. 3.4 "training"; Supp. C).

    python -m strata.train_cli --prep PREP --out RUN [--seed 0] [--config configs/final.json] [--max_steps N]
        [--device cuda]

configs/final.json is the configuration of the paper's model (50k steps, batch 128 windows of 96 frames, blink
masking in the prep, script perturbation on); the three released checkpoints are seeds 0, 1, 2 (about 40 min each on
one NVIDIA B200).  Writes RUN/checkpoint.pt (EMA and raw weights, configuration, training statistics), config.json and
history.json (loss every 500 steps).
"""
import argparse
import json
import os
import sys

from strata import train as TR

CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs", "final.json")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--prep", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--config", default=CONFIG)
    ap.add_argument("--max_steps", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--workers", type=int, default=0)
    a = ap.parse_args(argv)
    c = json.load(open(a.config))
    c.update(prep=os.path.abspath(a.prep), prep_manifest_sha256=None, seed=a.seed)
    TR.fit(c, a.out, a.device, max_steps=a.max_steps, workers=a.workers)
    return 0


if __name__ == "__main__":
    sys.exit(main())
