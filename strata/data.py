"""The released data (written by release/export/export_data.py from the paper's prep).

    data/clips.json      one record per clip: track_id, clip_id, dataset, split (train / val), speaker, n_frames,
                         status (script status: ok / no_speech / low_audio), meta (segment_id, track_start, time_sec)
    data/scripts.json    track_id -> script w: words with start / end frames (30 Hz), marks, language
    data/curves_<split>.npz   <index>__features (T, 63) float32 tracked curves x (Sec. 3.1), <index>__observed /
                         __quality / __is_gap_filled (T, 4) per channel group (tracking validity)
    data/names.json      the 52 blendshape names (ARKit order of the columns) and hz
The curves are MediaPipe tracks of TalkVid, RAVDESS and CREMA-D clips; see the dataset licences.
"""
import json
import os

import numpy as np

ARRAYS = ("features", "observed", "quality", "is_gap_filled")


def load(data_dir: str, splits=("train", "val")) -> tuple:
    """(clips [records], tracks {track_id: track dict as the encoder reads it}, scripts {track_id: script}, names, hz)."""
    clips = json.load(open(os.path.join(data_dir, "clips.json")))
    scripts = json.load(open(os.path.join(data_dir, "scripts.json")))
    info = json.load(open(os.path.join(data_dir, "names.json")))
    tracks = {}
    for sp in splits:
        with np.load(os.path.join(data_dir, "curves_%s.npz" % sp)) as z:
            for c in clips:
                if c["split"] != sp:
                    continue
                t = {"track_id": c["track_id"], "clip_id": c["clip_id"], **c["meta"]}
                for k in ARRAYS:
                    t[k] = z["%d__%s" % (c["index"], k)]
                tracks[c["track_id"]] = t
    return [c for c in clips if c["split"] in splits], tracks, scripts, info["names"], float(info["hz"])
