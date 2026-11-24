"""The condition c(N, w) (paper Sec. 3.4 "input"; Supp. C "input"): what the generator sees of the notes and
the script.  Per frame: X_obs (63), M (63), neutral values (63), state curves / asymmetry / state flag and tendency
values / mask (40), script timing (11) and viseme ids; with the noised motion, 319 values.
"""
import copy
import numpy as np
from strata import decoder as _decoder
from strata import encoder as _encoder
from strata import notes as _notes
from strata import script as _script
from strata import sparsify as _sparsify
from strata import tendency as _tendency
from strata import vocabulary as _vocabulary


PRIMARY_POLICY = "static_ref"


def load_constants(rest_stat: str = None) -> dict:
    """The calibrated constants used by every encode / decode / sparsify: neutral-value variant, decoder and
    encoder constants, tendency quantiles."""
    rest_stat = rest_stat or _vocabulary.VOCABULARY["rest"]["one_sided_stat"]
    return {"rest_stat": rest_stat, "fitted": _vocabulary.calibration_for(rest_stat), "tendency_q": _tendency.quantiles_for(rest_stat)}


def state_slots(vocabulary: dict = None) -> list:
    """[(slot, timeline, family, kind or None)]: l2_v2.timelines with the bs_tension timeline split by kind
    (vocabulary order)."""
    vocabulary = vocabulary or _vocabulary.VOCABULARY
    out = []
    for tl, fam, _ in _vocabulary.timelines(vocabulary):
        spec = vocabulary["families"][fam]
        if spec["type"] == "bs_tension":
            out += [("%s.%s" % (tl, k), tl, fam, k) for k in spec["kinds"]]
        else:
            out.append((tl, tl, fam, None))
    return out


def asym_slots(vocabulary: dict = None) -> list:
    """The state slots whose family / kind has a channel pair (l2_v2._pair_of)."""
    vocabulary = vocabulary or _vocabulary.VOCABULARY
    return [s for s, _, fam, kind in state_slots(vocabulary)
            if _vocabulary._pair_of(fam, {} if kind is None else {"kind": kind}, vocabulary)]


def entry_slot(e: dict, vocabulary: dict = None) -> str:
    vocabulary = vocabulary or _vocabulary.VOCABULARY
    tl = _vocabulary.entry_timeline(e, vocabulary)
    return "%s.%s" % (tl, e["kind"]) if vocabulary["families"][e["family"]]["type"] == "bs_tension" else tl


def tendency_mask(block, n_frames: int, hz: float) -> np.ndarray:
    """(T, 7) bool: which tendencies are specified on each frame (Supp. C "input": distinguishes
    "specified as neutral" from "left unspecified")."""
    m = np.zeros((n_frames, len(_tendency.PARAMS)), dtype=bool)
    if block is None:
        return m
    for k, p in enumerate(_tendency.PARAMS):
        if p in block.get("default", {}):
            m[:, k] = True
    for s in block.get("segments", []):
        a, b = _vocabulary.tau_frames(s["t"], hz)
        for k, p in enumerate(_tendency.PARAMS):
            if p in s:
                m[a:b, k] = True
    return m


def state_block(text: dict, decoded: dict, names: list, codec: dict, vocabulary: dict = None) -> tuple:
    """(state curves (T, 17), left-right asymmetry (T, 8)) of decoded notes: the per-timeline state curves,
    with mouth tension split by kind (Sec. 3.4 input; Supp. C "input")."""
    vocabulary = vocabulary or _vocabulary.VOCABULARY
    hz, n = float(text["hz"]), int(text["n_frames"])
    col = {tl: i for i, tl in enumerate(decoded["state_names"])}
    slots, aslots = state_slots(vocabulary), asym_slots(vocabulary)
    state = np.zeros((n, len(slots)))
    sig = _vocabulary.signal_names(names)
    for j, (_, tl, fam, kind) in enumerate(slots):
        if kind is None:
            state[:, j] = decoded["state"][:, col[tl]]
            continue
        es = [e for e in text["state"] if e["family"] == fam and e.get("kind") == kind]
        if es:
            state[:, j] = _decoder.decode_state(es, sig, n, hz, codec["fitted"], vocabulary)[1][tl]
    asym = np.zeros((n, len(aslots)))
    ai = {s: k for k, s in enumerate(aslots)}
    for e in sorted(text["state"], key=lambda e: _vocabulary.tau_frames(e["t"], hz)[0]):
        k = ai.get(entry_slot(e, vocabulary))
        if k is not None:
            a, _, _, b = _vocabulary.tau_frames(e["t"], hz)
            asym[a:b, k] = _vocabulary._entry_asym(e, vocabulary)
    return state, asym


def apply_header(text: dict, policy: str, prior: dict = None) -> dict:
    """The notes under a header policy; the paper uses "static_ref": the clip header as written."""
    if policy == "static_ref":
        return text
    if policy != "prior":
        raise ValueError("policy %r" % policy)
    if prior is None:
        raise ValueError("policy prior needs the training-prior header")
    return dict(text, rest=copy.deepcopy(prior["rest"]), reference=copy.deepcopy(prior["reference"]))


def notes_conditions(text: dict, names: list, policy: str = PRIMARY_POLICY, prior: dict = None, codec: dict = None) -> dict:
    """The note part of the condition c(N, w) (Sec. 3.4 "input"; Supp. C "input"): decode N (Eq. (2), (3)) and
    return {"X_obs", "M", "rest", "rot_curve", "reference", "delta", "state", "asym", "state_on", "tend",
    "tend_mask"}.  Training and inference call this one function; it reads nothing but the notes."""
    codec = codec or load_constants()
    text = apply_header(text, policy, prior)
    d = _decoder.decode(text, names, codec["rest_stat"], codec["fitted"], tendency_q=codec["tendency_q"])
    state, asym = state_block(text, d, names, codec)
    return {"X_obs": d["X_obs"], "M": d["M"], "rest": d["rest_row"], "rot_curve": d["rot_curve"],
            "reference": d["reference"], "delta": d["delta"], "state": state.astype(np.float32),
            "asym": asym.astype(np.float32), "state_on": np.array(1.0 if text["state"] else 0.0, dtype=np.float32),
            "tend": d["tendency"].astype(np.float32),
            "tend_mask": tendency_mask(text["tendency"], int(text["n_frames"]), float(text["hz"]))}


def projection_input(c: dict) -> dict:
    """The arguments of Pi_delta (Eq. (6)) rebuilt from condition arrays: {"M", "x" (= X_obs on M), "delta",
    "rot_curve", "reference"}."""
    return {"M": c["M"], "x": c["X_obs"], "delta": c["delta"], "rot_curve": c["rot_curve"], "reference": c["reference"]}


def normalise_conditions(c: dict, norm: dict) -> dict:
    """Network-ready condition arrays: X_obs = M (x - mean) / std with 0 where M = 0, rest likewise, state
    curves / training rms, tendency values standardised where specified (Supp. C "diffusion training")."""
    mean, std = np.asarray(norm["x_mean"]), np.asarray(norm["x_std"])
    m = np.asarray(c["M"], dtype=bool)
    xn = np.where(m, (np.where(m, c["X_obs"], 0.0) - mean) / std, 0.0)
    tm = np.asarray(c["tend_mask"], dtype=bool)
    tn = np.where(tm, (np.asarray(c["tend"], dtype=np.float64) - np.asarray(norm["tend_mean"])) / np.asarray(norm["tend_std"]),
                  0.0)
    n = m.shape[0]
    return {"X_obs": xn.astype(np.float32), "M": m.astype(np.float32),
            "rest": ((np.asarray(c["rest"]) - mean) / std).astype(np.float32),
            "state": (np.asarray(c["state"], dtype=np.float64) / np.asarray(norm["state_rms"])).astype(np.float32),
            "asym": np.asarray(c["asym"], dtype=np.float32),
            "state_on": np.full((n, 1), float(c["state_on"]), dtype=np.float32),
            "tend": tn.astype(np.float32), "tend_mask": tm.astype(np.float32)}


def sparsify(text: dict, k: int, c: int, tier: str, params: dict, codec: dict, quota_u: float = None) -> dict:
    """strata.sparsify.sparsify with the paper's settings (amplitude, quota and state-cap modes); quota_u: the
    per-clip uniform of the stochastic quota (None: derived from the clip id, deterministic)."""
    return _sparsify.sparsify(text, k, c, tier, fitted=codec["fitted"], amplitude=params["amplitude"], quota=params["quota"],
                       state_cap=params["state_cap"], quota_u=quota_u)


def validation_densities(full: dict, params: dict, codec: dict) -> list:
    """The three densities of one clip (Sec. 3.2; Supp. A "densities"): full notes, standard notes (at most 30
    events per minute, 6 state segments per family and minute) and lite notes (6 and 2)."""
    return [(_notes.strip_text(full), {"tier": "full"}),
            (sparsify(full, params["k_std"], params["c_std"], "standard", params, codec),
             {"tier": "standard", "K": params["k_std"], "C": params["c_std"]}),
            (sparsify(full, params["k_lite"], params["c_lite"], "lite", params, codec),
             {"tier": "lite", "K": params["k_lite"], "C": params["c_lite"]})]


def content_words(tt: dict) -> tuple:
    """(sentences, words) of a script for the encoder: sentence frame ranges and word frame ranges."""
    words = tt["words"]
    sentences = _script.sentences_from_words(words, _script.SENTENCE_END) if words else []
    return sentences, [(int(w["start_frame"]), int(w["end_frame"])) for w in words]


def encode_full_notes(track: dict, tt: dict, names: list, hz: float, codec: dict) -> dict:
    sentences, words = content_words(tt)
    return _encoder.encode(track, names, hz, codec["rest_stat"], codec["fitted"], sentences=sentences, words=words,
                          tendency_q=codec["tendency_q"])


STATE_PARTS = ("state", "asym", "state_on")  # normalise_conditions keys, input order


TENDENCY_PARTS = ("tend", "tend_mask")


def notes_block_parts(c: dict) -> tuple:
    """The keys of the notes block of the model input, in order (Supp. C "input")."""
    v = c["l2v2"]
    return (STATE_PARTS if v["state"] else ()) + (TENDENCY_PARTS if v["tendency"] else ())


def notes_block(c: dict, cn: dict) -> np.ndarray:
    """(T, 40) float32 notes block of normalised condition arrays: state curves (17), asymmetry (8), state
    flag (1), tendency values (7), tendency mask (7)."""
    return np.concatenate([np.asarray(cn[k], dtype=np.float32).reshape(len(cn["M"]), -1) for k in notes_block_parts(c)], axis=1)


def drop_text(text: np.ndarray, letters: np.ndarray) -> tuple:
    """The empty script: all timing features and viseme ids 0."""
    return np.zeros_like(text), np.zeros_like(letters)


def null_notes(x_obs: np.ndarray, m: np.ndarray, notes_in: np.ndarray, rest_null: np.ndarray) -> tuple:
    """The empty-notes condition: X_obs, M and the notes block 0, rest = the training-prior neutral row."""
    return (np.zeros_like(x_obs), np.zeros_like(m), np.zeros_like(notes_in),
            np.asarray(rest_null, dtype=np.float32).copy())


def model_inputs(c: dict, norm: dict, cond: dict, text: np.ndarray, lex: np.ndarray, rest_null: np.ndarray,
              no_text: bool = False, no_l2: bool = False) -> dict:
    """The model input of one condition (Sec. 3.4 "input"): normalised X_obs, M, rest, script timing features
    and viseme ids, and the 40-wide block of state / asymmetry / state flag / tendency values / tendency mask."""
    cn = normalise_conditions(cond, norm)
    xo, m, rest, notes_in = cn["X_obs"], cn["M"], cn["rest"], notes_block(c, cn)
    tx, lx = np.asarray(text, dtype=np.float32), np.asarray(lex, dtype=np.int64)
    if c["paths"]["G8"]:
        lx = np.zeros_like(lx)
    if no_text or c["arm"] == "l2_only":
        tx, lx = drop_text(tx, lx)
    if no_l2 or c["arm"] == "text_only":
        mean, std = np.asarray(norm["x_mean"]), np.asarray(norm["x_std"])
        xo, m, notes_in, rest = null_notes(xo, m, notes_in, ((np.asarray(rest_null) - mean) / std).astype(np.float32))
    return {"x_obs": xo, "m": m, "rest": rest, "text": tx, "letters": lx, "v2": notes_in}
