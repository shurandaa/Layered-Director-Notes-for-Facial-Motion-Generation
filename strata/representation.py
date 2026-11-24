"""Motion representation (paper Sec. 3.1).

A clip is a 30 Hz sequence X = (x_1, ..., x_T) of 63-D frames x_t = [b_t, p_t, r_t, g_t]: 52 ARKit blendshapes b_t
(columns 0-51), head translation p_t (52-54), head rotation r_t in the 6D representation (55-60) and gaze g_t
(vertical, horizontal iris offset; 61-62).  The director language expresses head rotation as
omega_t = (180 / pi) log(R_ref^T R(r_t))^vee, in degrees about the clip's reference orientation R_ref.
"""
import hashlib
import json
import numpy as np
import os
from dataclasses import dataclass
from typing import Tuple


SCHEMA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "geometry.schema.json")


@dataclass(frozen=True)
class ChannelSpec:
    """One channel's contract, mirrored verbatim from the JSON."""
    name: str
    start: int
    end: int
    dim: int
    track: str
    source_column: object
    quality_column: str
    units: str
    input_normalization: dict
    target_normalization: dict
    decoder_head: dict
    projection_dim: int
    loss: dict

    @property
    def slice(self) -> slice:
        return slice(self.start, self.end)


def _load_schema(path: str = SCHEMA_PATH) -> dict:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _build_channels(raw: dict) -> Tuple[ChannelSpec, ...]:
    channels = []
    cursor = 0
    for entry in raw["channels"]:
        if not entry.get("enabled", True):
            continue
        start, end = cursor, cursor + entry["dim"]
        cursor = end
        channels.append(ChannelSpec(
            name=entry["name"], start=start, end=end, dim=entry["dim"],
            track=entry["track"], source_column=entry["source_column"],
            quality_column=entry["quality_column"], units=entry["units"],
            input_normalization=entry["input_normalization"],
            target_normalization=entry["target_normalization"],
            decoder_head=entry["decoder_head"],
            projection_dim=entry["projection_dim"], loss=entry["loss"],
        ))
    return tuple(channels)


SCHEMA = _load_schema()


CHANNELS = _build_channels(SCHEMA)


CHANNEL_NAMES = tuple(spec.name for spec in CHANNELS)


TOTAL_DIM = CHANNELS[-1].end if CHANNELS else 0


_BY_NAME = {spec.name: spec for spec in CHANNELS}


def channel(name: str) -> ChannelSpec:
    if name not in _BY_NAME:
        raise KeyError("unknown channel %r; schema defines %s" % (name, list(CHANNEL_NAMES)))
    return _BY_NAME[name]


DEGENERATE_NORM_TOL = 1e-8


def rotation_matrix_to_6d(rotation: np.ndarray) -> np.ndarray:
    """First two columns of R, flattened -- the continuous 6D representation."""
    rotation = np.asarray(rotation, dtype=np.float64)
    return rotation[:, :2].T.reshape(6).astype(np.float32)


def rotation_6d_to_matrix(r6: np.ndarray) -> np.ndarray:
    """Inverse of rotation_matrix_to_6d via Gram-Schmidt.

    Column order matches rotation_matrix_to_6d exactly: that function emits
    R[:, :2].T.reshape(6), i.e. the first column's three components followed by
    the second's, and this reads them back in the same order. The pair is
    round-tripped in tests, because a silent transposition here would corrupt
    every rotation while still producing a valid-looking matrix.

    Degenerate input is handled rather than allowed to divide by zero. A network
    early in training, or an all-zero fill, can easily produce a first vector of
    zero length or two parallel vectors; Gram-Schmidt is undefined for both. Such
    input falls back to the identity rotation, which is a valid rotation and an
    honest "no information", instead of NaNs that propagate into the loss.
    """
    r6 = np.asarray(r6, dtype=np.float64).reshape(2, 3)
    a1, a2 = r6[0], r6[1]
    if not np.isfinite(r6).all():
        return np.eye(3)

    norm1 = np.linalg.norm(a1)
    if norm1 < DEGENERATE_NORM_TOL:
        return np.eye(3)
    b1 = a1 / norm1

    b2 = a2 - np.dot(b1, a2) * b1
    norm2 = np.linalg.norm(b2)
    if norm2 < DEGENERATE_NORM_TOL:
        # a2 is parallel to a1, so it carries no second axis. Pick any direction
        # orthogonal to b1 -- the rotation is underdetermined and this at least
        # stays on the manifold.
        fallback = np.array([1.0, 0.0, 0.0]) if abs(b1[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        b2 = fallback - np.dot(b1, fallback) * b1
        norm2 = np.linalg.norm(b2)
    b2 = b2 / norm2

    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=1)


def _quaternion_from_matrix(matrix: np.ndarray) -> np.ndarray:
    """Rotation matrix -> unit quaternion (w, x, y, z), branch-stable."""
    m = np.asarray(matrix, dtype=np.float64)
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        q = np.array([0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s])
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        q = np.array([(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s])
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        q = np.array([(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s])
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        q = np.array([(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s])
    norm = np.linalg.norm(q)
    return q / norm if norm > 1e-12 else np.array([1.0, 0.0, 0.0, 0.0])


def _matrix_from_quaternion(q: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(q, dtype=np.float64) / max(np.linalg.norm(q), 1e-12)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def loss_weight(observed: np.ndarray, quality: np.ndarray, is_gap_filled: np.ndarray,
                is_padding: np.ndarray, gap_fill_weight: float = 0.0,
                use_quality: bool = True) -> np.ndarray:
    """Combine the four signals into a per-frame, per-channel loss weight.

    Policy, in order of severity:
        padding      -> 0, always. It is not data.
        gap filled   -> gap_fill_weight, 0 by default. Supervising a model on
                        motion that was interpolated teaches it the interpolator.
        not observed -> 0. The value present is a neutral fill, not a measurement.
        otherwise    -> the quality proxy, or 1 when use_quality is off.

    Returns (T, n_channels). Channel granularity matters: a gaze dropout must not
    silence the blendshape supervision for the same frame.
    """
    observed = np.asarray(observed, dtype=bool)
    quality = np.asarray(quality, dtype=np.float64)
    is_gap_filled = np.asarray(is_gap_filled, dtype=bool)
    is_padding = np.asarray(is_padding, dtype=bool)

    weight = quality.copy() if use_quality else np.ones_like(quality)
    weight = np.clip(weight, 0.0, 1.0)
    weight[~observed] = 0.0
    if is_gap_filled.ndim == 1:
        is_gap_filled = is_gap_filled[:, None]
    if is_padding.ndim == 1:
        is_padding = is_padding[:, None]
    weight = np.where(np.broadcast_to(is_gap_filled, weight.shape), gap_fill_weight, weight)
    weight = np.where(np.broadcast_to(is_padding, weight.shape), 0.0, weight)
    return weight


CHANNEL_RULES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "l1_rules.json")


ROTATION_CHANNEL = "headpose_rotation_6d"


TIME_TOLERANCE_FRAMES = 0.05


def _load_channel_rules(path: str = None) -> dict:
    path = path or CHANNEL_RULES_PATH
    with open(path, "rb") as handle:
        raw = handle.read()
    rules = json.loads(raw.decode("utf-8"))
    rules["_source"] = os.path.abspath(path)
    rules["_sha256"] = hashlib.sha256(raw).hexdigest()
    return rules


CHANNEL_RULES = _load_channel_rules()


TIME_DECIMALS = int(CHANNEL_RULES["time"]["decimals"])


def _matrix_from_rotvec_deg(rotvec_deg) -> np.ndarray:

    vector = np.radians(np.asarray(rotvec_deg, dtype=np.float64).reshape(3))
    angle = np.linalg.norm(vector)
    if angle < 1e-12:
        return np.eye(3)
    return _matrix_from_quaternion(np.concatenate([[np.cos(angle / 2.0)], np.sin(angle / 2.0) * vector / angle]))


def _rotvec_deg_from_quaternion(q) -> np.ndarray:
    """Log map. q is put on the w >= 0 hemisphere, which selects the angle in
    [0, 180] degrees and makes q and -q (one rotation) give the same vector."""
    q = np.asarray(q, dtype=np.float64)
    if q[0] < 0.0:
        q = -q
    sin_half = np.linalg.norm(q[1:])
    if sin_half < 1e-12:
        return np.zeros(3)
    angle = 2.0 * np.arctan2(sin_half, q[0])
    return np.degrees(q[1:] / sin_half * angle)


def _quaternions(r6, reference_deg=None) -> list:
    """Sign-aligned unit quaternions of R_ref^T @ R for each 6-vector."""

    inverse_reference = _matrix_from_rotvec_deg(reference_deg if reference_deg is not None else np.zeros(3)).T
    out = []
    for vector in np.asarray(r6, dtype=np.float64).reshape(-1, 6):
        q = _quaternion_from_matrix(inverse_reference @ rotation_6d_to_matrix(vector))
        if out and np.dot(q, out[0]) < 0.0:
            q = -q
        out.append(q)
    return out


def omega_from_r6(r6, reference_deg=None) -> np.ndarray:
    """(T, 6) head rotations r_t -> (T, 3) omega_t in degrees, relative to R_ref (Sec. 3.1):
    omega_t = (180 / pi) log(R_ref^T R(r_t))^vee.  `reference_deg` is R_ref as a rotation vector in degrees."""
    quaternions = _quaternions(r6, reference_deg)
    return np.array([_rotvec_deg_from_quaternion(q) for q in quaternions]).reshape(-1, 3)


def r6_from_omega(rotvec_deg, reference_deg=None) -> np.ndarray:
    """(T, 3) omega_t (degrees, relative to R_ref) -> (T, 6) r_t, via the exponential map (inverse of
    omega_from_r6)."""

    reference = _matrix_from_rotvec_deg(reference_deg if reference_deg is not None else np.zeros(3))
    rotvec = np.asarray(rotvec_deg, dtype=np.float64).reshape(-1, 3)
    return np.stack([rotation_matrix_to_6d(reference @ _matrix_from_rotvec_deg(v)) for v in rotvec]).astype(np.float64)


def mean_omega(r6, reference_deg=None) -> np.ndarray:
    """Mean rotation (relative to the reference) as a rotation vector.

    The normalised mean of hemisphere-aligned quaternions is the chordal L2 mean
    on the rotation manifold -- the multi-sample form of the nlerp used by
    src/sequence/resample.py -- rather than an element-wise mean of 6-vectors.
    """
    mean = np.mean(_quaternions(r6, reference_deg), axis=0)
    return _rotvec_deg_from_quaternion(mean / max(np.linalg.norm(mean), 1e-12))


def reference_orientation(r6, described) -> np.ndarray:
    """The reference orientation R_ref of the clip header (Sec. 3.2; Supp. B "clip header"): the mean head
    orientation over the validly tracked frames, as a rotation vector in degrees, rounded as the notes store it."""
    reference = CHANNEL_RULES["channels"][ROTATION_CHANNEL]["reference"]
    described = np.asarray(described, dtype=bool)
    if reference["method"] == "identity" or not described.any():
        return np.zeros(3)
    if reference["method"] != "window_mean":
        raise ValueError("unknown rotation reference method %r" % reference["method"])
    return np.round(mean_omega(np.asarray(r6)[described]), int(reference["decimals"]))


def tracked_valid(observed, quality, is_gap_filled, is_padding) -> np.ndarray:
    """(T, n_channels) bool: frames with valid tracking for each channel group (observed, not gap-filled, not
    padding)."""
    return loss_weight(observed, quality, is_gap_filled, is_padding) > 0.0


def frame_to_time(frame: int, hz: float) -> float:
    return round(frame / hz, TIME_DECIMALS)


def time_to_frame(t: float, hz: float) -> int:
    exact = float(t) * hz
    frame = int(np.floor(exact + 0.5))
    if abs(exact - frame) > TIME_TOLERANCE_FRAMES:
        raise ValueError("time %s does not fall on the %.3f Hz grid" % (t, hz))
    return frame


ROTATION_TEXT = CHANNEL_RULES["channels"][ROTATION_CHANNEL]["text_name"]


ROTATION_AXES = tuple(CHANNEL_RULES["channels"][ROTATION_CHANNEL]["axes"])


def priors_sha256(priors: dict) -> str:
    """sha256 of the priors' canonical JSON (sorted keys): the version id recorded next to every decode."""
    return hashlib.sha256(json.dumps(priors, sort_keys=True).encode("utf-8")).hexdigest()
