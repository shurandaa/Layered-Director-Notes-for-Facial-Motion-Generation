"""Decoder (paper Sec. 3.3 "Decoder", Eq. (2), (3); Supp. B) and the projection Pi_delta (Eq. (6)).

x_hat_t = rest + sum_s v_s phi_tau_s(t) u_f_s + sum_e A_e y_e(t) u_f_e; the cells covered by events form the
constraint mask M and X_obs = x_hat (.) M.  Rule-based and deterministic; reads only the notes.
"""
import numpy as np
from strata import notes as _notes
from strata import representation as _representation
from strata import tendency as _tendency
from strata import vocabulary as _vocabulary


def psi(u: np.ndarray) -> np.ndarray:
    """psi(u) = 3u^2 - 2u^3 on u clipped to [0, 1] (Supp. B, decoder)."""
    u = np.clip(u, 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)


def phi_trapezoid(taus: list, values: list, n: int) -> np.ndarray:
    """phi_tau(t) of a timeline (Supp. B, decoder): each state is a trapezoid with plateau value v over
    [tau1, tau2), rising from the previous plateau (if both share the transition) or 0 over [tau0, tau1) and falling
    to 0 over [tau2, tau3) with psi; taus = [(tau0, tau1, tau2, tau3)] in frames.  Frame t of a transition [a, b)
    sits at u = (t - a + 1) / (b - a + 1)."""
    y = np.zeros(n)
    for i, ((s0, s1, r0, r1), v) in enumerate(zip(taus, values)):
        prev = values[i - 1] if i > 0 and taus[i - 1][2] == s0 and taus[i - 1][3] == s1 else 0.0
        f = np.arange(s0, s1)
        y[s0:s1] = prev + (v - prev) * psi((f - s0 + 1) / (s1 - s0 + 1))
        y[s1:r0] = v
        shared_next = i + 1 < len(taus) and taus[i + 1][0] == r0 and taus[i + 1][1] == r1
        if not shared_next:
            f = np.arange(r0, r1)
            y[r0:r1] = v * (1.0 - psi((f - r0 + 1) / (r1 - r0 + 1)))
    return y


def _state_value(e: dict, tl: str, fitted: dict, vocabulary: dict) -> float:
    """v_s of a state: its value, or (levels only) the representative training-set value of its level
    (Sec. 3.3) with the sign of its direction."""
    if "value" in e:
        return float(e["value"])
    spec = vocabulary["families"][e["family"]]
    axis = tl.split(".", 1)[1] if "." in tl else None
    rep = fitted["state_repr"][tl][vocabulary["state"]["levels"].index(e["level"])]
    return _vocabulary._entry_sign(e, spec, axis) * rep


def _member_values(fam: str, e: dict, v: float, loadings: dict, vocabulary: dict) -> dict:
    """Per-channel deviations for family value v: v * u_f (the calibrated channel weights), with the
    left / right split of the asymmetry (Supp. B "states")."""
    spec = vocabulary["families"][fam]
    if spec["type"] == "bs_tension":
        load = loadings[fam][e["kind"]]
    elif spec["type"] == "bs_diff":
        load = loadings[fam]["plus" if v >= 0 else "minus"]
    else:
        load = loadings[fam]
    out = {c: v * load[c] for c in spec["channels"]}
    pair = _vocabulary._pair_of(fam, e, vocabulary)
    if pair:
        lp, a = 0.5 * (load[pair[0]] + load[pair[1]]), _vocabulary._entry_asym(e, vocabulary)
        out[pair[0]], out[pair[1]] = v * lp * (1.0 + a), v * lp * (1.0 - a)
    return out


def decode_state(entries: list, names: list, n: int, hz: float, fitted: dict, vocabulary: dict = None) -> tuple:
    """The state part of Eq. (2): (per-channel deviations sum_s v_s phi_tau_s(t) u_f, per-timeline state
    curves)."""
    vocabulary = vocabulary or _vocabulary.VOCABULARY
    ix = {nm: k for k, nm in enumerate(names)}
    dev = np.zeros((n, len(names)))
    curves = {}
    by_tl = {tl: [] for tl, _, _ in _vocabulary.timelines(vocabulary)}
    for e in entries:
        by_tl[_vocabulary.entry_timeline(e, vocabulary)].append(e)
    for tl, fam, axis in _vocabulary.timelines(vocabulary):
        es = sorted(by_tl[tl], key=lambda e: e["t"][0])
        spans = [_vocabulary.tau_frames(e["t"], hz) for e in es]
        vs = [_state_value(e, tl, fitted, vocabulary) for e in es]
        curves[tl] = phi_trapezoid(spans, vs, n)
        spec = vocabulary["families"][fam]
        if spec["type"] == "vector":
            dev[:, ix[_vocabulary._axis_signal(spec, axis)]] += curves[tl]
            continue
        members = [_member_values(fam, e, v, fitted["state_loadings"], vocabulary) for e, v in zip(es, vs)]
        for c in spec["channels"]:
            dev[:, ix[c]] += phi_trapezoid(spans, [m[c] for m in members], n)
    return dev, curves


def y_event(n: int, tau0: int, tau1: int, tau2: int, tau3: int, b: float, h_f: float, kappa_up: float,
                     kappa_down: float) -> np.ndarray:
    """y_e(t), the pedestal-plus-exponential-peak template of one event unit (Supp. B, decoder):
    y_e = P + (1 - P) exp(-d / kappa), P = b + (h_f - b) psi(u), d = frames to the apex, kappa_up on the rising side
    and kappa_down on the falling side; 1 on the apex [tau1, tau2), 0 outside [tau0, tau3)."""
    y = np.zeros(n)
    f = np.arange(tau0, tau1)
    P = b + (h_f - b) * psi((f - tau0 + 1) / (tau1 - tau0 + 1))
    y[tau0:tau1] = P + (1.0 - P) * np.exp(-(tau1 - f) / float(kappa_up))
    y[tau1:tau2] = 1.0
    f = np.arange(tau2, tau3)
    P = b + (h_f - b) * psi((tau3 - f) / (tau3 - tau2 + 1))
    y[tau2:tau3] = P + (1.0 - P) * np.exp(-(f - tau2 + 1) / float(kappa_down))
    return y


def event_template_params(fam: str, fitted: dict) -> tuple:
    """(rise, fall, base rule) recorded for a family's event template (the "peak" form uses h_f, kappa_up,
    kappa_down from the calibrated constants)."""
    sh = (fitted or {}).get("event_shape", {}).get(fam)
    return (sh["rise"], sh["fall"], sh.get("base")) if sh else (None, None, None)


def _event_thresholds(fam: str, kind, vocabulary: dict) -> tuple:
    """(theta_low, theta_high) of the family's event detector (mouth tension: of the kind)."""
    spec = vocabulary["families"][fam]
    ev = spec["kinds"][kind] if spec["type"] == "bs_tension" else spec["event"]
    return ev["dev_low"], ev["dev_high"]


def b_e(fam: str, kind, amplitude: float, vocabulary: dict = None) -> float:
    """b_e = min(theta_low / |A_e|, theta_low / theta_high): the pedestal start of y_e, so that decoding
    and detection agree at the onset (Supp. B, decoder)."""
    theta_low, theta_high = _event_thresholds(fam, kind, vocabulary or _vocabulary.VOCABULARY)
    return float(min(theta_low / abs(amplitude), theta_low / theta_high)) if amplitude else theta_low / theta_high


def _event_amplitude(e: dict, fitted: dict, vocabulary: dict) -> float:
    """A_e of an event: its value (vector: the component of the event axis), or the representative
    training-set value of its level."""
    fam = e["family"]
    spec = vocabulary["families"][fam]
    v = e.get("value")
    if spec["type"] == "vector":
        axis = _vocabulary.entry_timeline(e, vocabulary).split(".", 1)[1]
        if isinstance(v, list):
            return float(v[spec["axes"].index(axis)])
        core = {"kind": e.get("kind"), "axis": axis}
    else:
        core = e
    if v is None:
        v = fitted["event_repr"][_vocabulary.event_level_key(fam, core)][vocabulary["event"]["levels"].index(e["level"])]
    return float(v)


def _event_values(e: dict, fitted: dict, vocabulary: dict) -> dict:
    """channel -> peak deviation A_e u_f of one event."""
    fam = e["family"]
    spec = vocabulary["families"][fam]
    if spec["type"] == "vector":
        axis = _vocabulary.entry_timeline(e, vocabulary).split(".", 1)[1]
        v = e.get("value")
        if isinstance(v, list):
            return {_vocabulary._axis_signal(spec, ax): float(x) for ax, x in zip(spec["axes"], v)}
        if v is None:
            core = {"kind": e.get("kind"), "axis": axis}
            v = _vocabulary._entry_sign(e, spec, axis) * fitted["event_repr"][_vocabulary.event_level_key(fam, core)][
                vocabulary["event"]["levels"].index(e["level"])]
        return {_vocabulary._axis_signal(spec, axis): float(v)}
    v = e.get("value")
    if v is None:
        v = fitted["event_repr"][_vocabulary.event_level_key(fam, e)][vocabulary["event"]["levels"].index(e["level"])]
    return _member_values(fam, e, float(v), fitted["event_loadings"], vocabulary)


def decode_events(events: list, names: list, n: int, hz: float, fitted: dict, vocabulary: dict = None) -> tuple:
    """The event part of Eq. (2): (per-channel deviations sum_e A_e y_e(t) u_f, {family: frames covered
    by its events})."""
    vocabulary = vocabulary or _vocabulary.VOCABULARY
    ix = {nm: k for k, nm in enumerate(names)}
    dev = np.zeros((n, len(names)))
    frames = {fam: np.zeros(n, dtype=bool) for fam in vocabulary["families"]}
    for e in events:
        vals = _event_values(e, fitted, vocabulary)
        _, _, rule = event_template_params(e["family"], fitted)
        b = b_e(e["family"], e.get("kind"), _event_amplitude(e, fitted, vocabulary), vocabulary) \
            if rule == "dev_low" else 0.0
        sh = (fitted or {}).get("event_shape", {}).get(e["family"]) or {}
        for unit in _vocabulary.event_units(e, hz):
            s, a0, a1, end = unit
            shape = y_event(n, *unit, b, sh["h"], sh["tau_up"], sh["tau_dn"])
            frames[e["family"]][unit[0]:unit[3]] = True
            for sig, v in vals.items():
                dev[:, ix[sig]] += v * shape
    return dev, frames


def _gaze_to_eyelook(dev: np.ndarray, names: list, fitted: dict, vocabulary: dict) -> None:
    """In place: the eyeLook* deviations from the gaze deviations through the calibrated 8 x 2 map
    (Supp. B "properties and constants")."""
    spec = vocabulary["families"]["gaze"]
    ix = {nm: k for k, nm in enumerate(names)}
    g = dev[:, [ix[_vocabulary._axis_signal(spec, a)] for a in spec["axes"]]]
    dev[:, [ix[m] for m in spec["members"]]] = g @ np.asarray(fitted["eyelook_map"]).T


def decode(track: dict, blendshape_names: list, rest_stat: str = None, fitted: dict = None,
                 vocabulary: dict = None, tendency_q: dict = None) -> dict:
    """Decode a set of director notes N (Eq. (1)) without any other input (Sec. 3.3, Eq. (2), (3)).

    Returns {"x": x_hat (T, 63) = rest + sum_s v_s phi(t) u_f + sum_e A_e y_e(t) u_f (head rotation summed in
    rotation-vector space about R_ref), "M": the constraint mask of Eq. (3), "X_obs": x_hat on M and NaN elsewhere,
    "state": (T, 14) per-timeline state curves, "rest_row": the clip header's neutral values as a 63-D row, "delta":
    delta_c, "rot_curve": the decoded head rotation omega (degrees), "reference": R_ref, "tendency": (T, 7) the
    tendency values per frame}.  The notes are validated first (strata.notes.validate_track)."""
    vocabulary = vocabulary or _vocabulary.VOCABULARY
    _notes.validate_track(track, vocabulary)
    fitted = fitted if fitted is not None else _vocabulary.calibration_for(rest_stat)
    hz, n = float(track["hz"]), int(track["n_frames"])
    names = _vocabulary.signal_names(blendshape_names)
    rest = _vocabulary.rest_from_dict(track["rest"], names)
    sdev, curves = decode_state(track["state"], names, n, hz, fitted, vocabulary)
    edev, ev_frames = decode_events(track["event"], names, n, hz, fitted, vocabulary)
    dev = sdev + edev
    _gaze_to_eyelook(dev, names, fitted, vocabulary)
    y = rest + dev
    reference = np.array([float(track["reference"][_vocabulary.ROTATION_TEXT][a]) for a in _vocabulary.ROTATION_AXES])
    cols = _vocabulary.signal_columns(blendshape_names)
    ix = {nm: k for k, nm in enumerate(names)}
    x_hat = np.zeros((n, _representation.TOTAL_DIM))
    bs = _representation.channel("blendshape")
    for k, name in enumerate(blendshape_names):
        if name in _vocabulary.EXCLUDED:
            x_hat[:, bs.start + k] = _representation.CHANNEL_RULES["channels"]["blendshape"]["excluded_fill"]
    rot_names = ["%s.%s" % (_vocabulary.ROTATION_TEXT, a) for a in _vocabulary.ROTATION_AXES]
    for name in names:
        if name not in rot_names:
            x_hat[:, cols[name][0]] = y[:, ix[name]]
    rot_curve = y[:, [ix[r] for r in rot_names]]
    x_hat[:, _representation.channel(_representation.ROTATION_CHANNEL).slice] = _representation.r6_from_omega(rot_curve, reference)
    M = np.zeros((n, _representation.TOTAL_DIM), dtype=bool)
    for fam, fr in ev_frames.items():
        if fr.any():
            M[np.ix_(fr, _vocabulary.family_columns(fam, blendshape_names, vocabulary))] = True
    rest_row = x_hat[0].copy() if n else np.zeros(_representation.TOTAL_DIM)
    for name in names:
        if name not in rot_names:
            rest_row[cols[name][0]] = rest[ix[name]]
    rest_row[_representation.channel(_representation.ROTATION_CHANNEL).slice] = _representation.r6_from_omega(rest[[ix[r] for r in rot_names]][None], reference)[0]
    tl_names = [tl for tl, _, _ in _vocabulary.timelines(vocabulary)]
    tend = _tendency.decode_tendency(track["tendency"], n, hz, tendency_q, rest_stat)
    return {"x": x_hat, "X_obs": np.where(M, x_hat, np.nan), "M": M, "state": np.stack([curves[t] for t in tl_names], axis=1),
            "state_names": tl_names, "rest_row": rest_row, "delta": _vocabulary.delta_c_row(blendshape_names, vocabulary),
            "rot_curve": rot_curve, "reference": reference,
            "tendency": tend["values"], "tendency_names": tend["names"], "tendency_dropped": tend["dropped"]}


def Pi_delta(y: np.ndarray, decoded: dict) -> np.ndarray:
    """Pi_delta of Eq. (6): every constrained entry (M = 1) of y clipped to x_hat +- delta_c; head rotation
    clipped per axis in rotation-vector space about R_ref and mapped back (Supp. C, sampling).  Entries with M = 0
    are returned unchanged."""
    M, x_hat, delta = decoded["M"], decoded["x"], decoded["delta"]
    y_proj = np.array(y, dtype=np.float64, copy=True)
    rot = _representation.channel(_representation.ROTATION_CHANNEL).slice
    flat = M.copy()
    flat[:, rot] = False
    d = np.broadcast_to(delta, y_proj.shape)
    y_proj[flat] = np.clip(y_proj[flat], (x_hat - d)[flat], (x_hat + d)[flat])
    rows = np.flatnonzero(M[:, rot].any(axis=1))
    if rows.size:
        dr = float(delta[rot][0])
        ref = decoded["reference"]
        rv = np.stack([_representation.omega_from_r6(y_proj[r, rot][None], ref)[0] for r in rows])
        c = decoded["rot_curve"][rows]
        y_proj[rows, rot] = _representation.r6_from_omega(np.clip(rv, c - dr, c + dr), ref)
    return y_proj
