"""Sampling with hard constraints by projection (paper Sec. 3.4 "sampling", Eq. (6); Supp. C).
"""
import dataclasses
import numpy as np
import torch
from strata import conditions as _conditions
from strata import decoder as _decoder
from strata import network as _network
from strata import representation as _representation


def window_starts(n: int, window: int) -> list:
    """Start frames of the overlapping training-length windows that cover n frames (Supp. C "long
    sequences")."""
    if n <= window:
        return [0]
    stride = window // 2
    starts = list(range(0, n - window + 1, stride))
    if starts[-1] != n - window:
        starts.append(n - window)
    return starts


def window_weights(window: int) -> np.ndarray:
    """Triangular blending weights of one window (largest at the centre; Supp. C "long sequences")."""
    i = np.arange(window)
    return np.minimum(i + 1, window - i).astype(np.float64)


WINDOWS = {"W96": 96, "W192": 192, "Wfull": 960}


def window_len(c: dict) -> int:
    return WINDOWS[c["window"]]


class Space:
    """The diffusion variable of one clip: x = base + d * scale (base = training mean, scale = training std),
    so d_0 is the per-channel normalised motion (Sec. 3.4 "training")."""

    def __init__(self, base: np.ndarray, scale: np.ndarray):
        self.base, self.scale = np.asarray(base, dtype=np.float64), np.asarray(scale, dtype=np.float64)

    def to_var(self, x: np.ndarray) -> np.ndarray:
        return (np.asarray(x, dtype=np.float64) - self.base) / self.scale

    def to_x(self, d: np.ndarray) -> np.ndarray:
        return self.base + np.asarray(d, dtype=np.float64) * self.scale


def space(c: dict, norm: dict, x_dec: np.ndarray, m: np.ndarray, rest: np.ndarray, excluded: list) -> Space:
    """The diffusion variable of one clip: base = training mean, scale = training std."""
    T = len(m)
    return Space(np.broadcast_to(np.asarray(norm["x_mean"], dtype=np.float64), (T, _network.D)), np.asarray(norm["x_std"]))


def finish(y: np.ndarray, x_dec: np.ndarray, m: np.ndarray, rest: np.ndarray, excluded: list) -> np.ndarray:
    """Clean-up of a raw sample on unconstrained entries: blendshapes clipped to [0, 1], 6D rotation
    re-orthonormalised, excluded channels set to rest."""
    y = np.array(y, dtype=np.float64, copy=True)
    m = np.asarray(m, dtype=bool)
    unc = ~m
    bs = np.clip(y[:, _network.BS], 0.0, 1.0)
    y[:, _network.BS] = np.where(unc[:, _network.BS], bs, y[:, _network.BS])
    rot = _network.r6_orthonormal(torch.as_tensor(y[:, _network.ROT], dtype=torch.float64)).numpy()
    y[:, _network.ROT] = np.where(unc[:, _network.ROT], rot, y[:, _network.ROT])
    if len(excluded):
        y[:, list(excluded)] = np.asarray(rest, dtype=np.float64)[list(excluded)]
    return np.where(m, np.asarray(x_dec, dtype=np.float64), y)


class PiDeltaProjection:
    """Pi_delta (Eq. (6)) applied to the predicted clean motion d0_hat at every DDIM step, in the diffusion
    variable of one clip: constrained entries clamped to (x_hat +- delta - base) / scale; head-rotation rows with a
    constrained entry clipped per axis in rotation-vector space about R_ref."""

    def __init__(self, sp: Space, proj: dict):
        m = np.asarray(proj["M"], dtype=bool)
        flat = m.copy()
        flat[:, _network.ROT] = False
        delta = np.broadcast_to(np.asarray(proj["delta"], dtype=np.float64), m.shape)
        if not np.isfinite(delta[m]).all():
            raise ValueError("BoundedVar: no delta on an M cell")
        if np.any(sp.scale <= 0):
            raise ValueError("BoundedVar: the variable scale must be > 0")
        curve = np.where(m, np.asarray(proj["x"], dtype=np.float64), 0.0)
        self.lo = np.where(flat, (curve - delta - sp.base) / sp.scale, -np.inf)
        self.hi = np.where(flat, (curve + delta - sp.base) / sp.scale, np.inf)
        self.rows = np.flatnonzero(m[:, _network.ROT].any(axis=1))
        self.sp, self.ref = sp, np.asarray(proj["reference"], dtype=np.float64)
        self.rot_lo = self.rot_hi = None
        if self.rows.size:
            dr = float(delta[self.rows[0], _network.ROT.start])
            c = np.asarray(proj["rot_curve"], dtype=np.float64)[self.rows]
            self.rot_lo, self.rot_hi = c - dr, c + dr
        self._t = {}

    def numpy(self, d: np.ndarray) -> np.ndarray:
        out = np.clip(np.asarray(d, dtype=np.float64), self.lo, self.hi)
        if self.rows.size:
            r = self.rows
            x = self.sp.base[r][:, _network.ROT] + out[r][:, _network.ROT] * self.sp.scale[_network.ROT]
            rv = np.clip(_representation.omega_from_r6(x, self.ref), self.rot_lo, self.rot_hi)
            r6 = _representation.r6_from_omega(rv, self.ref)
            out[np.ix_(r, np.arange(_network.ROT.start, _network.ROT.stop))] = (r6 - self.sp.base[r][:, _network.ROT]) / self.sp.scale[_network.ROT]
        return out

    def torch(self, x0: torch.Tensor) -> torch.Tensor:
        """The projection of a float32 x0 (n, 63) on its device (the clamp in torch, the rotation rows via numpy)."""
        key = (str(x0.device), x0.dtype)
        if key not in self._t:
            self._t[key] = (torch.as_tensor(self.lo, dtype=x0.dtype, device=x0.device),
                            torch.as_tensor(self.hi, dtype=x0.dtype, device=x0.device))
        lo, hi = self._t[key]
        out = torch.maximum(torch.minimum(x0, hi), lo)
        if self.rows.size:
            r = torch.as_tensor(self.rows, device=x0.device)
            rot = self.numpy(_rows_full(out, self.rows))[self.rows][:, _network.ROT]
            out = out.clone()
            out[r, _network.ROT.start:_network.ROT.stop] = torch.as_tensor(rot, dtype=x0.dtype, device=x0.device)
        return out


def _rows_full(x0: torch.Tensor, rows: np.ndarray) -> np.ndarray:
    """A float64 (n, 63) array holding x0's rows `rows` (others 0): the input of BoundedVar.numpy for those rows."""
    a = np.zeros(tuple(x0.shape), dtype=np.float64)
    a[rows] = x0[torch.as_tensor(rows, device=x0.device)].double().cpu().numpy()
    return a


def finish_bounded(y: np.ndarray, proj: dict, rest: np.ndarray, excluded: list) -> np.ndarray:
    """Output of the sampler: `finish` on every entry, then Pi_delta (Eq. (6)) in float64, so every constrained
    entry is within delta_c of the decoded event."""
    m0 = np.zeros(np.shape(y), dtype=bool)
    y0 = finish(y, np.zeros(np.shape(y)), m0, rest, excluded)
    return _decoder.Pi_delta(y0, proj)


@dataclasses.dataclass
class Resources:
    """The trained generator and its training statistics (normalisation, excluded columns, the empty-notes
    neutral row)."""
    model: torch.nn.Module
    config: dict
    norm: dict
    excluded: list
    rest_null: np.ndarray
    decode: object = None  # unused (None)
    names: list = None
    unit: str = None
    device: str = "cpu"
    batch: int = 256

    @property
    def letters(self) -> bool:
        return not self.config["paths"]["G8"]


def make_branch(res: Resources, cond: dict, text, lex, no_text: bool, no_l2: bool) -> dict:
    """One model input and its diffusion variable for a condition (strata.conditions.model_inputs, the
    function training uses too); the empty-notes branch uses the training-prior neutral row."""
    c = res.config
    out = _conditions.model_inputs(c, res.norm, cond, text, lex, res.rest_null, no_text, no_l2)
    l2_off = no_l2 or c["arm"] == "text_only"
    rest_eff = res.rest_null if l2_off else np.asarray(cond["rest"], dtype=np.float64)
    out["space"] = space(c, res.norm, np.asarray(cond["X_obs"], dtype=np.float64), out["m"].astype(bool), rest_eff,
                            res.excluded)
    return out


CONDITION_KEYS = ("X_obs", "M", "rest", "rot_curve", "reference", "delta", "state", "asym", "state_on", "tend",
                "tend_mask")


def make_item(res: Resources, cond: dict, text: tuple, guidance=None) -> dict:
    """One clip to sample from its condition arrays (notes_conditions) and script features (timing (T, 11),
    viseme ids (T,)): model inputs, the diffusion variable, and Pi_delta for the constrained entries."""
    if not isinstance(cond, dict) or set(cond) != set(CONDITION_KEYS):
        raise ValueError("item_v2: cond must hold exactly the condition_v2 arrays %s" % (CONDITION_KEYS,))
    m = np.asarray(cond["M"], dtype=bool)
    n = len(m)
    tx, lx = (np.asarray(a) for a in text)
    if tx.shape != (n, _network.N_TEXT) or lx.shape != (n,):
        raise ValueError("text features must be (T, 11) and (T,)")
    cb = make_branch(res, cond, tx, lx, False, False)
    x_dec = np.asarray(cond["X_obs"], dtype=np.float64)
    out = {"n": n, "x_dec": x_dec, "m_real": m.copy(), "rest": np.asarray(cond["rest"], dtype=np.float64), "cond": cb,
           "branches": [(1.0, cb)]}
    m_in = cb["m"].astype(bool)
    out["m_in"] = np.zeros_like(m_in)
    out["obs_var"] = np.zeros(m_in.shape)
    if m_in.any():
        proj = _conditions.projection_input(cond)
        out["proj"] = proj
        out["project"] = PiDeltaProjection(cb["space"], proj)
        out["finish"] = lambda y: finish_bounded(y, proj, out["rest"], res.excluded)
    return out


def _renoise(z_prev, t_prev: int, t: int, g):
    a_p, s_p = (float(v) for v in _network.alpha_sigma(torch.tensor([int(t_prev)])))
    a_t, s_t = (float(v) for v in _network.alpha_sigma(torch.tensor([int(t)])))
    r = a_t / a_p
    return r * z_prev + float(np.sqrt(max(s_t ** 2 - (r * s_p) ** 2, 0.0))) * torch.randn(z_prev.shape, generator=g)


@torch.no_grad()
def ddim_sample(res: Resources, items: list, seeds: list, indices: list = None, steps: int = None,
                 repaint_u: int = None) -> list:
    """25-step DDIM (eta = 0, no guidance; Sec. 3.4 "sampling") for a list of clips.

    At each step i the network predicts v; the clean motion d0_hat = alpha_i z_i - sigma_i v_hat is computed per
    window, blended across windows with triangular weights (Supp. C "long sequences"), projected with Pi_delta
    (Eq. (6)) and used for the next step.  The initial noise of clip k uses torch.Generator(seed * 1000003 + index_k),
    so a clip's sample does not depend on the other clips in the batch.  Returns [x (T, 63) float64]."""
    model, c, device = res.model, res.config, res.device
    model.eval()
    steps = int(steps or c["sampler"]["steps"])
    U = int(repaint_u or c["sampler"]["repaint_u"])
    L = window_len(c)
    wts = torch.as_tensor(window_weights(L), dtype=torch.float32, device=device)
    indices = list(range(len(items))) if indices is None else indices
    gens, st = [], []
    for it, seed, idx in zip(items, seeds, indices):
        g = torch.Generator().manual_seed(int(seed) * 1000003 + int(idx))
        gens.append(g)
        n = it["n"]
        s = {"z": torch.randn((n, _network.D), generator=g).to(device), "starts": window_starts(n, L),
             "m_in": torch.as_tensor(it["m_in"], device=device),
             "obs": torch.as_tensor(it["obs_var"], dtype=torch.float32, device=device),
             "br": [], "project": it.get("project")}
        base_c, scale = it["cond"]["space"].base, it["cond"]["space"].scale
        for coef, b in it["branches"]:
            shift = (b["space"].base - base_c) / scale
            s["br"].append({"coef": float(coef), "shift": torch.as_tensor(shift, dtype=torch.float32, device=device),
                            **{k: torch.as_tensor(b[k], device=device) for k in ("x_obs", "m", "rest", "text", "letters")
                               + (("timeline",) if "timeline" in b else ()) + (("v2",) if "v2" in b else ())}})
        st.append(s)
    jobs = [(i, q, s0) for i, s in enumerate(st) for q in range(len(s["br"])) for s0 in s["starts"]]
    ts = np.linspace(_network.N_T - 1, 0, steps).round().astype(int)

    def cut(a, s0, fill=0):
        piece = a[s0:s0 + L]
        if piece.shape[0] < L:
            piece = torch.cat([piece, torch.full((L - piece.shape[0],) + tuple(piece.shape[1:]), fill, dtype=piece.dtype,
                                                 device=device)])
        return piece

    def predict_d0_hat(t: int):
        acc = [torch.zeros_like(s["z"]) for s in st]
        accw = [torch.zeros(s["z"].shape[0], 1, device=device) for s in st]
        alpha_i, sigma_i = _network.alpha_sigma(torch.tensor([int(t)], device=device))
        for b0 in range(0, len(jobs), res.batch):
            chunk = jobs[b0:b0 + res.batch]
            zs, xo, mm, rr, tx, lt, pad, sh, tl, v2 = [], [], [], [], [], [], [], [], [], []
            for i, q, s0 in chunk:
                br = st[i]["br"][q]
                zs.append(cut(st[i]["z"] - alpha_i * br["shift"], s0))
                xo.append(cut(br["x_obs"], s0))
                mm.append(cut(br["m"], s0))
                rr.append(br["rest"])
                tx.append(cut(br["text"], s0))
                lt.append(cut(br["letters"], s0))
                pad.append(torch.arange(L, device=device) < (st[i]["z"].shape[0] - s0))
                sh.append(cut(br["shift"], s0))
                if "timeline" in br:
                    tl.append(cut(br["timeline"], s0))
                if "v2" in br:
                    v2.append(cut(br["v2"], s0))
            z = torch.stack(zs)
            tt = torch.full((len(chunk),), int(t), device=device, dtype=torch.long)
            v = model(z, tt, torch.stack(xo), torch.stack(mm), torch.stack(rr), torch.stack(tx), torch.stack(lt),
                      torch.stack(pad), torch.stack(tl) if tl else None, **({"v2": torch.stack(v2)} if v2 else {}))
            d0_hat = alpha_i * z - sigma_i * v + torch.stack(sh)  # in the conditional branch's variable
            for r, (i, q, s0) in enumerate(chunk):
                n_here = min(L, st[i]["z"].shape[0] - s0)
                coef = st[i]["br"][q]["coef"]
                acc[i][s0:s0 + n_here] += coef * wts[:n_here, None] * d0_hat[r, :n_here]
                if q == len(st[i]["br"]) - 1:
                    accw[i][s0:s0 + n_here] += wts[:n_here, None]
        out = []
        for i, s in enumerate(st):
            d0_hat = acc[i] / accw[i]
            d0_hat = torch.where(s["m_in"], s["obs"], d0_hat)  # replacement on the fused x0
            if s["project"] is not None:  # Pi_delta, Eq. (6)
                d0_hat = s["project"].torch(d0_hat)
            out.append(d0_hat)
        return out

    for j, t in enumerate(ts):
        t_prev = int(ts[j + 1]) if j + 1 < len(ts) else -1
        for u in range(U if t_prev >= 0 else 1):
            d0_hats = predict_d0_hat(int(t))
            alpha_i, sigma_i = (float(v) for v in _network.alpha_sigma(torch.tensor([int(t)])))
            for i, s in enumerate(st):
                if t_prev < 0:
                    s["z"] = d0_hats[i]
                    continue
                eps = (s["z"] - alpha_i * d0_hats[i]) / sigma_i
                alpha_prev, sigma_prev = (float(v) for v in _network.alpha_sigma(torch.tensor([t_prev])))
                s["z"] = alpha_prev * d0_hats[i] + sigma_prev * eps
                if u < U - 1:
                    s["z"] = _renoise(s["z"].cpu(), t_prev, int(t), gens[i]).to(device)
    out = []
    for it, s in zip(items, st):
        y = it["cond"]["space"].to_x(s["z"].double().cpu().numpy())
        if "finish" in it:  # finish_bounded
            out.append(it["finish"](y))
            continue
        out.append(finish(y, it["x_dec"], it["m_in"], it["rest"], res.excluded))
    return out
