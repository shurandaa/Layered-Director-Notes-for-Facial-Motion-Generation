# Layered Director Notes for Facial Motion Generation

Code, data and models for the paper *Layered Director Notes for Facial Motion Generation*. With this repository you
can encode tracked facial motion into director notes, generate motion from notes, rebuild the training data, train
the generator, and score generated motion with the metrics of the paper.

**Demo videos:** https://shurandaa.github.io/Layered-Director-Notes-for-Facial-Motion-Generation/

## Install

Python 3.12:

```bash
python3.12 -m venv env && source env/bin/activate
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128
export STRATA_NLTK_DATA=$DATA/nltk_data   # CMUdict and the POS tagger used by g2p_en (shipped with the data)
```

## Data and models (`$DATA`)

```bash
pip install -U huggingface_hub
hf download HF_DATASET_ID --repo-type dataset --local-dir $DATA
cd $DATA && sha256sum -c SHA256SUMS
```

| Path | Contents | Size |
|---|---|---|
| `clips.json`, `scripts.json`, `names.json`, `curves_{train,val}.npz` | tracked 63-D curves and the script of every clip: 6,684 training and 1,485 validation clips of TalkVid, RAVDESS and CREMA-D (`strata/data.py`) | 0.2 GB |
| `checkpoints/strata_s{0,1,2}.pt` | the generator of the paper, three seeds | 62 MB |
| `rows/ours_s{0,1,2}/` | the paper's generated validation motion (Tab. 1 "ours"), full / standard / lite notes, 4 samples per clip | 2.1 GB |
| `reference/`, `golden/` | the paper's numbers and hashes used by `tests/test_release.py` | 80 MB |
| `nltk_data/` | CMUdict and the averaged perceptron tagger (for visemes of new scripts) | 3 MB |

The curves come from MediaPipe tracks of the three datasets; their licences apply. Raw videos are not included.

## Use

```bash
# training and validation data (encoder, random masking, densities): about 40 min on one CPU core
python -m strata.build_prep --data $DATA --out prep

# generate from your own notes and script (see strata/notes.py and strata/script.py for the formats)
python -m strata.generate notes --checkpoint $DATA/checkpoints/strata_s0.pt --notes notes.json --script script.json \
    --out motion.npz --device cuda

# the paper's validation rows (full / standard / lite, 4 samples per clip)
python -m strata.generate val --checkpoint $DATA/checkpoints/strata_s0.pt --prep prep --out run_s0 --device cuda

# Tab. 1 metrics (seeds pooled)
python -m strata.score bundles --prep prep --out bundles
python -m strata.score fmd --bundles bundles --rows run_s0,run_s1,run_s2 --out score --densities standard
python -m strata.score roundtrip --prep prep --data $DATA --rows run_s0,run_s1,run_s2 --out score/rt --densities standard

# training (50k steps; the released checkpoints are seeds 0, 1, 2)
python -m strata.train_cli --prep prep --out train_s0 --seed 0 --device cuda
```

## Reproduction tests

```bash
export OMP_NUM_THREADS=8   # CPU stages; many more threads than the CPU quota of a container stall training
python tests/test_release.py --data $DATA --work work --device cuda
```

| Stage | What it checks | Required | Fresh machine (NVIDIA L4, AMD EPYC 7542, AVX2) |
|---|---|---|---|
| prep | the rebuilt prep against the paper's prep, covering every array of every clip, every notes record and the training statistics | notes identical, statistics to 1e-9 | notes identical; 98.4 % of 647,592 arrays bit-identical, the rest differ in the last bit (<= 3e-17); 18 min |
| golden | 18 samples generated on the CPU against the research code's CPU output | 1e-4 | max difference 1.2e-6 |
| score | Tab. 1 metrics of the paper's generated rows, and the round-trip counts of every clip | metrics to 1e-6, counts exact | FMD / seam / W1 equal to the paper's printed digits; round-trip counts identical (180 clip-density checks, 20 clips per dataset and density); about 2.3 h per density for FMD on 14 cores |
| generate | the validation rows generated again on this GPU and then scored | reported | 72 min per checkpoint on an L4 |
| train | 200 training steps on the CPU against the paper code's loss history | 1e-5 | relative difference 2.6e-9; 14 min |

**Exactness and hardware.** On an x86 CPU with AVX-512 (the paper's machine, AMD EPYC 9745) data preparation,
CPU sampling, scoring and training reproduce the research code bit for bit. Other CPUs run other numpy / PyTorch
float kernels, so arrays differ in the last bits; the notes, the round-trip counts and the reported metrics do not
change. The rows of the paper were sampled on an NVIDIA B200. On other GPUs or on the CPU the float32 network outputs
differ by about 1e-3, so regenerated motion matches the paper only to that level; the test reports how much each
Tab. 1 metric moves.

Comments in the code cite internal experiment and decision identifiers (`T0486`, `D242`, ...) of the research log;
they are kept so that the code matches the research code it was extracted from.

## Modules and paper sections

| Module | Paper | Contents |
|---|---|---|
| `strata/representation.py` | Sec. 3.1 | 63-D frame layout, 6D rotation ↔ rotation vector ω about R_ref, tracking validity |
| `strata/vocabulary.py` | Sec. 3.2, Supp. A | families f, channels 𝒞(f), 14 timelines, tolerances δ_c, constant files |
| `strata/notes.py` | Eq. (1), Supp. A | what a set of director notes may contain (validation) |
| `strata/decoder.py` | Sec. 3.3, Eq. (2), (3), (6); Supp. B | φ_τ, y_e, decode N → x̂, M, X_obs; projection Π_δ |
| `strata/encoder.py` | Sec. 3.3, Eq. (4); Supp. B | clip header, states, events by hysteresis, encode curves → N |
| `strata/tendency.py` | Sec. 3.2; Supp. A, B | the seven tendency statistics and their levels |
| `strata/sparsify.py` | Sec. 3.2; Supp. A | standard and lite notes from full notes |
| `strata/script.py` | Sec. 3.4; Supp. C | word-timing features and viseme ids of the script w |
| `strata/conditions.py` | Sec. 3.4; Supp. C | the condition 𝐜(N, w) and the model input |
| `strata/network.py` | Sec. 3.4, Eq. (5); Supp. C | v_θ (Transformer), cosine schedule α_i, σ_i |
| `strata/sampler.py` | Sec. 3.4, Eq. (6); Supp. C | 25-step DDIM, window blending, Π_δ on d̂_0 at every step |
| `strata/postprocess.py` | Supp. C | head smoothing σ_h = 0.75 frames, re-projection |
| `strata/data.py` | Sec. 4.1 | loading the released curves and scripts |
| `strata/prep.py`, `strata/build_prep.py` | Sec. 3.4 training; Supp. C | full notes, randomly masked variants R(N), validation densities |
| `strata/train.py`, `strata/train_cli.py` | Sec. 3.4, Eq. (5); Supp. C | training of v_theta |
| `strata/generate.py` | Sec. 3.4 | generation from notes, validation rows |
| `strata/evaluate.py`, `strata/roundtrip.py`, `strata/score.py` | Sec. 4.1; Supp. D | FMD, seam, W1; re-encoding round trip |

## Symbols and code

| Paper | Code | Where |
|---|---|---|
| x_t = [b_t, p_t, r_t, g_t] (63-D) | columns 0–51 / 52–54 / 55–60 / 61–62 | `representation`, `network.BS` |
| R_ref | `reference` (rotation vector, degrees) | `representation.reference_orientation` |
| ω_t = (180/π) log(R_refᵀ R(r_t))^∨ | `omega_from_r6`, inverse `r6_from_omega` | `representation` |
| N = (h, 𝒯, 𝒮, ℰ) (Eq. 1) | notes dict `{hz, n_frames, rest, reference, tendency, state, event}` | `notes.validate_track` |
| family f, 𝒞(f) | `fam`, `channels_of(f)` | `vocabulary` |
| τ = (τ0, τ1, τ2, τ3) | `"t"` in seconds; `tau0..tau3` / `taus` in frames | `vocabulary.tau_frames` |
| ℓ, v, A_e | `"level"`, `"value"` (state / event) | notes fields |
| ψ(u) = 3u² − 2u³ | `psi` | `decoder` |
| φ_τ(t) | `phi_trapezoid` | `decoder` |
| y_e(t), P(t), b_e, h_f, κ^f_↑, κ^f_↓ | `y_event` (`P`, `b`, `h_f`, `kappa_up`, `kappa_down`), `b_e` | `decoder` |
| u_f | `state_loadings`, `event_loadings` (calibrated) | `decoder._member_values` |
| x̂ (Eq. 2) | `decode(...)["x"]`, locally `x_hat` | `decoder.decode` |
| M, X_obs (Eq. 3) | `"M"`, `"X_obs"` | `decoder.decode` |
| d̃ = 𝒢_σ=5 ∗ med_31(x − rest) (Eq. 4) | `smooth_d_tilde`, locally `d_tilde` | `encoder` |
| s^(f)_t, g_f | `g_f(...)`, locally `s_f` | `encoder.g_f` |
| e^f_k, ℓ_t | `level_boundaries`, `frame_levels` (`e_f`, `level`) | `vocabulary`, `encoder` |
| ŝ, ρ = x − rest − ŝ | `s_hat`, `rho` | `encoder.encode` |
| θ^f_low, θ^f_high, d^f_min, d^f_max | `theta_low`, `theta_high`, `d_min`, `d_max` | `encoder.detect_events` |
| δ_c | `delta` (`delta_c_row`) | `vocabulary.delta_c_row` |
| 𝐜(N, w) | `notes_conditions` (notes part), `model_inputs` (with the script) | `conditions` |
| w (script) | words with start / end frames → `script.features` | `script` |
| d_0 | `(x − mean) / std`, the diffusion variable `Space` | `sampler.Space` |
| α_i, σ_i (Eq. 5) | `alpha_sigma`, `cosine_alpha_bar` | `network` |
| v_θ | `Denoiser` | `network` |
| d̂_0 = α_i z_i − σ_i v̂ | `d0_hat` | `sampler.ddim_sample` |
| Π_δ (Eq. 6) | `Pi_delta` (output), `PiDeltaProjection` (every step) | `decoder`, `sampler` |
| σ_h | `SIGMA_H` | `postprocess` |

## Constants (field names in the files)

`strata/assets/l2_v2_vocabulary.json` holds the hand-set constants. `strata/assets/l2_v2_fitted.json` holds the
constants calibrated once on the 480 TalkVid training clips (Supp. B, "properties and constants"). Both files are
byte-identical to the files used in the paper, and the code checks their sha256. Their field names are part of the
file format, so they are kept as they are:

| Paper | Field |
|---|---|
| e^f_0 (activation threshold) | `families.<f>.e0` |
| e^f_1, e^f_2 (state level boundaries) | `state_edges.<timeline>` = [e0, e1, e2] (fitted) |
| θ^f_low, θ^f_high | `families.<f>.event.dev_low`, `dev_high` (mouth tension: per kind) |
| d^f_min, d^f_max | `families.<f>.event.min_s`, `max_s` (seconds) |
| δ_c | `families.<f>.event.delta`; non-driver blendshapes `delta.non_driver_blendshape` |
| median window 31, σ = 5 (Eq. 4) | `state.median_frames`, `state.gauss_sigma_frames` |
| minimum state length 1 s | `state.min_hold_frames` (30) |
| 10 % / 90 % time points | `state.release_frac`, `state.settled_frac` |
| apex: above 90 % of the peak | `event.apex_frac` |
| u_f | `state_loadings`, `event_loadings`; gaze `eyelook_map` (8 × 2) |
| h_f, κ^f_↑, κ^f_↓ | `event_shape.<f>.h`, `tau_up`, `tau_dn` (frames) |
| amplitude tertiles, representative values | `event_edges`, `event_repr`, `state_repr` |

Differences between the paper's words and the notes format:
- **Levels:** the paper's "moderate" is written `clear`. The three levels are `slight`, `clear` and `strong`.
- **Family names:** eyelids = `eyelid`, frown = `brow_lower`, head rotation = `head_rot`, head translation =
  `head_pos`.
- **The letter κ:** the supplement uses it twice. κ_e is an event's kind and direction (`kind`, `dir`), and κ^f_↑ and
  κ^f_↓ are the template time constants (`kappa_up`, `kappa_down`).

