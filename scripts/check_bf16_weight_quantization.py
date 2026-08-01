"""Are the ~99% exact-zero warm-start scores caused by bf16 weight quantization?

Hypothesis. Training runs with bf16=True, so the weights themselves are bf16. bf16 carries 8
ation bits, so the representable spacing at a weight w is ulp(w) = 2^(floor(log2|w|)) * 2^-7.
At lr=5e-7 a single AdamW step moves a weight by roughly lr in magnitude, which for |w| ~ 1e-2
(ulp ~ 4e-5) is ~80x below the spacing. Such an update rounds away entirely and the weight does
not change, which would make the delta exactly zero rather than merely small.

Test. For each sampled tensor, compare each coordinate's step-50 delta against ulp(base weight):

  * if the mechanism is weight-level quantization, every nonzero delta is an exact integer
    multiple of ulp(base), and coordinates stay at zero when the accumulated pressure is below
    one ulp;
  * if instead the deltas were merely small numbers rounded during *storage*, the nonzero values
    would not line up with ulp(base) at all.

Also reports what fraction of coordinates would need a displacement below one ulp, which is the
prediction for the zero fraction.

Usage:
  DELTA_DIR=/path/to/<run>/deltas/<dataset> python scripts/check_bf16_weight_quantization.py
"""
import os
import sys

import torch

DELTA_DIR = os.environ.get("DELTA_DIR")
if not DELTA_DIR:
    raise SystemExit("set DELTA_DIR to a delta-log directory containing base_state.pt")
SAMPLE_KEYS = [
    "model.layers.10.self_attn.q_proj.weight",
    "model.layers.10.mlp.gate_proj.weight",
    "model.embed_tokens.weight",
]
LR = 5e-7
STEPS = 50

print("loading base_state.pt ...", flush=True)
base = torch.load(f"{DELTA_DIR}/base_state.pt", map_location="cpu")
print("loading deltas_step_50.pt ...", flush=True)
dl = torch.load(f"{DELTA_DIR}/deltas_step_50.pt", map_location="cpu")

print(f"base dtype sample : {next(iter(base.values())).dtype}")
print(f"delta dtype sample: {next(iter(dl.values())).dtype}")
print()

for k in SAMPLE_KEYS:
    if k not in base or k not in dl:
        print(f"{k}: absent (base={k in base}, delta={k in dl})")
        continue
    w = base[k]
    d = dl[k]
    wf = w.float().abs()
    df = d.float()

    # bf16 ulp at each base weight: 2^(exponent) * 2^-7 for an 8-bit significand.
    exp = torch.floor(torch.log2(wf.clamp_min(1e-30)))
    ulp = torch.pow(2.0, exp - 7.0)

    nz = df != 0
    n = df.numel()
    n_nz = int(nz.sum())

    out = [f"{k}", f"  numel={n:,}  nonzero={n_nz:,} ({n_nz/n:.6f})"]

    if n_nz > 0:
        ratio = df[nz].abs() / ulp[nz]
        # Integer-multiple test: distance from the nearest integer, in units of ulp.
        frac_part = (ratio - torch.round(ratio)).abs()
        out.append(f"  |delta|/ulp(base): median={ratio.median():.4f} "
                   f"min={ratio.min():.4f} max={ratio.max():.4f}")
        out.append(f"  distance to nearest integer multiple of ulp: "
                   f"median={frac_part.median():.6f} "
                   f"frac within 1e-3 of an integer={float((frac_part < 1e-3).float().mean()):.6f}")
        out.append(f"  frac of nonzero deltas that are exactly 1 ulp: "
                   f"{float((torch.round(ratio) == 1).float().mean()):.6f}")

    # Prediction: coordinates whose plausible accumulated movement is under one ulp cannot move.
    budget = LR * STEPS
    below = (ulp > budget)
    out.append(f"  ulp(base) > lr*steps ({budget:.2e}) for {float(below.float().mean()):.6f} "
               f"of coordinates  <- predicted zero fraction")
    out.append(f"  median ulp(base)={ulp.median():.3e}  median |w|={wf.median():.3e}")
    print("\n".join(out), flush=True)
    print()

print("Reading: if nonzero deltas sit at integer multiples of ulp(base) and the predicted zero")
print("fraction tracks the observed one, the zeros are weight-level bf16 quantization, not a")
print("logging artifact -- i.e. at this lr most weights never move at all.")
