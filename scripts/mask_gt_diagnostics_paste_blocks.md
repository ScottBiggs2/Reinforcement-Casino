# Mask GT analysis — information collection (Explorer)

Paste blocks **in order** on the cluster. Replace `JOBID` and paths if yours differ.

Default layout (adjust `MASK_DIR` if needed):

- `REPO=/home/biggs.s/rl_casino`
- `MASK_DIR=/scratch/biggs.s/rl_casino_masks/tulu3_500_h200_fresh_0409_again`
- `COMP_DIR=$MASK_DIR/comparisons_vs_ground_truth`

---

## Block 1 — Environment (fresh shell)

```bash
export REPO="/home/biggs.s/rl_casino"
export MASK_DIR="/scratch/biggs.s/rl_casino_masks/tulu3_500_h200_fresh_0409_again"
export COMP_DIR="${MASK_DIR}/comparisons_vs_ground_truth"
export JOBID="5951603"   # <-- your Slurm job id

cd "$REPO" || exit 1
echo "COMP_DIR=$COMP_DIR"
```

---

## Block 2 — Job accounting (confirm state, workdir)

```bash
sacct -j "$JOBID" --format=JobID,JobName,State,ExitCode,Elapsed,Start,End,WorkDir%80
```

---

## Block 3 — Scan Slurm stdout/stderr + tee log for CKA / errors

```bash
# stdout / stderr from sbatch (repo-relative logs/)
for f in "$REPO"/logs/mask_gt_analysis_${JOBID}.out "$REPO"/logs/mask_gt_analysis_${JOBID}.err; do
  echo "===== $f ====="
  ls -la "$f" 2>&1
done

echo ""
echo "---- CKA / failures / tracebacks (stdout) ----"
grep -nE 'WARNING: CKA|OK: CKA|Traceback|OutOfMemory|CUDA|timeout|ERROR' \
  "$REPO"/logs/mask_gt_analysis_${JOBID}.out 2>/dev/null | head -80

echo ""
echo "---- stderr tail ----"
tail -120 "$REPO"/logs/mask_gt_analysis_${JOBID}.err 2>/dev/null
```

---

## Block 4 — Tee log (if present): OK vs WARNING CKA counts

```bash
# Find the run log (name includes RUN_ID; adjust glob if needed)
LOG=$(ls -t "$REPO"/logs/mask_gt_analysis_gt_analysis_*.log 2>/dev/null | head -1)
echo "LOG=$LOG"
if [ -n "$LOG" ]; then
  echo "OK CKA count:    $(grep -c 'OK: CKA' "$LOG" 2>/dev/null || echo 0)"
  echo "WARN CKA count:  $(grep -c 'WARNING: CKA' "$LOG" 2>/dev/null || echo 0)"
  echo ""
  echo "---- First 20 CKA warnings ----"
  grep -n 'WARNING: CKA' "$LOG" 2>/dev/null | head -20
fi
```

---

## Block 5 — Artifact counts (jaccard vs cka vs csv)

```bash
echo "jaccard JSON: $(ls -1 "$COMP_DIR"/jaccard_gt_vs_*.json 2>/dev/null | wc -l | tr -d ' ')"
echo "cka JSON:     $(ls -1 "$COMP_DIR"/cka_gt_vs_*.json 2>/dev/null | wc -l | tr -d ' ')"
echo "layer CSV:    $(ls -1 "$COMP_DIR"/layer_metrics_gt_vs_*.csv 2>/dev/null | wc -l | tr -d ' ')"
echo ""
echo "---- Tags with jaccard but NO cka (first 30) ----"
comm -23 \
  <(ls -1 "$COMP_DIR"/jaccard_gt_vs_*.json 2>/dev/null | xargs -n1 basename | sed 's/^jaccard_//;s/\.json$//' | sort) \
  <(ls -1 "$COMP_DIR"/cka_gt_vs_*.json 2>/dev/null | xargs -n1 basename | sed 's/^cka_//;s/\.json$//' | sort) \
  | head -30
```

---

## Block 6 — Inspect one `cka_*.json` (pick an existing file)

Replace `TAG` with a real tag, e.g. `gt_vs_warm_magnitude_meta_llama_llama_3_1_8b_instruct_tulu3_sparsity97.5pct_step200`.

```bash
TAG="gt_vs_warm_magnitude_meta_llama_llama_3_1_8b_instruct_tulu3_sparsity97.5pct_step200"
CKA_JSON="$COMP_DIR/cka_${TAG}.json"
ls -la "$CKA_JSON"
python3 - <<PY
import json, os
p = r"""$CKA_JSON"""
if not os.path.isfile(p):
    print("MISSING:", p)
else:
    d = json.load(open(p))
    pl = d.get("per_layer_cka") or {}
    ck = d.get("cka") or {}
    print("top keys:", list(d.keys())[:12])
    print("cka block:", ck)
    print("per_layer_cka count:", len(pl))
    print("sample entries:", list(pl.items())[:5])
PY
```

(If your shell does not expand `$CKA_JSON` inside the heredoc, use `ls` on the path and then:)

```bash
python3 -c "import json; d=json.load(open('$COMP_DIR/cka_${TAG}.json')); pl=d.get('per_layer_cka',{}); print('per_layer_cka', len(pl), 'cka', d.get('cka')); print(list(pl.items())[:3])"
```

---

## Block 7 — CSV: fraction of non-NaN `cka` (same TAG)

```bash
TAG="gt_vs_warm_magnitude_meta_llama_llama_3_1_8b_instruct_tulu3_sparsity97.5pct_step200"
CSV="$COMP_DIR/layer_metrics_${TAG}.csv"
python3 -c "
import csv, math
p = r'''$CSV'''
with open(p, newline='', encoding='utf-8') as f:
    rows = list(csv.DictReader(f))
cka = [r.get('cka', '') for r in rows]

def ok(x):
    s = str(x).strip().lower()
    if s in ('', 'nan', 'none'):
        return False
    try:
        v = float(s)
        return math.isfinite(v)
    except Exception:
        return False
n_ok = sum(1 for x in cka if ok(x))
print(p.split('/')[-1], 'rows', len(rows), 'cka_numeric', n_ok)
"
```

---

## Block 8 — Jaccard null band width (same formulas as plotter)

Uses one `layer_metrics_*.csv` path; set `CSV` as in block 7.

```bash
TAG="gt_vs_warm_magnitude_meta_llama_llama_3_1_8b_instruct_tulu3_sparsity97.5pct_step200"
export CSV="$COMP_DIR/layer_metrics_${TAG}.csv"
python3 - <<'PY'
import csv, math, os

def to_float(x):
    if x is None:
        return float("nan")
    s = str(x).strip()
    if s == "" or s.lower() == "none":
        return float("nan")
    try:
        return float(s)
    except Exception:
        return float("nan")

def to_int_params(x):
    if x is None:
        return None
    s = str(x).strip()
    if s == "" or s.lower() == "none":
        return None
    try:
        return int(float(s))
    except Exception:
        return None

def global_density(rows, which):
    key_d, key_s = f"density_{which}", f"sparsity_{which}"
    num = den = 0.0
    for r in rows:
        n_p = to_int_params(r.get("n_params")) or 0
        if n_p <= 0:
            continue
        d = to_float(r.get(key_d))
        if math.isnan(d):
            s = to_float(r.get(key_s))
            d = 1.0 - s if not math.isnan(s) else float("nan")
        if math.isnan(d):
            continue
        num += n_p * d
        den += n_p
    return num / den if den > 0 else float("nan")

def bern_mean(da, db):
    denom = da + db - da * db
    if denom <= 0:
        return 0.0
    return (da * db) / denom

def var_j(rho, n):
    if n < 1 or not (0.0 <= rho <= 1.0):
        return float("nan")
    den = (2.0 - rho * rho) ** 3
    if den <= 0:
        return float("nan")
    return 2.0 * (1.0 - rho) / (float(n) * den)

p = os.environ["CSV"]
with open(p, newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))
n_params_col = [to_int_params(r.get("n_params")) for r in rows]
n_tot = sum(n for n in n_params_col if n is not None and n > 0)
da_g = global_density(rows, "a")
db_g = global_density(rows, "b")
em = bern_mean(da_g, db_g)
rho_m = 0.5 * (da_g + db_g)
vj = var_j(rho_m, n_tot)
sig = math.sqrt(max(0.0, vj)) if not math.isnan(vj) else float("nan")
print("file:", os.path.basename(p))
print("n_tot (sum n_params):", n_tot)
print("rho_a_hat, rho_b_hat:", da_g, db_g)
print("E[J] null:", em)
print("rho_bar:", rho_m)
print("Var(J) theory:", vj)
print("sigma:", sig)
print("E-sigma, E+sigma:", em - sig, em + sig)
print("relative band half-width sigma/E:", (sig / em) if em and em > 0 else float("nan"))
PY
```

If `sigma/E` is ~1e-10, the horizontal band is **invisible** on a log plot next to E[J] ~ 1e-2 — expected with Var ∝ 1/N at global N.

---

## Block 9 — Jaccard JSON aggregates (`mask_to_jaccard` report)

```bash
TAG="gt_vs_warm_magnitude_meta_llama_llama_3_1_8b_instruct_tulu3_sparsity97.5pct_step200"
JPATH="$COMP_DIR/jaccard_${TAG}.json"
ls -la "$JPATH"
python3 -c "
import json
d=json.load(open('$JPATH'))
# mask_to_jaccard: top keys mask_a, mask_b, jaccard{aggregate,mean,...}, per_layer_jaccard
print('keys:', list(d.keys()))
j = d.get('jaccard') or {}
print('jaccard block:', j)
pl = d.get('per_layer_jaccard')
if isinstance(pl, dict):
    print('per_layer_jaccard layers:', len(pl))
elif isinstance(pl, list):
    print('per_layer_jaccard len:', len(pl))
"
```

---

## What to send back

1. Output of **Block 2** (sacct).
2. **Block 3** stderr tail (or “file not found” if paths differ).
3. **Block 4** OK/WARN counts and a few `WARNING: CKA` lines.
4. **Block 5** three counts + a few “jaccard but no cka” tag names.
5. **Block 6** + **Block 7** for **one** tag (exists vs missing `cka_*.json`).
6. Full **Block 8** printout for the same CSV.

That is enough to decide: rerun CKA only, fix export, or adjust Jaccard null visualization/theory labeling.
