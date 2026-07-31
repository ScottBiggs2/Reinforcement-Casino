"""End-to-end CPU test for src/analysis/certifiability_margins.py on a synthetic run.

The fixture reproduces the features that actually broke things on the real artifacts: 1-D norm
weights that must be out of scope, a tied ``lm_head`` that must be de-duplicated, and a heavy exact-
zero mass in the deltas (bf16 at lr=5e-7 — RESULTS.md §3.0) so the degeneracy path is exercised.
"""

import json
import os
import sys

import numpy as np
import pytest
import torch

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.analysis import certifiability_margins as cm  # noqa: E402

MILESTONES = [50, 100, 150, 200]
HID = 64
VOCAB = 96


def _make_run(tmp_path, *, zero_frac: float = 0.9, seed: int = 0):
    """Writes base_state.pt, deltas_step_*.pt, and a final state dict; returns the paths."""
    g = torch.Generator().manual_seed(seed)
    names_2d = ["model.embed_tokens.weight"]
    for layer in range(2):
        for proj in ("self_attn.q_proj", "self_attn.k_proj", "mlp.gate_proj", "mlp.down_proj"):
            names_2d.append(f"model.layers.{layer}.{proj}.weight")

    base = {}
    for n in names_2d:
        rows = VOCAB if "embed_tokens" in n else HID
        base[n] = (torch.randn(rows, HID, generator=g) * 0.02).to(torch.bfloat16)
    # Tied head: bit-identical to the embedding, so the builder must drop exactly one of them.
    base["lm_head.weight"] = base["model.embed_tokens.weight"].clone()
    # 1-D norms: in the state dict, out of the builder's scope.
    for layer in range(2):
        base[f"model.layers.{layer}.input_layernorm.weight"] = torch.ones(HID, dtype=torch.bfloat16)
    base["model.norm.weight"] = torch.ones(HID, dtype=torch.bfloat16)

    run = tmp_path / "run"
    deltas = run / "deltas"
    deltas.mkdir(parents=True)
    torch.save(base, deltas / "base_state.pt")

    # Cumulative deltas theta^j - theta^0, mostly exactly zero, growing with j.
    cumulative = {n: torch.zeros_like(base[n], dtype=torch.float32) for n in base}
    for step in MILESTONES:
        snap = {}
        for n, b in base.items():
            step_move = torch.randn(b.shape, generator=g) * 1e-4
            keep = torch.rand(b.shape, generator=g) > zero_frac
            cumulative[n] = cumulative[n] + step_move * keep
            snap[n] = cumulative[n].to(torch.bfloat16)
        torch.save(snap, deltas / f"deltas_step_{step}.pt")

    final = {n: (base[n].float() + cumulative[n]).to(torch.bfloat16) for n in base}
    final_path = run / "final_state.pt"
    torch.save(final, final_path)
    return deltas / "base_state.pt", final_path, deltas


def _run_pipeline(tmp_path, extra=(), zero_frac: float = 0.9, tag: str = "out"):
    base_path, final_path, deltas = _make_run(tmp_path, zero_frac=zero_frac)
    out_dir = tmp_path / tag
    argv = [
        "--initial_model", str(base_path),
        "--final_model", str(final_path),
        "--delta_log_dir", str(deltas),
        "--out_dir", str(out_dir),
        "--magnitude_milestones", ",".join(str(s) for s in MILESTONES),
        "--sparsity_percents", "97.5,50",
        "--histogram_bins", "2000",
        "--kth_num_bins", "64",
        "--execution_mode", "full",
        *extra,
    ]
    cm.main(argv)
    return out_dir


def test_pipeline_end_to_end(tmp_path):
    out_dir = _run_pipeline(tmp_path)

    assert (out_dir / "certifiability_summary.csv").is_file()
    assert (out_dir / "certifiability_margins.npz").is_file()
    diag = json.loads((out_dir / "certifiability_diagnostics.json").read_text())

    # Scope: 9 distinct 2-D weights (embed + 2 layers x 4 proj) after dropping the tied lm_head,
    # and no 1-D norms.
    scope = diag["meta"]["scope"]
    assert scope["n_tensors"] == 9, scope
    assert scope["score_scope"] == "builder_2d"
    assert scope["scope_source"] == "initial"

    # Every arm x rho combination produced a tau entry and a cert fraction.
    expected = {f"{a}_rho{r}" for a in ("oracle", "warm_k50", "warm_k100", "warm_k150", "warm_k200", "random_seed42") for r in ("97.5", "50")}
    assert set(diag["arms"]) == expected, sorted(set(diag["arms"]) ^ expected)

    for tag, rec in diag["arms"].items():
        assert 0.0 <= rec["cert_strict_fraction"] <= 1.0, (tag, rec["cert_strict_fraction"])
        assert rec["tau"]["n_total"] == scope["covered_elements"], tag


def test_all_arms_share_one_coverage(tmp_path):
    """The bug this replaces: warm arms were scored over a different N than oracle/random, so tau
    described a mask that was never built."""
    out_dir = _run_pipeline(tmp_path)
    diag = json.loads((out_dir / "certifiability_diagnostics.json").read_text())
    totals = {rec["tau"]["n_total"] for rec in diag["arms"].values()}
    keeps = {(rec["tau"]["sparsity_percent"], rec["tau"]["keep_count"]) for rec in diag["arms"].values()}
    assert len(totals) == 1, totals
    assert len(keeps) == 2, keeps  # one keep budget per rho, identical across arms


def test_selection_accounting_is_self_consistent(tmp_path):
    """Every coordinate at or above tau either had signal or was picked by the tie-break; the two
    counts must partition the selection exactly, and it must be the right size."""
    out_dir = _run_pipeline(tmp_path)
    diag = json.loads((out_dir / "certifiability_diagnostics.json").read_text())
    for tag, rec in diag["arms"].items():
        n_sel = rec["n_at_or_above_tau"]
        if n_sel == 0:
            continue
        assert rec["selected_with_zero_score"] + rec["nonzero_captured"] == n_sel, tag
        # The global-phase selection sits between the global-phase budget and the full keep budget.
        t = rec["tau"]
        assert n_sel <= t["keep_count"] * 1.02, (tag, n_sel, t["keep_count"])
        assert n_sel >= t["r_remaining"] * 0.98, (tag, n_sel, t["r_remaining"])
        assert rec["nonzero_captured"] <= rec["nonzero_total"], tag


def test_random_arm_tau_is_near_the_quantile(tmp_path):
    """Sanity anchor with a known answer: U(0,1) scores put tau at ~1 - keep_fraction."""
    out_dir = _run_pipeline(tmp_path)
    taus = json.loads((out_dir / "certifiability_tau.json").read_text())["taus"]
    t = taus["random_seed42|50"]
    # rho=50 keeps half the coordinates; the hybrid per-layer floors take 0.25% first, so the
    # global-phase boundary sits slightly above 0.5.
    assert 0.49 < t["tau"] < 0.56, t
    assert t["tau_degenerate"] is False


def test_degeneracy_is_reported_when_budget_exceeds_support(tmp_path):
    """A keep budget larger than the signal support makes tau_rho exactly 0, so the reported tau is
    whatever the tie-break put at rank k. That must be flagged, not smoothed over.

    zero_frac=0.995 mirrors the real regime: RESULTS.md §3.1 measured only 0.83-1.13% of coordinates
    with resolvable |dtheta| against a 2.5% keep budget at rho=97.5%.
    """
    out_dir = _run_pipeline(tmp_path, zero_frac=0.995, tag="out_sparse")
    taus = json.loads((out_dir / "certifiability_tau.json").read_text())["taus"]
    warm = taus["warm_k50|97.5"]
    assert warm["n_raw_positive"] < warm["keep_count"], warm
    assert warm["tau_degenerate"] is True
    # The reported tau collapses to the tie-break amplitude, ~1e-12 of max|s|.
    assert warm["tau"] < warm["max_abs"] * 1e-9, warm
    # random is continuous, so it is never degenerate at any rho
    assert taus["random_seed42|97.5"]["tau_degenerate"] is False


def test_degenerate_arms_emit_no_tau_relative_margin(tmp_path):
    """Refusing to normalize by a noise-level tau is the point of the degeneracy flag."""
    out_dir = _run_pipeline(tmp_path, zero_frac=0.995, tag="out_sparse2")
    diag = json.loads((out_dir / "certifiability_diagnostics.json").read_text())
    rec = diag["arms"]["warm_k50_rho97.5"]
    assert rec["tau"]["tau_degenerate"] is True
    summary = (out_dir / "certifiability_summary.csv").read_text().splitlines()
    header = summary[0].split(",")
    i_arm, i_q, i_n = header.index("arm"), header.index("quantity"), header.index("n")
    rel_n = [
        int(float(r.split(",")[i_n]))
        for r in summary[1:]
        if r.split(",")[i_arm] == "warm_k50" and r.split(",")[i_q] == "margin_rel"
    ]
    assert rel_n and all(n == 0 for n in rel_n), rel_n


def test_oracle_is_degenerate_at_a_budget_beyond_its_support(tmp_path):
    """Same failure applies to the oracle: rho=50% asks for half the coordinates but only ~10% moved."""
    out_dir = _run_pipeline(tmp_path)
    taus = json.loads((out_dir / "certifiability_tau.json").read_text())["taus"]
    assert taus["oracle|50"]["tau_degenerate"] is True
    assert taus["oracle|97.5"]["tau_degenerate"] is False


def test_warm_support_grows_with_k(tmp_path):
    """sum_{j<=k}|dtheta| accumulates, so more coordinates carry signal at larger k."""
    out_dir = _run_pipeline(tmp_path)
    taus = json.loads((out_dir / "certifiability_tau.json").read_text())["taus"]
    support = [taus[f"warm_k{k}|97.5"]["n_raw_positive"] for k in MILESTONES]
    assert support == sorted(support), support
    assert support[-1] > support[0]


def test_tau_relative_margin_mass_at_boundary_equals_zero_score_share(tmp_path):
    """The headline readability claim: coordinates with no resolvable displacement land at m~ = 1."""
    out_dir = _run_pipeline(tmp_path)
    diag = json.loads((out_dir / "certifiability_diagnostics.json").read_text())
    data = np.load(out_dir / "certifiability_margins.npz")

    tag = "warm_k200_rho97.5"  # non-degenerate here, so tau > 0 and m~ exists
    rec = diag["arms"][tag]
    assert rec["tau"]["tau_degenerate"] is False
    zero_share = rec["frac_score_exactly_zero"]
    assert zero_share > 0.1

    counts = data[f"{tag}_margin_rel_counts"]
    edges = data[f"{tag}_margin_rel_log_edges"]
    total = counts.sum() + data[f"{tag}_margin_rel_underflow"][0] + data[f"{tag}_margin_rel_overflow"][0]
    # s_i == 0 gives m~_i == 1 exactly, so the zero mass is a single spike at log10(m~) = 0 and is
    # by far the largest bin. (Which side of the exact bin edge it lands on is float-rounding
    # dependent, so locate it by the spike rather than by the edge.)
    j = int(np.argmax(counts))
    width = float(edges[1] - edges[0])
    assert abs(float(edges[j])) <= width, (edges[j], width)
    assert counts[j] / total == pytest.approx(zero_share, rel=0.02), (counts[j] / total, zero_share)


def test_home_guard_refuses_out_dir_under_home(tmp_path, monkeypatch):
    base_path, final_path, deltas = _make_run(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    bad = tmp_path / "rl_casino_results" / "cert"
    with pytest.raises(SystemExit) as ei:
        cm.main(
            [
                "--initial_model", str(base_path),
                "--final_model", str(final_path),
                "--delta_log_dir", str(deltas),
                "--out_dir", str(bad),
                "--execution_mode", "cache_only",
            ]
        )
    assert "REFUSING" in str(ei.value)
    assert not (bad / "magnitude_caches").exists()


def test_home_guard_can_be_overridden(tmp_path, monkeypatch):
    base_path, final_path, deltas = _make_run(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    ok = tmp_path / "explicit" / "cert"
    cm.main(
        [
            "--initial_model", str(base_path),
            "--final_model", str(final_path),
            "--delta_log_dir", str(deltas),
            "--out_dir", str(ok),
            "--magnitude_milestones", "50",
            "--execution_mode", "cache_only",
            "--allow_home_out_dir",
        ]
    )
    assert (ok / "magnitude_caches" / "mag_aggregate_step_50.pt").is_file()


def test_tie_break_streams_are_independent_across_arms(tmp_path):
    """RESULTS.md §3.4: the old builder drew the random mask and the warm tie-break from one
    seed-42 stream, so the 'independent' baseline shared coordinates with the warm masks."""
    base_path, final_path, deltas = _make_run(tmp_path)
    import torch as _t

    sd = _t.load(base_path, map_location="cpu", weights_only=True)
    ctx = cm.ScoreContext(names=["model.layers.0.mlp.gate_proj.weight"], initial_sd=sd, final_sd=sd)
    rnd = next(iter(cm.selection_chunks(ctx, "random", tie_break_scale_abs=0.0)))
    zeros = cm.ScoreContext(names=ctx.names, initial_sd={k: _t.zeros_like(v) for k, v in sd.items()}, final_sd={k: _t.zeros_like(v) for k, v in sd.items()})
    tie = next(iter(cm.selection_chunks(zeros, "oracle", tie_break_scale_abs=1.0)))
    # Independent streams: the rank correlation between them should be ~0, not 1.
    r = float(_t.corrcoef(_t.stack([rnd.double(), tie.double()]))[0, 1].item())
    assert abs(r) < 0.05, r
