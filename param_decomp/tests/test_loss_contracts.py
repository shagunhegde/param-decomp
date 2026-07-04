"""The loss-normalization contract registry (recon.Normalizer).

Each `LossTerm` variant declares the honest global count its value divides by; the step
divides by the DECLARED counts only. There are no manual gradient reductions on the JAX
line — autodiff of the global-mean loss absorbs them — so this count is the whole
device-count-invariance story per term (SPEC D4), and the recurring scaling-bug class is
a wrong/re-derived count, not a reduction. Three layers of enforcement:

1. declaration — adding a `LossTerm` variant without a `normalizer` fails here;
2. numeric agreement — the loss impls and the built plans divide by exactly the count
   the contract names (hand-computed tiny fixtures; sampler family sizes vs `n_draws`);
3. structure — `make_train_step` may not re-derive a forward tally, and the loss impls
   may not grow a hardcoded pool-size divisor.
"""

import inspect
import math
import typing

import jax
import jax.numpy as jnp
import numpy as np

import param_decomp.losses as losses_mod
import param_decomp.train as train_mod
from param_decomp.configs import (
    AnyLossMetricConfig,
    ChunkwiseSubsetReconLossConfig,
    FaithfulnessLossConfig,
    ImportanceMinimalityLossConfig,
    UniformKSubsetRoutingConfig,
)
from param_decomp.losses import (
    faithfulness_loss,
    importance_minimality_terms,
    kl_per_position,
)
from param_decomp.recon import (
    FaithfulnessTerm,
    ImportanceMinimalityTerm,
    LossTerm,
    Normalizer,
    ReconLossTerm,
    build_loss_terms,
)

_SITE_NAMES = ("s1", "s2", "s3", "s4")


def _surface(*recon_cfgs: "AnyLossMetricConfig"):
    return build_loss_terms(
        (
            FaithfulnessLossConfig(coeff=1.0),
            ImportanceMinimalityLossConfig(
                coeff=1.0,
                pnorm=1.0,
                p_anneal_start_frac=0.0,
                p_anneal_final_p=1.0,
                p_anneal_end_frac=1.0,
            ),
            *recon_cfgs,
        ),
        _SITE_NAMES,
    )


# ───────────────────────────── 1. declaration ─────────────────────────────


def test_every_loss_term_variant_declares_a_normalizer():
    """A new `LossTerm` variant must declare its normalization contract at the class —
    this is the loud failure the registry exists for."""
    variants = typing.get_args(LossTerm)
    assert len(variants) >= 3, "the LossTerm union lost members?"
    valid = set(typing.get_args(Normalizer))
    for variant in variants:
        declared = getattr(variant, "normalizer", None)
        assert declared in valid, (
            f"{variant.__name__} declares no valid normalizer (got {declared!r}) — every "
            f"LossTerm variant must pin its global-count contract (recon.Normalizer)"
        )


def test_declared_contracts_are_the_spec_ones():
    assert FaithfulnessTerm.normalizer == "global_param_numel"  # SPEC S17
    assert ImportanceMinimalityTerm.normalizer == "global_positions"  # SPEC S7/S8
    assert ReconLossTerm.normalizer == "plan_forwards"  # SPEC S10/S10'


# ───────────────────────────── 2. numeric agreement ─────────────────────────────


def test_recon_declared_n_forwards_matches_plan_and_samplers():
    """`n_forwards` (the applied divisor) equals the plan structure the configs asked
    for, and each entry's sampler really yields its declared `n_draws` routing draws."""
    from param_decomp.configs import (
        StochasticReconLayerwiseLossConfig,
        StochasticReconLossConfig,
    )

    surface = _surface(
        StochasticReconLossConfig(coeff=1.0, n_mask_samples=3),
        StochasticReconLayerwiseLossConfig(coeff=1.0, n_mask_samples=2),
        ChunkwiseSubsetReconLossConfig(
            routing=UniformKSubsetRoutingConfig(), coeff=1.0, sites_per_chunk=2, n_samples=2
        ),
    )
    by_name = {t.name: t for t in surface.recon}
    assert by_name["StochasticReconLoss"].n_forwards == 3  # 1 chunk x 3 draws
    assert by_name["StochasticReconLayerwiseLoss"].n_forwards == 8  # 4 sites x 2 draws
    assert by_name["ChunkwiseSubsetReconLoss"].n_forwards == 4  # 2 chunks x 2 draws

    key = jax.random.PRNGKey(0)
    for term in surface.recon:
        for entry in term.plan:
            assert len(entry.sample_routing(key, (2, 4))) == entry.n_draws, term.name


def test_kl_per_position_divides_by_global_positions():
    """`global_positions` for the recon comparison: total KL over a `(2, 3)` leading grid
    equals the mean of the 6 per-position KLs."""
    key1, key2 = jax.random.split(jax.random.PRNGKey(0))
    clean = jax.random.normal(key1, (2, 3, 7))
    masked = clean + 0.3 * jax.random.normal(key2, (2, 3, 7))
    per_position = [
        kl_per_position(masked[b, t][None, :], clean[b, t][None, :])
        for b in range(2)
        for t in range(3)
    ]
    expected = float(np.mean([float(v) for v in per_position]))
    assert math.isclose(float(kl_per_position(masked, clean)), expected, rel_tol=1e-6)


def test_faithfulness_divides_by_global_param_numel():
    """`global_param_numel`: one Σ‖Δ‖² over ALL sites divided by the SUMMED numel — not a
    per-site mean of means (which would weight small sites up)."""
    k1, k2 = jax.random.split(jax.random.PRNGKey(1))
    deltas = {
        "a": jax.random.normal(k1, (3, 5)),
        "b": jax.random.normal(k2, (2, 7)),
    }
    expected = (float((deltas["a"] ** 2).sum()) + float((deltas["b"] ** 2).sum())) / (15 + 14)
    assert math.isclose(float(faithfulness_loss(deltas)), expected, rel_tol=1e-6)


def test_imp_min_divides_by_global_positions_per_component():
    """`global_positions` for imp-min: with `pnorm=1, eps=0` the term is exactly
    `Σ_s Σ_c (Σ_positions ci) / n_positions` — the per-component mean over the GLOBAL
    position count, then summed over components and sites (never a per-site mean-of-C)."""
    k1, k2 = jax.random.split(jax.random.PRNGKey(2))
    ci = {
        "a": jax.random.uniform(k1, (2, 4, 3)),
        "b": jax.random.uniform(k2, (2, 4, 5)),
    }
    n_positions = 2 * 4
    expected = (float(ci["a"].sum()) + float(ci["b"].sum())) / n_positions
    lp, freq = importance_minimality_terms(ci, jnp.asarray(1.0), 0.0, None)
    assert math.isclose(float(lp), expected, rel_tol=1e-6)
    assert float(freq) == 0.0


# ───────────────────────────── 3. structure ─────────────────────────────


def test_structure_step_divides_by_declared_counts_only():
    """`make_train_step` reads `term.n_forwards` / `entry.n_draws` (the contract) and
    checks the samplers against them; a reintroduced runtime tally or an inline pool-size
    divisor is exactly the recurring scaling-bug class this registry closes."""
    src = inspect.getsource(train_mod.make_train_step)
    assert "/ term.n_forwards" in src, "term loss must divide by the DECLARED forward count"
    assert "/ entry.n_draws" in src, "ascent objective must divide by the DECLARED draw count"
    assert "len(routes_per_draw) == entry.n_draws" in src, (
        "every draw site must assert the sampler still matches the declaration"
    )
    assert "n_forwards += 1" not in src and "n_forwards = 0" not in src, (
        "the forward count must be declared plan structure, never a runtime tally"
    )


def test_structure_no_hardcoded_pool_divisors_in_loss_impls():
    """The loss impls divide by the counts their contract names — computed from the data
    (`math.prod(shape[:-1])`, `Σ numel`), never a hardcoded topology/pool constant."""
    kl_src = inspect.getsource(losses_mod.kl_per_position)
    assert "math.prod(masked_output.shape[:-1])" in kl_src
    faith_src = inspect.getsource(losses_mod.faithfulness_loss)
    assert "sum(delta.size for delta in weight_deltas.values())" in faith_src
    imp_src = inspect.getsource(losses_mod._imp_min_terms)
    assert "math.prod(ci.shape[:-1])" in imp_src
    for src in (kl_src, faith_src, imp_src, inspect.getsource(train_mod.make_train_step)):
        for stale_divisor in ("n_ci", "n_per_block", "n_pool"):
            assert stale_divisor not in src, f"pool-size divisor {stale_divisor!r} reappeared"
