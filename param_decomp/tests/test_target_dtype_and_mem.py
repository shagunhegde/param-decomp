"""Dtype/upcast guard on the frozen-target load path + AOT peak-memory regression gate.

The dtype guard pins the issue-#559 failure class end-to-end: the pretrain-cache
safetensors are fp32 on disk, so a loader that stops converting (drops the `dtype=` in
`_checkpoint_weight_getter`) silently doubles the frozen target's resident footprint
(32 GB instead of 16 GB/rank at 8B — an OOM found by a human crash, now a CI failure).

The memory gate wraps the engine's allocation-free AOT probe (`jit_util.aot_memory`, the
`profile.mem_profile` formula: peak = temp + argument + output − alias from XLA's static
`memory_analysis()`): it lowers + compiles the real train step for a tiny LlamaSimpleMLP
topology and compares against a committed per-backend baseline, so a silent upcast or an
XLA-side blow-up in the step's live set fails CI without a GPU. Baselines live in
`mem_baselines.json`; regenerate by running the baseline test with
`PD_UPDATE_MEM_BASELINES=1` after a deliberate memory-relevant change.
"""

import json
import os
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from safetensors.numpy import save_file

from param_decomp.components import SiteC, init_decomp_vu
from param_decomp.configs import (
    ChunkwiseSubsetReconLossConfig,
    FaithfulnessLossConfig,
    ImportanceMinimalityLossConfig,
    UniformKSubsetRoutingConfig,
)
from param_decomp.jit_util import aot_memory
from param_decomp.recon import build_loss_terms
from param_decomp.targets.llama_simple_mlp import (
    LlamaSimpleMLPConfig,
    canonical_site_cs,
    load_decomposed_lm_from_pretrain_cache,
    site_specs,
)
from param_decomp.tests.test_llama_simple_mlp import (
    _build_chunkwise_ci_fn,
    _tiny_cfg,
    _tiny_decomposed_model,
)
from param_decomp.train import TrainState, make_train_step

MEM_BASELINES_PATH = Path(__file__).parent / "mem_baselines.json"
MEM_RTOL = 0.10


def _write_synthetic_fp32_cache(cache_dir: Path, cfg: LlamaSimpleMLPConfig) -> None:
    """A tiny pretrain cache with the real checkpoint key layout, all weights fp32 on disk
    (matching the torch-converted safetensors the production cache holds)."""
    rng = np.random.default_rng(0)
    d, di = cfg.n_embd, cfg.n_intermediate
    qd, kvd = cfg.n_head * cfg.head_dim, cfg.n_kv_head * cfg.head_dim

    def w(*shape: int) -> np.ndarray:
        return rng.standard_normal(shape).astype(np.float32) * d**-0.5

    weights = {"wte.weight": w(cfg.vocab_size, d), "ln_f.weight": np.ones(d, np.float32)}
    for i in range(cfg.n_layer):
        weights |= {
            f"h.{i}.rms_1.weight": np.ones(d, np.float32),
            f"h.{i}.rms_2.weight": np.ones(d, np.float32),
            f"h.{i}.attn.q_proj.weight": w(qd, d),
            f"h.{i}.attn.k_proj.weight": w(kvd, d),
            f"h.{i}.attn.v_proj.weight": w(kvd, d),
            f"h.{i}.attn.o_proj.weight": w(d, qd),
            f"h.{i}.mlp.c_fc.weight": w(di, d),
            f"h.{i}.mlp.down_proj.weight": w(d, di),
        }
    save_file(weights, str(cache_dir / "model_step_100.safetensors"))


@pytest.mark.parametrize("requested_dtype", [jnp.bfloat16, jnp.float32], ids=["bf16", "fp32"])
def test_pretrain_cache_load_honors_requested_dtype(tmp_path: Path, requested_dtype: jnp.dtype):
    """Every frozen array leaf lands in the REQUESTED dtype regardless of the on-disk dtype
    (fp32) — the bf16 case is the #559 upcast guard. `inv_freq` is the one deliberate fp32
    buffer (RoPE frequencies, never checkpoint-loaded)."""
    cfg = _tiny_cfg()
    _write_synthetic_fp32_cache(tmp_path, cfg)
    sites = site_specs(
        cfg, canonical_site_cs((SiteC("h.1.attn.q_proj", 4), SiteC("h.0.mlp.c_fc", 4)))
    )

    lm = load_decomposed_lm_from_pretrain_cache(tmp_path, cfg, sites, requested_dtype)

    assert lm.inv_freq.dtype == jnp.float32
    leaves = jax.tree_util.tree_leaves_with_path(eqx.filter(lm, eqx.is_inexact_array))
    assert len(leaves) > 8 * cfg.n_layer, "leaf walk missed the layer stack"
    wrong = {
        jax.tree_util.keystr(path): str(leaf.dtype)
        for path, leaf in leaves
        if "inv_freq" not in jax.tree_util.keystr(path) and leaf.dtype != requested_dtype
    }
    assert not wrong, f"frozen leaves not in requested {requested_dtype.__name__}: {wrong}"


def _tiny_train_step_lowering_args():
    """The real `make_train_step` over the tiny mixed-site LlamaSimpleMLP topology
    (same fixture as `test_step_trains_and_has_vpd_signature`, minus PPGD so the
    lowering is adversary-free and fast)."""
    cfg = _tiny_cfg()
    site_cs = (SiteC("h.2.attn.q_proj", 8), SiteC("h.2.mlp.c_fc", 8))
    sites = site_specs(cfg, canonical_site_cs(site_cs))
    lm = _tiny_decomposed_model(cfg, sites, jax.random.PRNGKey(0))
    vu = init_decomp_vu(sites, jax.random.PRNGKey(1))
    ci_fn = _build_chunkwise_ci_fn(lm, jax.random.PRNGKey(2))
    opt_vu = optax.chain(optax.clip_by_global_norm(0.01), optax.adamw(1e-3, weight_decay=0.0))
    opt_ci = optax.adamw(1e-3, weight_decay=0.0)
    state = TrainState(
        components=vu, ci_fn=ci_fn,
        components_opt_state=opt_vu.init(eqx.filter(vu, eqx.is_array)),
        ci_fn_opt_state=opt_ci.init(eqx.filter(ci_fn, eqx.is_array)),
        adversaries={}, step=jnp.zeros((), jnp.int32),
    )  # fmt: skip
    loss_terms = build_loss_terms(
        (
            FaithfulnessLossConfig(coeff=1e5),
            ImportanceMinimalityLossConfig(
                coeff=5e-6,
                pnorm=2.0,
                p_anneal_start_frac=0.0,
                p_anneal_final_p=0.4,
                p_anneal_end_frac=1.0,
            ),
            ChunkwiseSubsetReconLossConfig(
                routing=UniformKSubsetRoutingConfig(), coeff=0.5, sites_per_chunk=2, n_samples=1
            ),
        ),
        lm.site_names,
    )
    step = make_train_step(
        lm=lm, losses=loss_terms, components_optimizer=opt_vu, ci_fn_optimizer=opt_ci,
        total_steps=100, remat_recon_forwards=True, remat_ci_fn=False, mesh=None,
    )  # fmt: skip
    tokens = jax.random.randint(jax.random.PRNGKey(4), (2, 16), 0, cfg.vocab_size)
    return step, lm, state, tokens, jax.random.PRNGKey(5)


def test_aot_probe_reports_real_argument_bytes():
    """The probe machinery itself: static argument bytes equal the traced input leaves'
    nbytes exactly, so a silent dtype promotion anywhere in the step's inputs (frozen
    target, V/U, CI fn, batch) moves this number and the baseline below."""
    step, lm, state, tokens, key = _tiny_train_step_lowering_args()
    mem = aot_memory(step, lm, state, tokens, key)

    traced_input_bytes = sum(
        leaf.nbytes
        for leaf in jax.tree_util.tree_leaves(eqx.filter((lm, state, tokens, key), eqx.is_array))
    )
    assert mem.argument_bytes == traced_input_bytes, (mem.argument_bytes, traced_input_bytes)
    assert mem.output_bytes > 0 and mem.temp_bytes > 0
    assert (
        mem.peak_bytes == mem.temp_bytes + mem.argument_bytes + mem.output_bytes - mem.alias_bytes
    )


@pytest.mark.slow
def test_aot_peak_memory_within_baseline():
    """Static per-device peak of the tiny train step vs the committed per-backend baseline
    (±10%). Catches silent upcasts and XLA live-set regressions before a GPU ever OOMs.
    On a deliberate memory-relevant change, regenerate with PD_UPDATE_MEM_BASELINES=1."""
    step, lm, state, tokens, key = _tiny_train_step_lowering_args()
    mem = aot_memory(step, lm, state, tokens, key)
    backend = jax.default_backend()

    baselines = json.loads(MEM_BASELINES_PATH.read_text())
    key_name = "llama_simple_mlp_tiny_step"
    if os.environ.get("PD_UPDATE_MEM_BASELINES"):
        baselines.setdefault(key_name, {})[backend] = mem.peak_bytes
        MEM_BASELINES_PATH.write_text(json.dumps(baselines, indent=2) + "\n")
        pytest.skip(f"baseline updated: {key_name}/{backend} = {mem.peak_bytes}")
    if backend not in baselines[key_name]:
        pytest.skip(f"no {backend} baseline for {key_name}; run with PD_UPDATE_MEM_BASELINES=1")

    expected = baselines[key_name][backend]
    assert abs(mem.peak_bytes - expected) <= MEM_RTOL * expected, (
        f"AOT peak {mem.peak_bytes}B deviates >{MEM_RTOL:.0%} from the {backend} baseline "
        f"{expected}B (argument={mem.argument_bytes} temp={mem.temp_bytes} "
        f"output={mem.output_bytes} alias={mem.alias_bytes})"
    )
