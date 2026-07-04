# Testing & regression-tooling branches

**Context.** These four branches were built as the test substrate for a MATS project proposal to Lee Sharkey's stream: *Amortised Adversarial Sampling for Parameter Decomposition* — replacing VPD's per-step PGD adversary with a co-trained network that predicts near-worst-case ablation sources in a single forward pass, sitting in a minimax loop with the decomposition. That experiment will modify exactly the most fragile parts of the trainer (the `MaskSourceStrategy` union, `TrainState.adversaries`, the fused backward that feeds both players of the minimax game, and the PGD machinery used as the evaluation baseline). Before touching any of that, the codebase needed: a frozen, gradient-level pin on the current PGD/PPGD semantics (the comparison baseline must not drift while the new adversary is built), structural enforcement of loss-term normalization (a mis-scaled term would masquerade as "amortisation collapse"), automated crash coverage of the composition roots, and a GPU-free way to quantify the memory claim. Each branch is independently reviewable and mergeable.

All four branches are cut from **`feature/jax`** (the JAX-migration integration branch, PR #560) — none depend on each other or on `main`.

### 1. `feature/smoke-config-harness`
**Auto-discovered SMOKE-config composition-root harness.** `param_decomp_lab/tests/test_smoke_configs.py` globs every `experiments/*/configs/*_SMOKE.yaml` and runs it end-to-end through its family's real `run.py::main` on CPU against a tmpdir `PARAM_DECOMP_OUT_DIR`: YAML schema → `BuiltRun` → pretrain → faith warmup → engine step loop → checkpoint → in-loop eval. Asserts completion, the family's eval keys (`eval/identity_ci_error/*` for the toys) landing in `metrics.jsonl`, finite scalars, and a checkpoint on disk. New experiment families get coverage automatically by shipping a SMOKE config; runs in ~18 s inside `make test`. **It caught a live bug on its first run:** both toy composition roots' eval closures still called `ci_fn(taps)` without the now-required keyword-only `remat` argument (the same wiring-rot class as #928) — nothing in CI executed those paths. Both call sites fixed in this branch. For the amortised-adversary work this is the fast iteration loop: a `*_amortised_SMOKE.yaml` dropped into a toy family's configs gets CI crash-coverage of the entire new wiring chain for free.

### 2. `feature/dtype-mem-guard`
**Frozen-target dtype guard + allocation-free peak-memory gate.** (a) `test_target_dtype_and_mem.py` writes a synthetic fp32 safetensors pretrain cache and loads it through the real `load_decomposed_lm_from_pretrain_cache` path, asserting every frozen leaf lands in the *requested* dtype — the #559 class (silent fp32 upcast → 2× resident target → OOM) as a CI failure instead of a human crash. Verified to bite: dropping the loader's `dtype=` conversion turns the bf16 case red. (b) The engine's `PD_MEM` probe formula is extracted into `jit_util.aot_memory` (`peak = temp + argument + output − alias` from XLA's static `memory_analysis()`, never executing the step) and shared by `run.py` and a slow-tier test that pins the real train step's AOT peak against a committed per-backend baseline (`mem_baselines.json`, ±10%, regen via `PD_UPDATE_MEM_BASELINES=1`). Static argument bytes equal traced-input nbytes exactly, so dtype regressions are visible on CPU with no GPU. For the amortised adversary, `aot_memory` also quantifies the proposal's core scaling claim (single-forward adversary vs PGD inner loop) at lowering time.

### 3. `feature/loss-normalizer-contract`
**Declarative normalization contract on the loss surface.** On the JAX line there are no manual gradient reductions — autodiff of the global-mean loss absorbs them — so the recurring scaling-bug class is a wrong or re-derived *global count*. Each `LossTerm` variant now declares its count as a `ClassVar` (`recon.Normalizer`): `FaithfulnessTerm → global_param_numel` (S17), `ImportanceMinimalityTerm → global_positions` (S7/S8), `ReconLossTerm → plan_forwards` (S10′). `ReconForward` carries its routing sampler's declared family size (`n_draws`); the term's `n_forwards` derives from its plan. `make_train_step` divides by the **declared** counts only (the runtime `n_forwards += 1` tally is deleted) and asserts the samplers match the declaration at every draw site. `tests/test_loss_contracts.py` enforces it on three levels: a new `LossTerm` variant without a normalizer fails CI; the loss impls and built plans numerically divide by exactly the declared counts (hand-computed fixtures); and `inspect.getsource` guards reject reintroduced runtime tallies or pool-size divisors. Both gates verified to bite. This is the branch the amortised adversary leans on most directly: the new adversary term enters as exactly the kind of union member this registry now gates, and a normalization bug on either side of the minimax would otherwise silently rescale one player.

### 4. `feature/grad-golden-regen`
**Torch↔JAX value + gradient goldens regenerated on the token contract, plus a multidevice trajectory golden.** The `tests/equivalence/` harness had been xfail since fixtures moved from a residual-fed to a token-id model contract. This branch: migrates the fixtures (`tokens` + `embed`; both frameworks derive `resid = embed[tokens]`, an exact integer gather), regenerates `torch_reference.json` from the `torch-oracle` tag in a worktree, and flips every xfail. **New:** per-term *gradient* goldens (`torch_reference_grads.npz`, 90 arrays) — ∂term/∂{V,U}, the `ci_lower`/`ci_upper` CI-fn boundary cotangents, and the PPGD source gradient that drives the S13/S14 ascent — all matching torch entrywise at fp32 tolerance, with a sensitivity test proving the gate catches an off-by-one in a term's forward count. The regen recipe is committed (`gen_reference_token_contract.py.txt`; full steps in the test docstring). The permanently-stale `tests/stacked_parity/` harness (its generator targeted a deleted pre-restructure JAX branch) is replaced by `tests/trajectory_golden/`: a frozen self-regression golden of the trainer's forwards + 2-step grad-exercising trajectory (per-step losses, `grad_norms/summary/*`, final V/U + PPGD sources), asserted at 1 device in the default suite **and re-asserted GSPMD-sharded over the 4-sim-CPU-device mesh** via the existing `multidevice` marker — so device-count/normalization regressions fail ordinary CI, not a booked multi-GPU node. Also fixes `experiments/invariance_check.py`, which had been silently broken since the embed-internal migration (now green at 4 sim devices, worst rel 6.8e-6). For the amortised-adversary experiment this branch freezes the PGD/PPGD baseline: refactoring `adversary.py`/`train.py` to admit a third adversary type cannot silently perturb the semantics the comparison depends on.

**Merge Notes:** 

- branches 3 and 4 both touch `param_decomp/tests/equivalence/test_equivalence.py`; whichever merges second has a one-line conflict (keep branch 4's file with branch 3's `"/ term.n_forwards"` structural assertion).
- `feature/loss-normalizer-contract` and `feature/grad-golden-regen` both touch `test_equivalence.py` (the contract branch changes one structural assertion to `"/ term.n_forwards"`; the golden branch rewrote the file's header). Whichever merges second gets a small, mechanical conflict — resolution is: keep the golden branch's file, with the contract branch's `"/ term.n_forwards"` assertion.
- The torch-oracle worktree at `../torch-oracle-wt` (with its patched pyproject + torch venv) can be deleted, or kept for future golden regens.

---


# Parameter Decomposition

Training tools for parameter decomposition on neural networks. For a compact implementation of
the core method, see [`nano_param_decomp/`](nano_param_decomp/).

## References

- **VPD paper (April 2026):** https://www.goodfire.ai/research/interpreting-lm-parameters. [VPD Code Release](https://github.com/goodfire-ai/param-decomp/releases/tag/vpd-paper)
  Canonical 4L-pile run: `goodfire/spd/runs/s-55ea3f9b`.
- **SPD paper (June 2025):** https://arxiv.org/abs/2506.20790. [SPD Code Release](https://github.com/goodfire-ai/param-decomp/releases/tag/v1).

## Install

This repo contains two Python distributions:

- `param-decomp`: the core library, importing as `param_decomp`
- `param-decomp-lab`: in-repo experiments, app, postprocessing, and CLI tooling, importing as
  `param_decomp_lab`

```bash
make install-dev  # workspace dev install: core + lab + dev dependencies + pre-commit hooks
make install      # core package only
make install-lab  # core + lab packages, without dev dependencies
```

## Run Experiments

The `pd-*` commands are installed by `param-decomp-lab`. Each in-repo experiment is a
self-contained script that reads a YAML and calls `optimize()`:

```bash
pd-tms       param_decomp_lab/experiments/tms/tms_5-2_config.yaml
pd-resid-mlp param_decomp_lab/experiments/resid_mlp/resid_mlp1_config.yaml
pd-lm        param_decomp_lab/experiments/lm/pile_llama_simple_mlp-4L.yaml
```

For a brand-new experiment, write your own `run.py` that builds the target model, the
train/eval dataloaders, the eval `Metric` list, the `PDConfig` and `RuntimeConfig`, a
`Cadence` (when to emit), and a `RunSink` (where output goes), then calls `optimize(...)`:

```python
from param_decomp.configs import Cadence, PDConfig, RuntimeConfig
from param_decomp.optimize import EvalLoop, optimize
from param_decomp_lab.batch_and_loss_fns import recon_loss_mse, run_batch_first_element
from param_decomp_lab.run_sink import RunSink

optimize(
    target_model=my_target_module,
    train_loader=train_loader,
    run_batch=run_batch_first_element,
    reconstruction_loss=recon_loss_mse,
    pd_config=PDConfig(...),
    runtime_config=RuntimeConfig(device=device),
    cadence=Cadence(
        train_log_every=100,
        save_every=5000,
    ),
    sink=RunSink.local(out_dir),
    eval_loop=EvalLoop(
        loader=eval_loader,
        metrics=[...],  # list of pre-instantiated Metric objects
        n_steps=10,
        every=1000,
        slow_every=5000,
    ),
)
```

The three in-repo `run.py` files
([tms](param_decomp_lab/experiments/tms/run.py),
 [resid_mlp](param_decomp_lab/experiments/resid_mlp/run.py),
 [lm](param_decomp_lab/experiments/lm/run.py)) are reference examples.

## Metrics

Configure training losses in `pd.loss_metrics` as a list of `{type: "<ClassName>", ...}`
entries. The `type` literal dispatches to a `Metric` subclass via
`param_decomp.metrics.dispatch.LOSS_METRIC_CLASSES`. Loss metrics must set `coeff`; they
are evaluated automatically alongside dedicated eval metrics. New loss metrics are added
by defining the class in `param_decomp/metrics/`, appending the config to
`AnyLossMetricConfig` in `configs.py`, and appending the class to `LOSS_METRIC_CLASSES`.

Eval metrics are caller-supplied: instantiate `Metric` objects in your `run.py` and pass
them via `EvalLoop(metrics=...)`. The in-repo experiments validate the YAML
`eval.metrics` list via the `AnyEvalMetricConfig` discriminated union on `EvalConfig`,
then dispatch each entry through `EVAL_METRIC_CLASSES` (both in
`param_decomp_lab.eval_metrics`):

```python
eval_metrics = [EVAL_METRIC_CLASSES[m.type](m) for m in cfg.eval.metrics]
```

## Packaging

The root `pyproject.toml` builds only the core `param-decomp` distribution. Lab scripts
and experiment tooling live in `param_decomp_lab/pyproject.toml` as the separate
`param-decomp-lab` distribution. Local development uses the uv workspace, so absolute
imports for both packages work after `make install-dev`.

Metric classes define a Pydantic config plus a class satisfying `__init__(cfg)`,
`bind(*, model, device)`, `reset()`, `update(ctx)`, and `compute()`. Use `LossMetricConfig`
for trainable losses and subclass `BaseConfig` directly for eval-only metrics; see
[`param_decomp/metrics/base.py`](param_decomp/metrics/base.py).

## Development

```bash
make check     # ruff format/lint + basedpyright
make type      # basedpyright only
make format    # ruff lint + format
make test      # tests not marked slow
make test-all  # all tests
```
