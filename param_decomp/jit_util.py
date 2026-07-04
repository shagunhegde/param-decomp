"""`eqx.filter_jit` + `compiler_options` passthrough.

`equinox.filter_jit` forwards `**jitkwargs` to `jax.jit` at runtime, but its typed
`@overload`s expose only `fun` + `donate` — so passing `compiler_options` (the native,
in-process way to set XLA compiler flags, and the way they enter the compile-cache key)
fails basedpyright with "no overloads match". This wrapper centralizes that one cast so
call sites stay clean and typed.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import equinox as eqx

DonateMode = Literal["all", "all-except-first", "warn", "warn-except-first", "none"]


@dataclass(frozen=True)
class AOTMemory:
    """XLA's static `memory_analysis()` of a compiled step, in bytes per device."""

    argument_bytes: int
    output_bytes: int
    temp_bytes: int
    alias_bytes: int

    @property
    def peak_bytes(self) -> int:
        return self.temp_bytes + self.argument_bytes + self.output_bytes - self.alias_bytes


def aot_memory(jitted: Any, *args: Any) -> AOTMemory:
    """Allocation-free peak-memory probe: lower + compile `jitted(*args)` and read XLA's
    static memory analysis, without executing the computation. `jitted` is a `jax.jit` /
    `eqx.filter_jit` object (typed `Any` because the filter_jit Callable alias hides
    `.lower`)."""
    compiled = jitted.lower(*args).compile()
    analysis = getattr(compiled, "compiled", compiled).memory_analysis()
    return AOTMemory(
        argument_bytes=analysis.argument_size_in_bytes,
        output_bytes=analysis.output_size_in_bytes,
        temp_bytes=analysis.temp_size_in_bytes,
        alias_bytes=analysis.alias_size_in_bytes,
    )


def filter_jit[**P, T](
    fn: Callable[P, T],
    *,
    donate: DonateMode = "none",
    compiler_options: dict[str, bool | int | str] | None = None,
) -> Callable[P, T]:
    """`eqx.filter_jit(fn, donate=…, compiler_options=…)` — `compiler_options` is forwarded
    to `jax.jit` (XLA compiler flags, native + in the compile-cache key). `None` = no
    options. CPU backends accept and ignore GPU flags."""
    return eqx.filter_jit(fn, donate=donate, compiler_options=compiler_options)  # pyright: ignore[reportCallIssue]
