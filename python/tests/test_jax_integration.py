"""Live end-to-end `jax` integration suite executing against `libtpu-rs` (`PJRT` C ABI).

Validates numerical parity against the `jax` CPU reference backend across:
1. `jnp.matmul` and `bfloat16` / `float32` matrix multiplication
2. `lax.dot_general` batched tensor contractions
3. `lax.conv_general_dilated` 2D spatial convolutions
4. `jax.jit`, `jax.grad`, and `jax.vmap` composable transformations
5. `lax.scan`, `lax.while_loop`, and `lax.cond` structured control flow
6. Multi-device `jax.experimental.shard_map` (`AllReduce`, `AllGather`, `ReduceScatter`, `AllToAll`)
7. `jax.experimental.pallas.pallas_call` Mosaic TPU kernels
8. `jax.pure_callback` host-device continuation callbacks
9. Multi-step Transformer self-attention + MLP `value_and_grad` training steps
10. `jax.profiler` live `XPlane` (`*.xplane.pb`) trace capture
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import libtpu

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")


def _detect_jax_pjrt_minor_version(lib_path: str) -> str:
    """Detects the PJRT C API minor version expected by the installed `jaxlib` runtime.

    Args:
        lib_path: Absolute filesystem path to the compiled `libtpu` shared library.

    Returns:
        Minor version string (e.g. `"54"` for JAX 0.4.x or `"117"` for OpenXLA PJRT v0.117).
    """
    probe_env = os.environ.copy()
    probe_env.pop("LIBTPU_PJRT_API_MINOR_VERSION", None)
    probe_env["PJRT_NAMES_AND_LIBRARY_PATHS"] = f"tpu:{lib_path}"
    probe_env["TPU_LIBRARY_PATH"] = lib_path
    probe_env["JAX_PLATFORMS"] = "tpu,cpu"
    proc = subprocess.run(
        [sys.executable, "-c", "import jax; jax.devices('tpu')"],
        env=probe_env,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode == 0:
        return "117"
    combined = f"{proc.stdout}\n{proc.stderr}"
    match = re.search(r"framework PJRT API version 0\.(\d+)", combined)
    return match.group(1) if match is not None else "54"


def _make_shard_map(fn: Any, *, mesh: Any, in_specs: Any, out_specs: Any) -> Any:
    """Wraps `fn` with `jax.shard_map` (JAX 0.6+) or `jax.experimental.shard_map.shard_map`.

    Args:
        fn: Per-shard callable executed across `mesh`.
        mesh: Multi-device `jax.sharding.Mesh`.
        in_specs: Input `PartitionSpec`.
        out_specs: Output `PartitionSpec`.

    Returns:
        Shard-mapped callable targeting the active `jax` release.
    """
    top_shard_map = getattr(jax, "shard_map", None)
    if top_shard_map is not None:
        return top_shard_map(
            fn,
            mesh=mesh,
            in_specs=in_specs,
            out_specs=out_specs,
            check_vma=False,
        )
    shard_map_mod: Any = pytest.importorskip("jax.experimental.shard_map")
    return shard_map_mod.shard_map(
        fn,
        mesh=mesh,
        in_specs=in_specs,
        out_specs=out_specs,
        check_rep=False,
    )


@pytest.fixture(scope="module", autouse=True)
def _configure_jax_tpu_plugin_env() -> Iterator[None]:
    """Configures `PJRT_NAMES_AND_LIBRARY_PATHS` for `jax` and restores env vars on teardown."""
    lib_path = libtpu.configure_library_path()
    saved = {
        key: os.environ.get(key)
        for key in (
            "LIBTPU_PJRT_API_MINOR_VERSION",
            "PJRT_NAMES_AND_LIBRARY_PATHS",
            "TPU_LIBRARY_PATH",
            "JAX_PLATFORMS",
        )
    }
    os.environ["LIBTPU_PJRT_API_MINOR_VERSION"] = _detect_jax_pjrt_minor_version(lib_path)
    os.environ["PJRT_NAMES_AND_LIBRARY_PATHS"] = f"tpu:{lib_path}"
    os.environ["TPU_LIBRARY_PATH"] = lib_path
    os.environ["JAX_PLATFORMS"] = "tpu,cpu"
    yield
    for key, val in saved.items():
        if val is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = val


def _tpu_and_cpu_devices() -> tuple[list[Any], Any]:
    """Returns `(tpu_devices, cpu_device_0)` from the initialized `jax` runtime.

    Returns:
        Tuple of the 4 TPU devices backed by `VirtualVfioGasketDevice` and the CPU reference device.
    """
    tpu_devs = list(jax.devices("tpu"))
    cpu_dev = jax.devices("cpu")[0]
    assert len(tpu_devs) == 4
    assert tpu_devs[0].platform == "tpu"
    assert "TPU" in str(tpu_devs[0].device_kind).upper()
    return tpu_devs, cpu_dev


def test_jax_matmul_dot_general_and_conv_general_dilated() -> None:
    """Verifies `jnp.matmul` (`f32`/`bf16`), `lax.dot_general`, and `lax.conv_general_dilated`."""
    from jax import lax

    with libtpu.TpuClient() as check_client:
        assert check_client.device_count() >= 4

    tpu_devs, cpu_dev = _tpu_and_cpu_devices()

    lhs_2d = np.arange(1, 7, dtype=np.float32).reshape(2, 3)
    rhs_2d = np.arange(1, 7, dtype=np.float32).reshape(3, 2)
    mm_tpu = jnp.matmul(jax.device_put(lhs_2d, tpu_devs[0]), jax.device_put(rhs_2d, tpu_devs[0]))
    mm_cpu = jnp.matmul(jax.device_put(lhs_2d, cpu_dev), jax.device_put(rhs_2d, cpu_dev))
    np.testing.assert_allclose(np.asarray(mm_tpu), np.asarray(mm_cpu), rtol=1e-5, atol=1e-5)

    bf16_lhs = np.array([[1.0, 2.0, -0.5, 1.5], [0.5, -1.0, 2.0, 0.25]], dtype=np.float32)
    bf16_rhs = np.array([[0.5, -1.0], [1.5, 0.5], [-2.0, 1.0], [0.25, 2.0]], dtype=np.float32)
    bf16_mm = lambda a, b: jnp.matmul(a.astype(jnp.bfloat16), b.astype(jnp.bfloat16)).astype(  # noqa: E731
        jnp.float32
    )
    bf16_tpu = jax.jit(bf16_mm)(
        jax.device_put(bf16_lhs, tpu_devs[0]),
        jax.device_put(bf16_rhs, tpu_devs[0]),
    )
    bf16_cpu = jax.jit(bf16_mm)(
        jax.device_put(bf16_lhs, cpu_dev),
        jax.device_put(bf16_rhs, cpu_dev),
    )
    np.testing.assert_allclose(np.asarray(bf16_tpu), np.asarray(bf16_cpu), rtol=1e-2, atol=2e-2)

    lhs_3d = np.arange(1, 13, dtype=np.float32).reshape(2, 2, 3)
    rhs_3d = np.arange(1, 13, dtype=np.float32).reshape(2, 3, 2)
    dim_nums = (((2,), (1,)), ((0,), (0,)))
    dg_tpu = lax.dot_general(
        jax.device_put(lhs_3d, tpu_devs[0]),
        jax.device_put(rhs_3d, tpu_devs[0]),
        dim_nums,
    )
    dg_cpu = lax.dot_general(
        jax.device_put(lhs_3d, cpu_dev),
        jax.device_put(rhs_3d, cpu_dev),
        dim_nums,
    )
    np.testing.assert_allclose(np.asarray(dg_tpu), np.asarray(dg_cpu), rtol=1e-5, atol=1e-5)

    img = np.arange(1, 17, dtype=np.float32).reshape(1, 4, 4, 1)
    kern = np.array([[1.0, 0.5], [-0.5, 2.0]], dtype=np.float32).reshape(2, 2, 1, 1)
    conv_dn = ("NHWC", "HWIO", "NHWC")
    cv_tpu = lax.conv_general_dilated(
        jax.device_put(img, tpu_devs[0]),
        jax.device_put(kern, tpu_devs[0]),
        (1, 1),
        "VALID",
        dimension_numbers=conv_dn,
    )
    cv_cpu = lax.conv_general_dilated(
        jax.device_put(img, cpu_dev),
        jax.device_put(kern, cpu_dev),
        (1, 1),
        "VALID",
        dimension_numbers=conv_dn,
    )
    np.testing.assert_allclose(np.asarray(cv_tpu), np.asarray(cv_cpu), rtol=1e-5, atol=1e-5)


def test_jax_jit_grad_vmap_and_structured_control_flow() -> None:
    """Verifies `jax.jit`, `jax.grad`, `jax.vmap`, `lax.scan`, `lax.while_loop`, and `lax.cond`."""
    from jax import lax

    tpu_devs, cpu_dev = _tpu_and_cpu_devices()

    def scalar_energy(vec: Any) -> Any:
        """Computes a smooth nonlinear scalar energy over `vec`.

        Args:
            vec: Input vector tensor.

        Returns:
            Scalar energy tensor.
        """
        return jnp.sum(jnp.sin(vec) * vec + vec * vec)

    batched_grad = jax.jit(jax.vmap(jax.grad(scalar_energy)))
    pts = np.array([[0.5, 1.0, 1.5], [2.0, -0.5, 0.25]], dtype=np.float32)
    g_tpu = batched_grad(jax.device_put(pts, tpu_devs[0]))
    g_cpu = jax.jit(jax.vmap(jax.grad(scalar_energy)))(jax.device_put(pts, cpu_dev))
    np.testing.assert_allclose(np.asarray(g_tpu), np.asarray(g_cpu), rtol=1e-4, atol=1e-4)

    def scan_step(carry: Any, elem: Any) -> tuple[Any, Any]:
        """Executes a single `lax.scan` accumulation step.

        Args:
            carry: Recurrent carry tensor.
            elem: Current sequence element tensor.

        Returns:
            Updated `(carry, output)` tensor pair.
        """
        nxt = carry + elem * 2.0
        return nxt, nxt

    xs = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
    init = np.array([0.5, -0.5], dtype=np.float32)
    c_tpu, y_tpu = jax.jit(lambda c, a: lax.scan(scan_step, c, a))(
        jax.device_put(init, tpu_devs[0]),
        jax.device_put(xs, tpu_devs[0]),
    )
    c_cpu, y_cpu = jax.jit(lambda c, a: lax.scan(scan_step, c, a))(
        jax.device_put(init, cpu_dev),
        jax.device_put(xs, cpu_dev),
    )
    np.testing.assert_allclose(np.asarray(c_tpu), np.asarray(c_cpu), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.asarray(y_tpu), np.asarray(y_cpu), rtol=1e-5, atol=1e-5)

    while_fn = lambda v0: lax.while_loop(lambda v: jnp.sum(v) < 20.0, lambda v: v + 3.0, v0)  # noqa: E731
    w_in = np.array([1.0, 2.0], dtype=np.float32)
    w_tpu = jax.jit(while_fn)(jax.device_put(w_in, tpu_devs[0]))
    w_cpu = jax.jit(while_fn)(jax.device_put(w_in, cpu_dev))
    np.testing.assert_allclose(np.asarray(w_tpu), np.asarray(w_cpu), rtol=1e-5, atol=1e-5)

    cond_fn = lambda pred, v: lax.cond(pred, lambda z: z * 3.0, lambda z: z - 5.0, v)  # noqa: E731
    c_in = np.array([2.0, 4.0], dtype=np.float32)
    for branch_flag in (True, False):
        br_tpu = jax.jit(cond_fn)(
            jax.device_put(branch_flag, tpu_devs[0]),
            jax.device_put(c_in, tpu_devs[0]),
        )
        br_cpu = jax.jit(cond_fn)(
            jax.device_put(branch_flag, cpu_dev),
            jax.device_put(c_in, cpu_dev),
        )
        np.testing.assert_allclose(np.asarray(br_tpu), np.asarray(br_cpu), rtol=1e-5, atol=1e-5)


def test_jax_shard_map_collectives_across_four_tpu_chips() -> None:
    """Verifies multi-device `shard_map` (`AllReduce`, `AllGather`, `ReduceScatter`, `AllToAll`)."""
    lax: Any = pytest.importorskip("jax.lax")
    sharding_mod: Any = pytest.importorskip("jax.sharding")
    mesh_cls = sharding_mod.Mesh
    named_sharding = sharding_mod.NamedSharding
    pspec = sharding_mod.PartitionSpec

    tpu_devs, _ = _tpu_and_cpu_devices()
    mesh = mesh_cls(np.array(tpu_devs), ("x",))
    mat = np.arange(1, 17, dtype=np.float32).reshape(4, 4)
    sharded_mat = jax.device_put(mat, named_sharding(mesh, pspec("x", None)))

    ar_fn = jax.jit(
        _make_shard_map(
            lambda v: lax.psum(v, "x"),
            mesh=mesh,
            in_specs=pspec("x", None),
            out_specs=pspec(None, None),
        )
    )
    np.testing.assert_allclose(
        np.asarray(ar_fn(sharded_mat)),
        np.sum(mat, axis=0, keepdims=True),
        rtol=1e-5,
    )

    ag_fn = jax.jit(
        _make_shard_map(
            lambda v: lax.all_gather(v, "x", axis=0, tiled=True),
            mesh=mesh,
            in_specs=pspec("x", None),
            out_specs=pspec(None, None),
        )
    )
    np.testing.assert_allclose(np.asarray(ag_fn(sharded_mat)), mat, rtol=1e-5)

    replicated_mat = jax.device_put(mat, named_sharding(mesh, pspec(None, None)))
    rs_fn = jax.jit(
        _make_shard_map(
            lambda v: lax.psum_scatter(v, "x", scatter_dimension=0, tiled=True),
            mesh=mesh,
            in_specs=pspec(None, None),
            out_specs=pspec("x", None),
        )
    )
    np.testing.assert_allclose(np.asarray(rs_fn(replicated_mat)), mat * 4.0, rtol=1e-5)

    a2a_fn = jax.jit(
        _make_shard_map(
            lambda v: lax.all_to_all(v, "x", split_axis=1, concat_axis=0, tiled=True),
            mesh=mesh,
            in_specs=pspec("x", None),
            out_specs=pspec("x", None),
        )
    )
    expected_a2a = np.concatenate([mat[:, col : col + 1] for col in range(4)], axis=0)
    np.testing.assert_allclose(np.asarray(a2a_fn(sharded_mat)), expected_a2a, rtol=1e-5)


def test_jax_pallas_mosaic_pure_callback_transformer_and_profiler(tmp_path: Path) -> None:
    """Verifies `pallas_call`, `pure_callback`, Transformer training step, and `XPlane` profiler.

    Args:
        tmp_path: Temporary directory for `jax.profiler` trace capture.
    """
    pl: Any = pytest.importorskip("jax.experimental.pallas")

    tpu_devs, cpu_dev = _tpu_and_cpu_devices()

    def mosaic_fused_fma_kernel(x_ref: Any, y_ref: Any, out_ref: Any) -> None:
        """Computes fused `x + 2 * y` inside a Pallas/Mosaic kernel.

        Args:
            x_ref: Input `x` scratchpad reference.
            y_ref: Input `y` scratchpad reference.
            out_ref: Output scratchpad reference.
        """
        out_ref[...] = x_ref[...] + y_ref[...] * 2.0

    px = np.full((8, 128), 3.0, dtype=np.float32)
    py = np.full((8, 128), 4.0, dtype=np.float32)
    pallas_tpu = pl.pallas_call(
        mosaic_fused_fma_kernel,
        out_shape=jax.ShapeDtypeStruct((8, 128), jnp.float32),
    )(jax.device_put(px, tpu_devs[0]), jax.device_put(py, tpu_devs[0]))
    pallas_cpu = pl.pallas_call(
        mosaic_fused_fma_kernel,
        out_shape=jax.ShapeDtypeStruct((8, 128), jnp.float32),
        interpret=True,
    )(jax.device_put(px, cpu_dev), jax.device_put(py, cpu_dev))
    np.testing.assert_allclose(np.asarray(pallas_tpu), np.asarray(pallas_cpu), rtol=1e-5)

    def with_host_callback(vec: Any) -> Any:
        """Applies a Python `jax.pure_callback` host round-trip inside JIT.

        Args:
            vec: Device input vector tensor.

        Returns:
            Post-processed device vector tensor.
        """
        cb_out = jax.pure_callback(
            lambda host_arr: (np.asarray(host_arr) * 3.0 + 1.0).astype(np.float32),
            jax.ShapeDtypeStruct(vec.shape, vec.dtype),
            vec,
        )
        return cb_out + 2.0

    cb_in = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    cb_tpu = jax.jit(with_host_callback)(jax.device_put(cb_in, tpu_devs[0]))
    cb_cpu = jax.jit(with_host_callback)(jax.device_put(cb_in, cpu_dev))
    np.testing.assert_allclose(np.asarray(cb_tpu), np.asarray(cb_cpu), rtol=1e-5)

    def transformer_loss(params: dict[str, Any], tokens: Any, targets: Any) -> Any:
        """Computes single-block Transformer MSE training loss.

        Args:
            params: Attention and MLP weight dictionary.
            tokens: Input token embedding tensor.
            targets: Target regression tensor.

        Returns:
            Scalar MSE loss tensor.
        """
        q = jnp.matmul(tokens, params["wq"])
        k = jnp.matmul(tokens, params["wk"])
        v = jnp.matmul(tokens, params["wv"])
        attn_weights = jax.nn.softmax(jnp.matmul(q, jnp.transpose(k)) * 0.5, axis=-1)
        hidden = jax.nn.relu(
            jnp.matmul(jnp.matmul(attn_weights, v) + tokens, params["w1"]) + params["b1"]
        )
        preds = jnp.matmul(hidden, params["w2"])
        return jnp.mean((preds - targets) ** 2)

    rng = np.random.default_rng(42)
    shapes = [
        ("wq", (4, 4)),
        ("wk", (4, 4)),
        ("wv", (4, 4)),
        ("w1", (4, 8)),
        ("b1", (8,)),
        ("w2", (8, 4)),
    ]
    init_params = {name: rng.normal(0.0, 0.2, shape).astype(np.float32) for name, shape in shapes}
    tok_np = rng.normal(0.0, 1.0, (3, 4)).astype(np.float32)
    tgt_np = rng.normal(0.0, 1.0, (3, 4)).astype(np.float32)

    tpu_params = jax.tree.map(lambda arr: jax.device_put(arr, tpu_devs[0]), init_params)
    cpu_params = jax.tree.map(lambda arr: jax.device_put(arr, cpu_dev), init_params)
    tpu_val_grad = jax.jit(jax.value_and_grad(transformer_loss))
    cpu_val_grad = jax.jit(jax.value_and_grad(transformer_loss))

    loss_tpu_0, grads_tpu = tpu_val_grad(
        tpu_params,
        jax.device_put(tok_np, tpu_devs[0]),
        jax.device_put(tgt_np, tpu_devs[0]),
    )
    loss_cpu_0, grads_cpu = cpu_val_grad(
        cpu_params,
        jax.device_put(tok_np, cpu_dev),
        jax.device_put(tgt_np, cpu_dev),
    )
    np.testing.assert_allclose(np.asarray(loss_tpu_0), np.asarray(loss_cpu_0), rtol=1e-5, atol=1e-5)
    for key in init_params:
        np.testing.assert_allclose(
            np.asarray(grads_tpu[key]),
            np.asarray(grads_cpu[key]),
            rtol=1e-5,
            atol=1e-5,
        )

    # Apply one SGD step on TPU and verify training loss strictly decreases.
    updated_tpu_params = jax.tree.map(lambda p, g: p - 0.1 * g, tpu_params, grads_tpu)
    loss_tpu_1, _ = tpu_val_grad(
        updated_tpu_params,
        jax.device_put(tok_np, tpu_devs[0]),
        jax.device_put(tgt_np, tpu_devs[0]),
    )
    assert float(np.asarray(loss_tpu_1)) < float(np.asarray(loss_tpu_0))

    trace_dir = tmp_path / "jax_trace"
    trace_dir.mkdir()
    jax.profiler.start_trace(str(trace_dir))
    _ = jnp.matmul(
        jax.device_put(np.ones((2, 2), dtype=np.float32), tpu_devs[0]),
        jax.device_put(np.ones((2, 2), dtype=np.float32), tpu_devs[0]),
    ).block_until_ready()
    jax.profiler.stop_trace()
    xplane_files = list(trace_dir.rglob("*.xplane.pb"))
    assert len(xplane_files) >= 1
    assert xplane_files[0].stat().st_size > 0
